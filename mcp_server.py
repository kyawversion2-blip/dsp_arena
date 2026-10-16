#!/usr/bin/env python3.11
"""
DSP Arena MCP Server
Exposes WAV processing, batch conversion, and binary classification tools.

Usage (stdio transport — registered in .mcp.json):
  python3.11 mcp_server.py

Tools:
  process_wav        — single WAV → DSP features
  batch_process      — folder of WAVs → training CSV
  classify_wav       — classify a WAV vs training CSV (binary or full)
  get_separation     — separation score between 2 groups
  list_wavs          — list WAVs + their labels
"""
import asyncio, csv, json, math, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

from ref_dsp import measure as dsp_measure

from path_policy import (
    MAX_CSV_BYTES,
    MAX_WAV_BYTES,
    PROJECT_ROOT,
    PathPolicyError,
    validate_directory,
    validate_read_path,
    validate_write_path,
)

DIR = os.path.dirname(os.path.abspath(__file__))
WAVS_DIR  = os.path.join(DIR, "wavs")
LABELS_CSV = os.path.join(DIR, "labels.csv")
PROJECT_ROOT_DIR = PROJECT_ROOT
WAV_INPUT_ROOT = Path(WAVS_DIR).resolve()
WAV_READ_ROOTS = (WAV_INPUT_ROOT,)
CSV_READ_ROOTS = (PROJECT_ROOT_DIR.resolve(),)
MCP_OUTPUT_ROOT = (PROJECT_ROOT_DIR / "dsp_out").resolve()
ALLOWED_WAV = {".wav"}
ALLOWED_CSV = {".csv"}


def _policy_failure(label: str, exc: Exception) -> dict:
    return {"ok": False, "error": f"{label}: {exc}"}


def _wav_relative(value: str) -> str:
    text = str(value)
    if text in {"wavs", "wavs/"}:
        return "."
    if text.startswith("wavs/"):
        return text[len("wavs/"):]
    return text


def _validate_mcp_dir(value: str) -> Path:
    text = str(value).strip()
    if text in {"", ".", "wavs", "wavs/"}:
        return WAV_INPUT_ROOT
    if text.startswith("wavs/"):
        text = text[len("wavs/"):]
    return validate_directory(text, allowed_roots=(WAV_INPUT_ROOT,), label="wav_dir")

app = Server("dsp-arena")

# ── helpers ─────────────────────────────────────────────────────────────────

def _load_labels(csv_path: str) -> dict[str, str]:
    if not csv_path or not os.path.exists(csv_path):
        return {}
    labels: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            return {}
        hl = [h.strip().lower() for h in header]
        fc = next((i for i, h in enumerate(hl) if h in ("file_name", "file", "filename")), 0)
        gc = next((i for i, h in enumerate(hl) if h in ("result", "group", "label")), 1)
        for row in reader:
            if len(row) <= max(fc, gc):
                continue
            fn = os.path.splitext(row[fc].strip())[0]
            grp = row[gc].strip().upper()
            if grp:
                labels[fn] = grp
    return labels


def _load_training_csv(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    FEATS = {"impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict = {}
            for k, v in row.items():
                kl = k.strip().lower()
                if kl in FEATS:
                    try:
                        parsed[kl] = float(v)
                    except (ValueError, TypeError):
                        parsed[kl] = float("nan")
                else:
                    parsed[k] = v.strip() if isinstance(v, str) else v
            parsed.setdefault("Group", row.get("Group", row.get("group", "?")))
            rows.append(parsed)
    return rows


def _centroid(rows: list[dict], feats: list[str]) -> dict[str, float]:
    c: dict[str, float] = {}
    for f in feats:
        vals = [r[f] for r in rows if isinstance(r.get(f), float) and math.isfinite(r[f])]
        c[f] = sum(vals) / len(vals) if vals else float("nan")
    return c


def _zscore_stats(rows: list[dict], feats: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for f in feats:
        vals = [r[f] for r in rows if isinstance(r.get(f), (int, float)) and math.isfinite(r.get(f, float("nan")))]
        if not vals:
            stats[f] = (0.0, 1.0)
            continue
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) or 1.0
        stats[f] = (mu, sd)
    return stats


def _euclidean(a: dict, b: dict, feats: list[str]) -> float:
    return math.sqrt(
        sum((a.get(f, 0.0) - b.get(f, 0.0)) ** 2
            for f in feats
            if math.isfinite(a.get(f, float("nan"))) and math.isfinite(b.get(f, float("nan"))))
    )


FEATS = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]

# ── tool definitions ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="process_wav",
            description=(
                "Process a single WAV file with the given DSP parameters and return features: "
                "impact, duration, peak_time, mel_mean_db, impact_freq_hz."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "wav_path": {"type": "string", "description": "Repository-relative path to WAV file"},
                    "sr":     {"type": "integer", "default": 44100,  "description": "Sample rate"},
                    "nfft":  {"type": "integer", "default": 4096,   "description": "FFT size (power of 2)"},
                    "hop":   {"type": "integer", "default": 1024,   "description": "Hop length"},
                    "topdb": {"type": "integer", "default": 35,     "description": "Top dB trim threshold"},
                    "win":   {"type": "number",  "default": 0.5,    "description": "Window seconds ±peak"},
                },
                "required": ["wav_path"],
            },
        ),
        types.Tool(
            name="batch_process",
            description=(
                "Process all WAVs in a directory with given DSP params and write a training CSV. "
                "Labels come from labels_csv. Optionally restrict to specific groups."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "wav_dir":    {"type": "string", "description": "Repository-relative WAV directory (default: wavs)"},
                    "labels_csv": {"type": "string", "description": "Repository-relative labels CSV path (default: labels.csv)"},
                    "output_csv": {"type": "string", "default": "training.csv", "description": "Repository-relative output under dsp_out/"},
                    "sr":         {"type": "integer", "default": 44100},
                    "nfft":       {"type": "integer", "default": 4096},
                    "hop":        {"type": "integer", "default": 1024},
                    "topdb":      {"type": "integer", "default": 35},
                    "win":        {"type": "number",  "default": 0.5},
                    "groups":     {"type": "array", "items": {"type": "string"}, "description": "e.g. ['A','B'] — only process these groups"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="classify_wav",
            description=(
                "Classify a new WAV file by comparing its DSP features to a training CSV. "
                "Uses centroid-based (nearest-mean) classifier in z-normalised feature space. "
                "Set groups=['A','B'] for binary classification."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "wav_path":     {"type": "string",  "description": "Repository-relative path to WAV to classify"},
                    "training_csv": {"type": "string",  "description": "Repository-relative path to training CSV"},
                    "groups":       {"type": "array",   "items": {"type": "string"}, "description": "Restrict to these groups (binary: 2 groups)"},
                    "sr":    {"type": "integer", "default": 44100},
                    "nfft":  {"type": "integer", "default": 4096},
                    "hop":   {"type": "integer", "default": 1024},
                    "topdb": {"type": "integer", "default": 35},
                    "win":   {"type": "number",  "default": 0.5},
                },
                "required": ["wav_path", "training_csv"],
            },
        ),
        types.Tool(
            name="get_separation",
            description="Compute Cohen's d effect size and separation score between two groups in a training CSV.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Path to training CSV"},
                    "group_a":  {"type": "string", "default": "A"},
                    "group_b":  {"type": "string", "default": "B"},
                },
                "required": ["csv_path"],
            },
        ),
        types.Tool(
            name="list_wavs",
            description="List all WAV files in the project's wavs/ directory with their group labels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "wav_dir":    {"type": "string", "description": "Repository-relative WAV directory"},
                    "labels_csv": {"type": "string", "description": "Repository-relative labels CSV"},
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        result = _dispatch(name, arguments)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"ok": False, "error": str(e)}))]


def _dispatch(name: str, args: dict) -> dict:
    if name == "process_wav":
        return _process_wav(args)
    elif name == "batch_process":
        return _batch_process(args)
    elif name == "classify_wav":
        return _classify_wav(args)
    elif name == "get_separation":
        return _get_separation(args)
    elif name == "list_wavs":
        return _list_wavs(args)
    else:
        return {"ok": False, "error": f"unknown tool: {name}"}


# ── tool implementations ──────────────────────────────────────────────────────

def _process_wav(args: dict) -> dict:
    try:
        wav = validate_read_path(str(_wav_relative(args["wav_path"])), allowed_roots=WAV_READ_ROOTS, extensions=ALLOWED_WAV, maximum=MAX_WAV_BYTES, label="wav_path")
    except PathPolicyError as exc:
        return _policy_failure("path policy", exc)
    path = str(wav)
    sr    = int(args.get("sr",    44100))
    nfft  = int(args.get("nfft",  4096))
    hop   = int(args.get("hop",   1024))
    topdb = int(args.get("topdb", 35))
    win   = float(args.get("win", 0.5))

    if not os.path.isfile(path):
        return {"ok": False, "error": f"file not found: {path}"}

    r = dsp_measure(path, sr, nfft, hop, topdb, win)
    if r is None:
        return {"ok": False, "error": "no signal after trim / WAV too short"}

    return {
        "ok": True,
        "file": os.path.basename(path),
        "params": {"sr": sr, "nfft": nfft, "hop": hop, "topdb": topdb, "win": win},
        "features": {f: round(r[f], 6) for f in FEATS},
    }


def _batch_process(args: dict) -> dict:
    try:
        wav_dir_path = _validate_mcp_dir(str(_wav_relative(args.get("wav_dir") or "wavs")))
        labels_path = validate_read_path(str(args.get("labels_csv") or "labels.csv"), allowed_roots=CSV_READ_ROOTS, extensions=ALLOWED_CSV, maximum=MAX_CSV_BYTES, label="labels_csv")
        output_path = validate_write_path(str(args.get("output_csv", "training.csv")), output_root=MCP_OUTPUT_ROOT, extensions=ALLOWED_CSV, label="output_csv")
    except PathPolicyError as exc:
        return _policy_failure("path policy", exc)
    wav_dir    = str(wav_dir_path)
    labels_csv = str(labels_path)
    output_csv = str(output_path)
    sr    = int(args.get("sr",    44100))
    nfft  = int(args.get("nfft",  4096))
    hop   = int(args.get("hop",   1024))
    topdb = int(args.get("topdb", 35))
    win   = float(args.get("win", 0.5))
    groups = args.get("groups")  # list[str] or None

    if not os.path.isdir(wav_dir):
        return {"ok": False, "error": f"wav_dir not found: {wav_dir}"}

    labels = _load_labels(labels_csv)
    wavs = []
    for filename in sorted(os.listdir(wav_dir)):
        if not filename.lower().endswith(".wav"):
            continue
        try:
            validate_read_path(filename, allowed_roots=(Path(wav_dir),), extensions=ALLOWED_WAV, maximum=MAX_WAV_BYTES, label="wav file")
        except PathPolicyError:
            continue
        wavs.append(filename)

    rows, n_ok, n_fail, group_counts = [], 0, 0, {}
    for fn in wavs:
        base = os.path.splitext(fn)[0]
        grp = labels.get(base)
        if groups and grp not in groups:
            continue
        r = dsp_measure(os.path.join(wav_dir, fn), sr, nfft, hop, topdb, win)
        if r is None:
            n_fail += 1
            continue
        row = {"File": fn, "Group": grp or "?", **{f: round(r[f], 6) for f in FEATS}}
        rows.append(row)
        n_ok += 1
        group_counts[grp or "?"] = group_counts.get(grp or "?", 0) + 1

    out_path = output_csv
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["File", "Group"] + FEATS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "ok": True,
        "n_ok": n_ok, "n_fail": n_fail,
        "output_csv": out_path,
        "group_counts": group_counts,
        "params": {"sr": sr, "nfft": nfft, "hop": hop, "topdb": topdb, "win": win},
    }


def _classify_wav(args: dict) -> dict:
    try:
        wav_path = str(validate_read_path(str(_wav_relative(args["wav_path"])), allowed_roots=WAV_READ_ROOTS, extensions=ALLOWED_WAV, maximum=MAX_WAV_BYTES, label="wav_path"))
        training_csv = str(validate_read_path(str(args["training_csv"]), allowed_roots=CSV_READ_ROOTS, extensions=ALLOWED_CSV, maximum=MAX_CSV_BYTES, label="training_csv"))
    except PathPolicyError as exc:
        return _policy_failure("path policy", exc)
    groups       = args.get("groups")
    sr    = int(args.get("sr",    44100))
    nfft  = int(args.get("nfft",  4096))
    hop   = int(args.get("hop",   1024))
    topdb = int(args.get("topdb", 35))
    win   = float(args.get("win", 0.5))

    if not os.path.isfile(wav_path):
        return {"ok": False, "error": f"WAV not found: {wav_path}"}
    if not os.path.isfile(training_csv):
        return {"ok": False, "error": f"training CSV not found: {training_csv}"}

    r = dsp_measure(wav_path, sr, nfft, hop, topdb, win)
    if r is None:
        return {"ok": False, "error": "WAV could not be processed"}

    test_feats = {f: r[f] for f in FEATS}
    training   = _load_training_csv(training_csv)

    if groups:
        training = [row for row in training if row.get("Group") in groups]

    avail_groups = sorted(set(row["Group"] for row in training if row.get("Group") not in ("?", None)))
    if not avail_groups:
        return {"ok": False, "error": "no valid groups in training CSV"}

    active_feats = [f for f in FEATS if math.isfinite(test_feats.get(f, float("nan")))]

    stats = _zscore_stats(training, active_feats)

    def normalize(d: dict) -> dict[str, float]:
        return {f: (d.get(f, stats[f][0]) - stats[f][0]) / stats[f][1] for f in active_feats}

    norm_test = normalize(test_feats)

    dists: dict[str, float] = {}
    for g in avail_groups:
        g_rows = [normalize(row) for row in training if row.get("Group") == g]
        if not g_rows:
            continue
        c = _centroid(g_rows, active_feats)
        dists[g] = round(_euclidean(norm_test, c, active_feats), 4)

    if not dists:
        return {"ok": False, "error": "could not compute distances"}

    predicted = min(dists, key=lambda g: dists[g])
    min_d = min(dists.values())
    max_d = max(dists.values())
    span = max_d - min_d or 1.0
    confidence = round(1.0 - (dists[predicted] - min_d) / span * 0.5, 3) if len(dists) > 1 else 1.0

    return {
        "ok": True,
        "file": os.path.basename(wav_path),
        "predicted_group": predicted,
        "confidence": confidence,
        "distances": dists,
        "test_features": {f: round(v, 6) for f, v in test_feats.items()},
        "params": {"sr": sr, "nfft": nfft, "hop": hop, "topdb": topdb, "win": win},
    }


def _get_separation(args: dict) -> dict:
    try:
        csv_path = str(validate_read_path(str(args["csv_path"]), allowed_roots=CSV_READ_ROOTS, extensions=ALLOWED_CSV, maximum=MAX_CSV_BYTES, label="csv_path"))
    except PathPolicyError as exc:
        return _policy_failure("path policy", exc)
    group_a  = args.get("group_a", "A").upper()
    group_b  = args.get("group_b", "B").upper()

    if not os.path.isfile(csv_path):
        return {"ok": False, "error": f"CSV not found: {csv_path}"}

    rows = _load_training_csv(csv_path)
    ga = [r for r in rows if r.get("Group") == group_a]
    gb = [r for r in rows if r.get("Group") == group_b]

    if not ga:
        return {"ok": False, "error": f"group {group_a} not found in CSV"}
    if not gb:
        return {"ok": False, "error": f"group {group_b} not found in CSV"}

    feat_results: dict[str, dict] = {}
    for f in FEATS:
        va = [r[f] for r in ga if isinstance(r.get(f), float) and math.isfinite(r[f])]
        vb = [r[f] for r in gb if isinstance(r.get(f), float) and math.isfinite(r[f])]
        if len(va) < 2 or len(vb) < 2:
            feat_results[f] = {"effect_size": 0.0, "score": 0.0}
            continue
        mu_a = sum(va) / len(va)
        mu_b = sum(vb) / len(vb)
        sd_a = math.sqrt(sum((v - mu_a) ** 2 for v in va) / len(va)) or 1.0
        sd_b = math.sqrt(sum((v - mu_b) ** 2 for v in vb) / len(vb)) or 1.0
        pooled_sd = math.sqrt((sd_a ** 2 + sd_b ** 2) / 2) or 1.0
        d = abs(mu_a - mu_b) / pooled_sd
        score = min(100.0, round(d * 40, 1))
        feat_results[f] = {
            "cohen_d": round(d, 4),
            "score": score,
            "mu_a": round(mu_a, 4),
            "mu_b": round(mu_b, 4),
            "n_a": len(va),
            "n_b": len(vb),
        }

    best = max(feat_results, key=lambda f: feat_results[f]["score"])
    overall = round(sum(v["score"] for v in feat_results.values()) / len(feat_results), 1)

    return {
        "ok": True,
        "group_a": group_a,
        "group_b": group_b,
        "overall_score": overall,
        "best_feature": best,
        "features": feat_results,
    }


def _list_wavs(args: dict) -> dict:
    try:
        wav_dir = str(_validate_mcp_dir(str(_wav_relative(args.get("wav_dir") or "wavs"))))
        labels_csv = str(validate_read_path(str(args.get("labels_csv") or "labels.csv"), allowed_roots=CSV_READ_ROOTS, extensions=ALLOWED_CSV, maximum=MAX_CSV_BYTES, label="labels_csv"))
    except PathPolicyError as exc:
        return _policy_failure("path policy", exc)

    if not os.path.isdir(wav_dir):
        return {"ok": False, "error": f"directory not found: {wav_dir}"}

    labels = _load_labels(labels_csv) if os.path.exists(labels_csv) else {}
    wavs = sorted(f for f in os.listdir(wav_dir) if f.lower().endswith(".wav"))
    result = [{"file": fn, "group": labels.get(os.path.splitext(fn)[0], "?")} for fn in wavs]
    by_group: dict[str, int] = {}
    for item in result:
        by_group[item["group"]] = by_group.get(item["group"], 0) + 1

    return {"ok": True, "total": len(result), "by_group": by_group, "wavs": result}


# ── entry point ───────────────────────────────────────────────────────────────

async def main():
    async with mcp.server.stdio.stdio_server() as (r, w):
        await app.run(
            r, w,
            InitializationOptions(
                server_name="dsp-arena",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
