#!/usr/bin/env python3
"""DSP Chart companion — stdlib-only CLI for WAV injection + serving.

Commands:
  python3 dsp_chart.py serve [--labels CSV]   serve chart + WAVs on 127.0.0.1:8125
  python3 dsp_chart.py add <files...>         copy WAVs into ./wavs/ + update manifest
  python3 dsp_chart.py clear                  empty manifest + ./wavs/

No pip dependencies — Python 3.6+ stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import http.server
import json
import os
import re
import subprocess
import shutil
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path

from path_policy import (
    MAX_CSV_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_WAV_BYTES,
    PROJECT_ROOT,
    PathPolicyError,
    validate_manifest_entries,
    validate_read_path,
    validate_write_path,
)

PORT = 8125
SERVER_LABELS = {}
DIR = os.path.dirname(os.path.abspath(__file__))
WAVS = os.path.join(DIR, "wavs")
MANIFEST = os.path.join(DIR, "manifest.json")
PROJECT_ROOT_DIR = PROJECT_ROOT
CHART_WAVS_ROOT = Path(WAVS).resolve()
CHART_READ_ROOTS = (PROJECT_ROOT_DIR.resolve(), CHART_WAVS_ROOT)
CHART_READ_EXTENSIONS = {".html", ".css", ".js", ".json", ".txt", ".wav"}
CHART_WRITE_ROOT = CHART_WAVS_ROOT
CHART_WRITE_EXTENSIONS = {".wav"}
DEFAULT_LABELS = os.path.expanduser("~/Downloads/dataset/dataset_records.csv")
BACKEND_PY = os.path.abspath(os.path.join(DIR, "..", "..", "..", "backend", ".venv", "bin", "python3"))

# ---------- R3: /optimize hardening ----------
ALLOWED_OPTIMIZE_MODES = frozenset({"generic", "diff", "sim"})
OPTIMIZE_TRIALS_MIN = 1
OPTIMIZE_TRIALS_MAX = 200
OPTIMIZE_TIMEOUT_SECONDS = 600
OPTIMIZE_OUTPUT_ROOT = Path(DIR) / "dsp_out"
OPTIMIZE_SCRIPT = Path(DIR) / "dsp_inverse.py"
OPTIMIZE_CONCURRENCY = 1
OPTIMIZE_MAX_REPORT_BYTES = 1 * 1024 * 1024
OPTIMIZE_MAX_OUTPUT_BYTES = 1 * 1024 * 1024
_POLL_INTERVAL_SECONDS = 0.25
_OPTIMIZE_SEM = threading.Semaphore(OPTIMIZE_CONCURRENCY)
_SPLIT_RE = re.compile(r"^[A-D]\|[A-D]$")
_PAIR_RE = re.compile(r"^[A-D]{2}$")
_CREDENTIAL_REDACT_RE = re.compile(r"(token|secret|password|passwd|credential|api[_-]?key|private[_-]?key)", re.I)

# ---------- helpers ----------

def ensure_dirs():
    os.makedirs(WAVS, exist_ok=True)

def read_manifest():
    manifest_path = Path(MANIFEST)
    if manifest_path.is_symlink():
        raise PathPolicyError("manifest: symlinks are not allowed")
    if not manifest_path.exists():
        return []
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise PathPolicyError("manifest: exceeds size cap")
    with open(MANIFEST, "r") as f:
        entries = json.load(f)
    return validate_manifest_entries(entries)

def write_manifest(entries):
    """Atomic write via tmp+rename, with manifest schema validation."""
    entries = validate_manifest_entries(entries)
    ensure_dirs()
    manifest_path = Path(MANIFEST)
    if manifest_path.is_symlink():
        raise PathPolicyError("manifest: symlinks are not allowed")
    payload = json.dumps(entries, indent=1)
    if len(payload.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise PathPolicyError("manifest: exceeds size cap")
    fd, tmp = tempfile.mkstemp(dir=DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        shutil.move(tmp, MANIFEST)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def _validate_chart_labels_path(path: str | None) -> Path | None:
    if not path:
        return None
    return validate_read_path(path, allowed_roots=CHART_READ_ROOTS, extensions={".csv"}, maximum=MAX_CSV_BYTES, label="labels csv")


def load_labels(csv_path):
    """Read labels CSV → {filename_without_ext: group} mapping.
    Supports columns: File_Name+Result, or File+Group, or any
    two-column CSV where col1=filename, col2=group.
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    labels = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            return {}
        col_file, col_group = 0, 1
        # detect column indices
        hl = [h.strip().lower() for h in header]
        for i, h in enumerate(hl):
            if h in ("file_name", "file", "filename", "wav"):
                col_file = i
            if h in ("result", "group", "label"):
                col_group = i
        for row in reader:
            if len(row) <= max(col_file, col_group):
                continue
            fn = os.path.splitext(row[col_file].strip())[0]
            grp = row[col_group].strip().upper()
            if grp:
                labels[fn] = grp
    return labels

def infer_group(filename, labels):
    """Resolve group for a WAV file.
    Priority: labels CSV match > filename pattern > None.
    """
    base = os.path.splitext(filename)[0]
    # 1) labels CSV
    if base in labels:
        return labels[base]
    # 2) filename pattern: 254_A.wav or A_254.wav
    import re
    m = re.match(r"(\d+)_([ABCD])", base)
    if m:
        return m.group(2)
    m = re.match(r"([ABCD])_(\d+)", base)
    if m:
        return m.group(1)
    return None

def copy_and_add(filenames, labels):
    """Copy approved WAVs into wavs/ and update the validated manifest."""
    ensure_dirs()
    entries = read_manifest()
    existing = {e["file"] for e in entries}
    added = 0
    for fn in filenames:
        src = Path(fn).expanduser().resolve()
        try:
            source_rel = os.path.relpath(str(src), str(PROJECT_ROOT_DIR))
            src = validate_read_path(source_rel, allowed_roots=CHART_READ_ROOTS, extensions=CHART_WRITE_EXTENSIONS, maximum=MAX_WAV_BYTES, label="chart source")
            basename = src.name
            dst = validate_write_path(basename, output_root=CHART_WRITE_ROOT, extensions=CHART_WRITE_EXTENSIONS, label="chart destination")
        except PathPolicyError as exc:
            raise SystemExit(f"  REJECT: {exc}")
        if not src.is_file():
            print(f"  SKIP: {fn} — not found", file=sys.stderr)
            continue
        shutil.copy2(src, dst)
        if basename not in existing:
            grp = infer_group(basename, labels)
            entry = {"file": basename}
            if grp:
                entry["group"] = grp
            entries.append(entry)
            added += 1
            print(f"  + {basename}" + (f" → group {grp}" if grp else ""))
        else:
            print(f"  = {basename} (already in manifest)")
    write_manifest(entries)
    print(f"manifest updated: {len(entries)} total, {added} new")

# ---------- commands ----------

def cmd_serve(args):
    global SERVER_LABELS
    validated_labels = _validate_chart_labels_path(args.labels)
    labels = load_labels(str(validated_labels) if validated_labels else None)
    SERVER_LABELS = labels
    print(f"dsp_chart: serving {DIR} on http://127.0.0.1:{PORT}/"
          + (f"  ({len(SERVER_LABELS)} labels loaded)" if SERVER_LABELS else "  (no labels — groups from filenames only)"))
    ensure_dirs()
    # if no manifest yet, seed from existing wavs/
    if not os.path.exists(MANIFEST):
        wav_files = sorted(f for f in os.listdir(WAVS) if f.lower().endswith(".wav"))
        entries = []
        for fn in wav_files:
            grp = infer_group(fn, labels)
            e = {"file": fn}
            if grp:
                e["group"] = grp
            entries.append(e)
        if entries:
            write_manifest(entries)
            print(f"seeded manifest with {len(entries)} existing WAVs")

    handler = ChartHandler
    handler._labels = labels
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"serving on http://127.0.0.1:{PORT}")
        print(f"  WAVs dir: {WAVS}")
        print(f"  labels: {args.labels or '(none — use --labels to auto-assign groups)'}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")

# ---------- R3 optimize helpers ----------

def _optimize_token() -> str:
    """Route credential; empty means the route is disabled (fail closed)."""
    return os.environ.get("DSP_CHART_OPTIMIZE_TOKEN", "")


def _token_present(headers) -> str | None:
    raw = headers.get("x-dsp-optimize-token", "")
    if not raw:
        authorization = headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            raw = authorization[7:].strip()
    return raw or None


def _optimize_authorized(headers, client_address) -> bool:
    configured = _optimize_token()
    if not configured or len(configured) < 16:
        return False
    try:
        peer = client_address[0]
    except (AttributeError, IndexError, TypeError):
        return False
    if peer not in {"127.0.0.1", "::1", "localhost"}:
        return False
    supplied = _token_present(headers)
    if not supplied:
        return False
    return hmac.compare_digest(supplied, configured)


def validate_optimize_query(query: dict) -> dict:
    """Strict, bounded /optimize schema. Raises ValueError on any violation."""
    allowed_keys = {"mode", "trials", "split", "pair"}
    unknown = sorted(set(query) - allowed_keys)
    if unknown:
        raise ValueError(f"unknown query fields: {unknown}")
    mode = query.get("mode", ["generic"])[0]
    if mode not in ALLOWED_OPTIMIZE_MODES:
        raise ValueError("invalid mode")
    trials_raw = query.get("trials", ["30"])[0]
    if not re.fullmatch(r"\d{1,3}", trials_raw):
        raise ValueError("trials must be an integer")
    trials = int(trials_raw)
    if trials < OPTIMIZE_TRIALS_MIN or trials > OPTIMIZE_TRIALS_MAX:
        raise ValueError("trials out of range")
    split = query.get("split", [""])[0]
    pair = query.get("pair", [""])[0]
    if split and not _SPLIT_RE.fullmatch(split):
        raise ValueError("invalid split")
    if pair and not _PAIR_RE.fullmatch(pair):
        raise ValueError("invalid pair")
    if mode == "diff" and not split:
        raise ValueError("diff mode requires split")
    if mode == "sim" and not pair:
        raise ValueError("sim mode requires pair")
    return {"mode": mode, "trials": trials, "split": split, "pair": pair}


def _optimize_python() -> str:
    """Pinned interpreter allowlist: backend venv python (preferred), else the
    current interpreter. Raises when no allowlisted interpreter is usable."""
    allowlist = (os.path.realpath(BACKEND_PY), os.path.realpath(sys.executable))
    for candidate in allowlist:
        if os.path.isfile(candidate):
            return candidate
    raise PermissionError("no allowed optimize interpreter is available")


def build_optimize_argv(params: dict, out_tag: str) -> list[str]:
    """argv-only construction with fixed script and generated output tag.

    Every element derives from validated schema values or pinned module
    constants; no caller-controlled path ever reaches argv.
    """
    if not re.fullmatch(r"webopt_[0-9]+_[0-9a-f]{8}", out_tag):
        raise ValueError("invalid output tag")
    script = os.path.realpath(str(OPTIMIZE_SCRIPT))
    if script != str(OPTIMIZE_SCRIPT.resolve()) or not os.path.isfile(script):
        raise PermissionError("optimize script unavailable")
    argv = [
        _optimize_python(),
        script,
        "optimize",
        "--wavdir", os.path.realpath(WAVS),
        "--labels", DEFAULT_LABELS,  # pinned constant input, never caller-controlled
        "--mode", params["mode"],
        "--trials", str(params["trials"]),
        "--out", out_tag,
    ]
    if params["mode"] == "diff":
        argv += ["--split", params["split"]]
    if params["mode"] == "sim":
        argv += ["--pair", params["pair"]]
    return argv


def _optimize_env() -> dict:
    allowed = ("PATH", "HOME", "LANG", "LC_ALL")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def run_optimize_subprocess(argv: list[str], timeout: float = OPTIMIZE_TIMEOUT_SECONDS):
    """Bounded argv execution: shell off, stdin closed, own process group, timed
    kill, and captured output capped at OPTIMIZE_MAX_OUTPUT_BYTES.

    Returns (returncode, output, timed_out, output_truncated).
    """
    proc = subprocess.Popen(
        argv,
        cwd=str(DIR),
        env=_optimize_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
        start_new_session=True,
    )
    chunks: list[str] = []
    truncated = threading.Event()

    def _pump(stream) -> None:
        total = 0
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            total += len(chunk)
            if total > OPTIMIZE_MAX_OUTPUT_BYTES:
                truncated.set()
                return
            chunks.append(chunk)

    reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    reader.start()
    timed_out = False
    deadline = time.monotonic() + timeout
    while True:
        try:
            proc.wait(timeout=_POLL_INTERVAL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            pass
        if truncated.is_set():
            _kill_process_group(proc)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _kill_process_group(proc)
            break
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
    reader.join(timeout=5)
    if proc.stdout is not None:
        proc.stdout.close()
    if timed_out:
        returncode = -1
    else:
        returncode = proc.returncode if proc.returncode is not None else -1
    return returncode, "".join(chunks), timed_out, truncated.is_set()


def _resolve_report(out_tag: str) -> Path:
    """Canonical report path confined to the approved output root; symlinks rejected."""
    root = OPTIMIZE_OUTPUT_ROOT.resolve()
    candidate = OPTIMIZE_OUTPUT_ROOT / out_tag / "report.json"
    part = candidate
    while part != OPTIMIZE_OUTPUT_ROOT:
        if part.is_symlink():
            raise PermissionError("report path symlink rejected")
        parent = part.parent
        if parent == part:
            raise PermissionError("report escapes output root")
        part = parent
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError("report escapes output root") from None
    if not resolved.is_file():
        raise FileNotFoundError("report missing")
    if resolved.stat().st_size > OPTIMIZE_MAX_REPORT_BYTES:
        raise PermissionError("report exceeds size cap")
    return resolved


def _redact_audit_value(value):
    """Recursively replace credential-shaped strings so denial events stay auditable."""
    if isinstance(value, str):
        return "[redacted]" if _CREDENTIAL_REDACT_RE.search(value) else value
    if isinstance(value, dict):
        return {_redact_audit_value(key): _redact_audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_audit_value(item) for item in value]
    return value


def _audit_optimize(record: dict) -> None:
    """Append-only structured audit; credential-shaped keys/values are redacted
    per-field instead of dropping the whole event."""
    safe = _redact_audit_value(record)
    line = json.dumps(safe, sort_keys=True, default=str)
    if _CREDENTIAL_REDACT_RE.search(line):
        return
    try:
        OPTIMIZE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        audit_path = OPTIMIZE_OUTPUT_ROOT / "optimize-audit.jsonl"
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


class ChartHandler(http.server.SimpleHTTPRequestHandler):
    _labels = {}
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIR, **kw)

    def end_headers(self):
        path = self.path.split("?")[0]
        if path == "/manifest.json" or path == "/manifest":
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif path.startswith("/wavs/") and path.lower().endswith(".wav"):
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _safe_relative_path(self, raw_path: str) -> str:
        path = raw_path.split("?", 1)[0]
        if path.startswith("//"):
            raise PathPolicyError("protocol-relative paths are not allowed")
        if path in ("", "/"):
            path = "/index.html"
        relative = path[1:] if path.startswith("/") else path
        return relative

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/manifest.json", "/manifest"):
            try:
                data = read_manifest()
            except (PathPolicyError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, explain=f"manifest rejected: {exc}")
                return
            # enrich with the serve-time labels (groups from dataset_records.csv!)
            for e in data:
                if "group" not in e:
                    g = infer_group(e["file"], SERVER_LABELS)
                    if g:
                        e["group"] = g
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/optimize":
            self.handle_optimize()
        else:
            try:
                relative = validate_read_path(
                    self._safe_relative_path(path),
                    allowed_roots=CHART_READ_ROOTS,
                    extensions=CHART_READ_EXTENSIONS,
                    maximum=MAX_WAV_BYTES,
                    label="chart path",
                )
            except PathPolicyError as exc:
                self.send_error(403, explain=f"path rejected: {exc}")
                return
            # Constrain SimpleHTTPRequestHandler to an already-validated relative path.
            self.path = "/" + relative.relative_to(PROJECT_ROOT_DIR).as_posix()
            super().do_GET()

    def do_HEAD(self):
        path = self.path.split("?", 1)[0]
        if path in ("/manifest.json", "/manifest", "/optimize"):
            self.send_error(405, explain="HEAD is not supported for this route")
            return
        try:
            relative = validate_read_path(
                self._safe_relative_path(path),
                allowed_roots=CHART_READ_ROOTS,
                extensions=CHART_READ_EXTENSIONS,
                maximum=MAX_WAV_BYTES,
                label="chart path",
            )
        except PathPolicyError as exc:
            self.send_error(403, explain=f"path rejected: {exc}")
            return
        self.path = "/" + relative.relative_to(PROJECT_ROOT_DIR).as_posix()
        super().do_HEAD()

    def handle_optimize(self):
        """R3-hardened /optimize: disabled unless explicitly credentialed.

        Fixed script + interpreter allowlist, bounded schema, argv-only
        execution, filtered environment, /dev/null stdin, process-group
        timeout kill, semaphore, output-root confinement, audit record.
        """
        from urllib.parse import urlparse, parse_qs

        started = time.time()
        peer = "unknown"
        try:
            peer = self.client_address[0]
        except Exception:
            pass

        if not _optimize_authorized(self.headers, getattr(self, "client_address", ())):
            _audit_optimize({"event": "optimize_denied", "reason": "auth_or_disabled",
                             "peer": peer, "at": started})
            body = json.dumps({"ok": False, "error": "forbidden"}).encode()
            self.send_response(403)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        parsed = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        try:
            params = validate_optimize_query(parsed)
        except ValueError as exc:
            _audit_optimize({"event": "optimize_rejected", "reason": str(exc),
                             "peer": peer, "at": started})
            body = json.dumps({"ok": False, "error": "invalid_request"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if not _OPTIMIZE_SEM.acquire(blocking=False):
            _audit_optimize({"event": "optimize_rejected", "reason": "busy",
                             "peer": peer, "at": started})
            body = json.dumps({"ok": False, "error": "busy"}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        out_tag = f"webopt_{int(time.time())}_{os.urandom(4).hex()}"
        try:
            try:
                argv = build_optimize_argv(params, out_tag)
            except (ValueError, PermissionError) as exc:
                _audit_optimize({"event": "optimize_rejected", "reason": str(exc),
                                 "peer": peer, "at": started})
                body = json.dumps({"ok": False, "error": "invalid_request"}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            returncode, log, timed_out, output_truncated = run_optimize_subprocess(argv)
            duration = round(time.time() - started, 3)
            _audit_optimize({
                "event": "optimize_run", "peer": peer, "mode": params["mode"],
                "trials": params["trials"], "out": out_tag,
                "returncode": returncode, "timed_out": timed_out,
                "output_truncated": output_truncated,
                "duration_s": duration, "at": started,
            })
            if timed_out:
                body = json.dumps({"ok": False, "error": "timeout"}).encode()
                self.send_response(504)
            elif output_truncated:
                body = json.dumps({"ok": False, "error": "output_limit"}).encode()
                self.send_response(500)
            elif returncode != 0:
                body = json.dumps({"ok": False, "error": "optimize_failed"}).encode()
                self.send_response(500)
            else:
                try:
                    report_path = _resolve_report(out_tag)
                    with report_path.open("r", encoding="utf-8") as fh:
                        report = json.load(fh)
                    report["ok"] = True
                    report["out"] = out_tag
                    report["log_tail"] = log[-2000:]
                    body = json.dumps(report, indent=2).encode()
                    self.send_response(200)
                except (OSError, ValueError, PermissionError, json.JSONDecodeError):
                    body = json.dumps({"ok": False, "error": "report_missing"}).encode()
                    self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            _OPTIMIZE_SEM.release()

    def log_message(self, format, *a):
        # quiet — only log manifest requests
        path = a[0].split("?")[0] if a else ""
        if "manifest" in path or "add" in str(a):
            super().log_message(format, *a)

def cmd_add(args):
    if not args.files:
        print("usage: dsp_chart.py add <file1.wav> [file2.wav ...]", file=sys.stderr)
        sys.exit(1)
    validated_labels = _validate_chart_labels_path(args.labels)
    labels = load_labels(str(validated_labels) if validated_labels else None)
    copy_and_add(args.files, labels)

def cmd_clear(args):
    if not getattr(args, "yes", False):
        print("REJECTED: clear requires --yes confirmation", file=sys.stderr)
        sys.exit(2)
    if Path(WAVS).is_symlink() or Path(MANIFEST).is_symlink():
        raise PathPolicyError("clear: refusing to delete through a symlink")
    # remove wavs/
    if os.path.isdir(WAVS):
        shutil.rmtree(WAVS)
        print("cleared wavs/")
    # remove manifest
    if os.path.exists(MANIFEST):
        os.unlink(MANIFEST)
        print("cleared manifest.json")
    ensure_dirs()
    write_manifest([])
    print("done")

# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description="DSP Chart companion CLI")
    sub = p.add_subparsers(dest="cmd")

    srv = sub.add_parser("serve", help="serve chart + WAVs on localhost:8125")
    srv.add_argument("--labels", "-l", help="path to labels CSV (File_Name,Result)")

    add = sub.add_parser("add", help="copy WAVs into wavs/ and update manifest")
    add.add_argument("--labels", "-l", help="path to labels CSV")
    add.add_argument("files", nargs="+", help="WAV files to add")

    clear = sub.add_parser("clear", help="empty manifest + wavs/ (requires --yes)")
    clear.add_argument("--yes", action="store_true", help="confirm destructive chart data reset")

    args = p.parse_args()
    if args.cmd == "serve":
        cmd_serve(args)
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "clear":
        cmd_clear(args)
    else:
        p.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
