#!/usr/bin/env python3
"""DSP Chart companion — stdlib-only CLI for WAV injection + serving.

Commands:
  python3 dsp_chart.py serve [--labels CSV]   serve chart + WAVs on 127.0.0.1:8125
  python3 dsp_chart.py add <files...>         copy WAVs into ./wavs/ + update manifest
  python3 dsp_chart.py clear                  empty manifest + ./wavs/

No pip dependencies — Python 3.6+ stdlib only.
"""

import argparse
import csv
import http.server
import json
import os
import subprocess
import shutil
import socketserver
import sys
import tempfile
import time

PORT = 8125
SERVER_LABELS = {}
DIR = os.path.dirname(os.path.abspath(__file__))
WAVS = os.path.join(DIR, "wavs")
MANIFEST = os.path.join(DIR, "manifest.json")
DEFAULT_LABELS = os.path.expanduser("~/Downloads/dataset/dataset_records.csv")
BACKEND_PY = os.path.abspath(os.path.join(DIR, "..", "..", "..", "backend", ".venv", "bin", "python3"))

# ---------- helpers ----------

def ensure_dirs():
    os.makedirs(WAVS, exist_ok=True)

def read_manifest():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, "r") as f:
        return json.load(f)

def write_manifest(entries):
    """Atomic write via tmp+rename."""
    ensure_dirs()
    fd, tmp = tempfile.mkstemp(dir=DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entries, f, indent=1)
        shutil.move(tmp, MANIFEST)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise

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
    """Copy WAVs to wavs/ dir and update manifest."""
    ensure_dirs()
    entries = read_manifest()
    existing = {e["file"] for e in entries}
    added = 0
    for fn in filenames:
        src = os.path.abspath(fn)
        if not os.path.isfile(src):
            print(f"  SKIP: {fn} — not found", file=sys.stderr)
            continue
        basename = os.path.basename(src)
        dst = os.path.join(WAVS, basename)
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
    SERVER_LABELS = load_labels(args.labels) if args.labels else {}
    print(f"dsp_chart: serving {DIR} on http://127.0.0.1:{PORT}/"
          + (f"  ({len(SERVER_LABELS)} labels loaded)" if SERVER_LABELS else "  (no labels — groups from filenames only)"))
    labels = load_labels(args.labels) if args.labels else {}
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

class ChartHandler(http.server.SimpleHTTPRequestHandler):
    _labels = {}
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIR, **kw)

    def end_headers(self):
        path = self.path.split("?")[0]
        if path == "/manifest.json" or path == "/manifest":
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
        elif path.startswith("/wavs/") and path.lower().endswith(".wav"):
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/manifest.json" or path == "/manifest":
            data = read_manifest()
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
            super().do_GET()

    def handle_optimize(self):
        """Run the real Python DSP inverse optimizer and return report JSON.

        Query params:
          ?mode=generic&trials=30
          ?mode=diff&split=AB|CD&trials=30
          ?mode=sim&pair=AB&trials=30

        This uses real WAV recalculation; it never moves dots artificially.
        """
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        mode = q.get("mode", ["generic"])[0]
        trials = q.get("trials", ["30"])[0]
        split = q.get("split", [""])[0]
        pair = q.get("pair", [""])[0]
        out = "webopt_" + str(int(time.time()))
        py = BACKEND_PY if os.path.exists(BACKEND_PY) else sys.executable
        cmd = [py, os.path.join(DIR, "dsp_inverse.py"), "optimize",
               "--wavdir", WAVS,
               "--labels", DEFAULT_LABELS,
               "--mode", mode,
               "--trials", str(int(float(trials))),
               "--out", out]
        if mode == "diff" and split:
            cmd += ["--split", split]
        if mode == "sim" and pair:
            cmd += ["--pair", pair]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        try:
            cp = subprocess.run(cmd, cwd=DIR, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=1800)
            report_path = os.path.join(DIR, "dsp_out", out, "report.json")
            if cp.returncode != 0 or not os.path.exists(report_path):
                body = json.dumps({"ok": False, "command": cmd, "log": cp.stdout[-4000:]}, indent=2).encode()
                self.send_response(500)
            else:
                report = json.load(open(report_path))
                report["ok"] = True
                report["out"] = out
                report["log_tail"] = cp.stdout[-2000:]
                body = json.dumps(report, indent=2).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"ok": False, "error": str(e)}, indent=2).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *a):
        # quiet — only log manifest requests
        path = a[0].split("?")[0] if a else ""
        if "manifest" in path or "add" in str(a):
            super().log_message(format, *a)

def cmd_add(args):
    if not args.files:
        print("usage: dsp_chart.py add <file1.wav> [file2.wav ...]", file=sys.stderr)
        sys.exit(1)
    labels = load_labels(args.labels) if args.labels else {}
    copy_and_add(args.files, labels)

def cmd_clear(args):
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

    sub.add_parser("clear", help="empty manifest + wavs/")

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
