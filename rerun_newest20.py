#!/usr/bin/env python3
"""Re-run the newest-20 measurement with the exact recipe from the previous run.

Recipe (from dsp_out/newest20/relationships.json):
  sr=44100, nfft=4096, hop=1024, topdb=35, win=0.5
Outputs (same format as before):
  dsp_out/newest20/features.csv
  ~/Downloads/dataset/dataset_records.csv
"""
import csv
import os
import shutil

from dsp_inverse import Dataset, Params

WAV_DIR = "/Users/kyawnandar/Downloads/dataset"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "dsp_out", "newest20")
RECORDS_CSV = os.path.join(WAV_DIR, "dataset_records.csv")
PREV_FEATURES = os.path.join(OUT_DIR, "features.csv")

# --- 1. Pick the newest 20 WAVs (by mtime, same as previous run) ---
wavs = sorted(
    (os.path.join(WAV_DIR, f) for f in os.listdir(WAV_DIR) if f.endswith(".wav")),
    key=os.path.getmtime, reverse=True)[:20]
picks = [os.path.basename(p) for p in wavs]
print(f"picked {len(picks)} files: {picks[0]} ... {picks[-1]}")

# --- 2. Recover authoritative labels from the previous features.csv ---
labels = {}
if os.path.exists(PREV_FEATURES):
    with open(PREV_FEATURES) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["File"]] = row["Group"]
missing = [p for p in picks if p not in labels]
for p in missing:
    labels[p] = "A"
print(f"labels loaded: {len(labels)} (missing -> A: {missing if missing else 'none'})")

# --- 3. Stage into a temp wavdir + labels.csv ---
stage = "/tmp/newest20_stage"
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage)
labels_csv = os.path.join(stage, "labels.csv")
with open(labels_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["File_Name", "Result"])
    for p in picks:
        w.writerow([p, labels[p]])
        shutil.copy2(os.path.join(WAV_DIR, p), os.path.join(stage, p))

# --- 4. Measure (default Params == the previous run's recipe) ---
params = Params()
ds = Dataset(stage, labels_csv)
rows = ds.measure_all(params)
print(f"measured {len(rows)} rows")

cols = ["file", "group", "impact", "duration", "peak_time",
        "mel_mean_db", "impact_freq_hz"]

def fmt6(v):
    """6-decimal format with trailing zeros stripped — matches the previous run."""
    if v is None:
        return ""
    return f"{v:.6f}".rstrip("0").rstrip(".")

# --- 5. Write dsp_out/newest20/features.csv (same format as before) ---
os.makedirs(OUT_DIR, exist_ok=True)
HEADER = ["File", "Group", "Impact", "Duration", "Peak_Time",
          "Mel_Mean_dB", "Impact_Freq_Hz"]
with open(os.path.join(OUT_DIR, "features.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADER)
    for r in sorted(rows, key=lambda x: int(x["file"].split(".")[0])):
        row = [r["file"], r["group"]]
        for c in ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]:
            row.append(fmt6(r.get(c)))
        w.writerow(row)
print(f"wrote {OUT_DIR}/features.csv")

# --- 6. Write dataset_records.csv (File_Name,Result + measured features) ---
with open(RECORDS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["File_Name", "Result"] + HEADER[2:])
    for r in sorted(rows, key=lambda x: int(x["file"].split(".")[0])):
        row = [r["file"], r["group"]]
        for c in ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]:
            row.append(fmt6(r.get(c)))
        w.writerow(row)
print(f"wrote {RECORDS_CSV}")

# --- 7. Sanity: compare impact values vs previous run ---
prev = {}
if os.path.exists(PREV_FEATURES):
    with open(PREV_FEATURES) as f:
        for row in csv.DictReader(f):
            prev[row["File"]] = float(row["Impact"])
now = {r["file"]: r.get("impact") for r in rows}
diffs = []
for k in now:
    if k in prev and now[k] is not None:
        diffs.append(abs(now[k] - prev[k]))
if diffs:
    print(f"impact diff vs previous run: max={max(diffs):.4f} mean={sum(diffs)/len(diffs):.4f}")