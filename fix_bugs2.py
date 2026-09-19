#!/usr/bin/env python3
# fix_bugs2.py — Bug 1 (regex-robust) + real session ranges + diff-split fix. stdlib only.
import os, re, shutil

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
shutil.copy2(PATH, PATH + ".bak2")
with open(PATH) as f:
    src = f.read()

print("== structural check ==")
for marker in ["const BLOCK_MODES", "function sessDuration", "function axisVal",
               "function drawFixedAxes", "function drawFingerprint",
               "function extractRipple", "function drawRipple", "bindRippleMs();",
               "function applyHash", "tryWavMode().then", "</html>"]:
    print(("OK      " if marker in src else "MISSING ") + marker)
print("lines =", src.count("\n") + 1)

# ---------- Bug 1: sessDuration must be called from axisVal ----------
m = re.search(r"function axisVal\(r,f\)\{.*?\n\}", src, re.S)
if not m:
    print("\n[FAIL] axisVal not found — file is not the Phase1/2 version!")
else:
    has = "sessDuration(r)" in m.group(0)
    print("\naxisVal has duration hook:", has)
    if not has:
        new_block = re.sub(
            r"(\n[ \t]*)return\s+r\[f\];",
            lambda mm: mm.group(1) + "if(f==='duration') return sessDuration(r);"
                      + mm.group(1) + "return r[f];",
            m.group(0), count=1)
        src = src[:m.start()] + new_block + src[m.end():]
        m2 = re.search(r"function axisVal\(r,f\)\{.*?\n\}", src, re.S)
        print("after:\n" + m2.group(0))
        print("[OK] Bug 1 fixed")

# ---------- session detection: real ranges (S1=249-280, S2=281-298, S3=299-321) ----------
if "const SESSION_RANGES" not in src:
    old_cols = "const SESSION_COLS=['session','rec_session','session_id','session_no','batch','set','recording','file_no'];"
    new_cols = old_cols + ("\n/* real sessions (user-verified): numbered WAVs outside these ranges get NO session */\n"
                           "const SESSION_RANGES=[{lo:249,hi:280,name:'S1'},{lo:281,hi:298,name:'S2'},{lo:299,hi:321,name:'S3'}];\n"
                           "function sessionForNum(n){n=Number(n);for(const s of SESSION_RANGES){if(n>=s.lo&&n<=s.hi)return s.name;}return null;}")
    if old_cols in src:
        src = src.replace(old_cols, new_cols)
        print("[OK] SESSION_RANGES added")
    else:
        print("[skip] SESSION_COLS line not found — session patch skipped")

old_h = "String(Math.floor(Number(m[1])/10))"
if old_h in src:
    src = src.replace(old_h, "sessionForNum(m[1])")
    print("[OK] detectSession now uses real session ranges")
else:
    print("[skip] tens-digit heuristic string not found — session patch skipped")

# ---------- pre-existing bug: MAKE-DIFFERENT split parsing dropped the 4th group ----------
old_split = "const [p1,p2,p3,p4]=key.split('').map(c=>c);"
new_split = "const [p1p,p2p]=key.split('|');const [p1,p2]=p1p.split('');const [p3,p4]=p2p.split('');"
if old_split in src:
    src = src.replace(old_split, new_split)
    print("[OK] diff-split parsing fixed (AB|CD now uses A,B vs C,D)")
else:
    print("[skip] split line not found — check manually")

with open(PATH, "w") as f:
    f.write(src)

# ---------- deterministic mirror of the JS session logic (evidence) ----------
def sess(n):
    for lo, hi, name in [(249, 280, "S1"), (281, 298, "S2"), (299, 321, "S3")]:
        if lo <= n <= hi: return name
    return None
print("\n== session detection (JS logic mirrored) ==")
for name in ["249.wav","280.wav","281.wav","298.wav","299.wav","321.wav","5.wav","event_001.wav"]:
    n = re.match(r"^(\d+)\.wav$", name)
    print(f"  {name} -> {sess(int(n.group(1))) if n else None}")
print("\nbackup: index.html.bak2 · done")
