#!/usr/bin/env python3
"""Fix Bug 1 (sessDuration dead code) + Bug 2 (CSS vars in canvas)."""
import re, os, shutil

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
bak = PATH + ".bak"
shutil.copy2(PATH, bak)

with open(PATH, "r") as f:
    src = f.read()

# Bug 1: wire sessDuration into axisVal
src = src.replace(
    """  if(f==='ratio_impact_dur'){
    const a=r.impact,b=r.duration;
    return (Number.isFinite(a)&&Number.isFinite(b)&&b>0)?a/b:NaN;
  }
  return r[f];""",
    """  if(f==='ratio_impact_dur'){
    const a=r.impact,b=r.duration;
    return (Number.isFinite(a)&&Number.isFinite(b)&&b>0)?a/b:NaN;
  }
  if(f==='duration') return sessDuration(r);
  return r[f];"""
)

# Bug 2: CSS vars -> hex in all canvas 2D context calls
for prop in ("fillStyle", "strokeStyle"):
    pat = rf"(ctx\.{prop}\s*=\s*['\"])(var\(--(\w+)\))(['\"])"
    def _repl(m):
        prefix, _var, name, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
        hexmap = {"dim": "#8b93a7", "txt": "#e8ecf6", "warn": "#ffd166"}
        return prefix + hexmap.get(name, _var) + suffix
    src = re.sub(pat, _repl, src)

with open(PATH, "w") as f:
    f.write(src)

remaining = re.findall(r'ctx\.\w+Style\s*=\s*[\'"]var\(--', src)
if remaining:
    print(f"WARNING {len(remaining)} CSS-var canvas refs remain:")
    for r in remaining:
        print(f"   {r}")
else:
    print("Bug 2 fix: 0 CSS-var canvas refs remaining")

if "sessDuration(r)" in src:
    print("Bug 1 fix: sessDuration(r) present in axisVal")
else:
    print("WARNING Bug 1 fix NOT found")

print(f"Done. Backup -> {bak}")
