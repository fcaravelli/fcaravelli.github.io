"""
run_all.py  —  Master script: Memristive Reservoir Satellite Control
====================================================================
Runs the full pipeline in order:
  1. satellite_ctdv.py  → static plots (fig1–fig5)
  2. make_gif.py        → animated GIF

Usage:
    python run_all.py

Outputs land in ./outputs/
"""
import subprocess, sys, os, pathlib

ROOT = pathlib.Path(__file__).parent
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

scripts = [
    ("Static plots (fig1–fig5)", ROOT / "satellite_ctdv.py"),
    ("Animated GIF",             ROOT / "make_gif.py"),
]

for label, script in scripts:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    result = subprocess.run([sys.executable, str(script)], check=True)

print("\n" + "="*62)
print("  All done. Outputs:")
for f in sorted(OUT.iterdir()):
    size = f.stat().st_size / 1024
    print(f"    {f.name:<35}  {size:6.0f} KB")
print("="*62)
