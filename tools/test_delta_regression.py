#!/usr/bin/env python3
"""
test_delta_regression.py - Zero-argument regression: does the standalone entry
point (tools/delta_from_csv.py) reproduce Table S16 of the paper?
All conventions (rho, C_bulk, C_surface, L, plateau_from) and expected values
are hardcoded from the canonical instrument (tools/delta_sensitivity.py).
Usage: python tools/test_delta_regression.py     -> prints OK/FAIL per alloy
"""
import subprocess, sys
CASES = [
    ("316SS", ["--profile","data/316SS_eds_profile.csv","--mass-loss","0.547",
     "--density","8.0","--c-bulk","17.0","--c-surface","1.2","--domain-um","80",
     "--plateau-from","25","--expect-delta-pct","-67.3","--expect-plateau-pct","-103.2","--expect-regime","consistent"]),
    ("Hastelloy-N", ["--profile","data/HN_eds_profile.csv","--mass-loss","0.724",
     "--density","8.86","--c-bulk","7.53","--c-surface","0.82","--domain-um","15",
     "--plateau-from","6","--expect-delta-pct","81.7","--expect-plateau-pct","87.8","--expect-regime","deficit"]),
    ("316H", ["--profile","data/316H_eds_profile.csv","--mass-loss","3.184",
     "--density","8.0","--c-bulk","16.6","--c-surface","3.3","--domain-um","12",
     "--plateau-from","5","--expect-delta-pct","94.6","--expect-plateau-pct","94.4","--expect-regime","deficit"]),
]
fail = 0
for name, args in CASES:
    r = subprocess.run([sys.executable, "tools/delta_from_csv.py"] + args,
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"{name}: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(r.stdout[-400:]); fail += 1
sys.exit(1 if fail else 0)
