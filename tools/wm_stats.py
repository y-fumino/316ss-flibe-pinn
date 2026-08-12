"""
wm_stats.py — Weight-sensitivity statistics across w_mass variants (batches N4–N5).
"""

import json, statistics as st
for f in ["results/316SS_coupled.json", "results/316SS_wm20.json",
          "results/316SS_wm30.json", "results/316SS_wm40.json"]:
    d = json.load(open(f))["316SS"]["seeds"]
    print(f"{f:32s} a_med {st.median(r['alpha'] for r in d):+.4f}"
          f" | D0_med {st.median(r['D0'] for r in d):.3e}"
          f" | pass {sum(r['audit']['audit_pass'] for r in d)}/11")