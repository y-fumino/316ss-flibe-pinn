"""
delta_check.py — Quick per-ledger deficit readout (convenience view).
"""

import json, statistics as st
for f in [r"results/HN_eds_only.json", r"results/316H_eds_only.json"]:
    d = json.load(open(f))
    for alloy, blk in d.items():
        if not isinstance(blk, dict) or "seeds" not in blk:
            continue
        mr = st.median(r["mass_ratio"] for r in blk["seeds"])
        print(f"{alloy:12s} median mass_ratio = {mr:.4f}  ->  deficit = {100*(1-mr):.1f} %")