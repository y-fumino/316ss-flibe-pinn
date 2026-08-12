"""
viol_stats.py — Violation-total statistics across ledgers, both conventions.
signed = w(0,1) + w_gap  (algebraically w(1,1) − ∫(1−u)dx: the unmet remainder)
abs    = |w(0,1)| + |w_gap|
Usage: python tools/viol_stats.py results/HN_coupled.json results/316H_coupled.json ...
"""
import json, statistics as st, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    for alloy, blk in d.items():
        if not isinstance(blk, dict) or "seeds" not in blk: continue
        sg = [r["audit"]["w_pred_at_x0_t1"] + r["audit"]["w_gap"] for r in blk["seeds"]]
        ab = [abs(r["audit"]["w_pred_at_x0_t1"]) + abs(r["audit"]["w_gap"]) for r in blk["seeds"]]
        print(f"{f:34s} {alloy:12s} signed {st.mean(sg):+.4f} +/- {st.stdev(sg):.4f} | abs {st.mean(ab):.4f} +/- {st.stdev(ab):.4f} (n={len(sg)})")