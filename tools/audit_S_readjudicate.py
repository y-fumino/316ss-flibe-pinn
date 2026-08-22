#!/usr/bin/env python3
"""
audit_S_readjudicate.py - Re-adjudicate every archived run under the proposed
signed-sum gate, without retraining.

  S = w(0,t1) + w_gap  =  w(1,t1) - integral(1-u)dx   (the conserved total)

Proposed rule:   S > +0.01  -> DEFICIT (would gate parameters)
                 S < -0.01  -> SYSTEMATICS (reported, not gating)
                 |S| <= 0.01 -> CLOSED

Reads every ledger in results/ (seeds- and cases-structured), prints per-run
S, the old channel-wise verdict, the new class, and a change summary.
Usage (from re_run/):  python audit_S_readjudicate.py
"""
import json, glob, statistics as st
from pathlib import Path

THR = 0.01
rows = []
for fn in sorted(glob.glob("results/*.json")):
    try:
        d = json.load(open(fn, encoding="utf-8"))
    except Exception:
        continue
    for key, blk in d.items():
        if not isinstance(blk, dict):
            continue
        recs = blk.get("seeds") or blk.get("cases") or []
        for r in recs:
            if not isinstance(r, dict) or "audit" not in r:
                continue
            a = r["audit"]
            if "w_pred_at_x0_t1" not in a or "w_gap" not in a:
                continue
            S = a["w_pred_at_x0_t1"] + a["w_gap"]
            old = "pass" if a.get("audit_pass") else "FAIL"
            new = "DEFICIT" if S > THR else ("SYSTEMATICS" if S < -THR else "CLOSED")
            tag = blk.get("run_tag", key)
            rows.append((tag, Path(fn).stem, r.get("seed", r.get("case", "?")), S, old, new))

print(f"{'campaign':<34}{'seed':>6}{'S':>10}  old -> new")
changed = []
for tag, stem, seed, S, old, new in rows:
    # a change of substance: old pass that is now DEFICIT, or old FAIL now CLOSED
    subst = (old == "pass" and new == "DEFICIT") or (old == "FAIL" and new == "CLOSED")
    mark = "  <-- SUBSTANTIVE" if subst else ""
    if subst:
        changed.append((tag, seed, S, old, new))
    print(f"{tag:<34}{str(seed):>6}{S:>10.4f}  {old} -> {new}{mark}")

print(f"\nruns evaluated: {len(rows)}")
by = {}
for *_, S, old, new in [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]:
    by[(old, new)] = by.get((old, new), 0) + 1
for (old, new), n in sorted(by.items()):
    print(f"  {old} -> {new}: {n}")
print(f"\nSUBSTANTIVE changes (pass->DEFICIT or FAIL->CLOSED): {len(changed)}")
for tag, seed, S, old, new in changed:
    print(f"  {tag} seed {seed}: S={S:+.4f} {old}->{new}")
if not changed:
    print("  none - every gating decision survives the signed-sum rule.")

# per-campaign sum statistics (the noise-floor evidence)
print("\nper-campaign S statistics (mean, seed SD):")
tags = sorted(set(r[0] for r in rows))
for t in tags:
    Ss = [r[3] for r in rows if r[0] == t]
    if len(Ss) > 1:
        print(f"  {t:<34} {st.mean(Ss):+.4f}  SD {st.stdev(Ss):.4f}  (n={len(Ss)})")
