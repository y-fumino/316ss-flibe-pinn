#!/usr/bin/env python3
"""
compare_ledgers.py -- Bit-identity verdict between two result ledgers.

Usage:
    python compare_ledgers.py <reference.json> <candidate.json>

Matches alloy blocks by name, then compares every numeric field per seed
(D0, alpha, D_wall, loss, eds_rmse, mass_rmse, mass_ratio, audit fields).
Verdict: BIT-IDENTICAL (all fields exactly equal) / MATCH (rel < 1e-12)
/ DIFFER (inspect the reported worst field against the pre-registered
difference mask for the run pair in question).
"""
import json
import sys

FIELDS = ["D0", "alpha", "D_wall", "loss", "eds_rmse", "eds_nrmse",
          "mass_rmse", "mass_ratio", "lbfgs_steps", "converged"]
AUDIT_FIELDS = ["w_pred_at_x1_t1", "w_pred_at_x0_t1", "integral_1_minus_u",
                "w_gap", "w_target", "u_min_on_grid", "u_max_on_grid", "audit_pass"]


def blocks(d):
    return {k: v for k, v in d.items()
            if isinstance(v, dict) and isinstance(v.get("seeds"), list)}


def main(p1, p2):
    ref = blocks(json.load(open(p1, encoding="utf-8")))
    cand = blocks(json.load(open(p2, encoding="utf-8")))
    for alloy, b2 in cand.items():
        if alloy not in ref:
            print(f"[{alloy}] no matching block in reference (blocks: {list(ref)})")
            continue
        s1 = {r["seed"]: r for r in ref[alloy]["seeds"] if isinstance(r, dict)}
        exact = near_n = n = 0
        worst = (0.0, "", None)
        for r2 in b2["seeds"]:
            r1 = s1.get(r2["seed"])
            if r1 is None:
                print(f"  seed {r2['seed']}: absent from reference")
                continue
            pairs = [(f, r1.get(f), r2.get(f)) for f in FIELDS]
            pairs += [(f"audit.{f}", r1.get("audit", {}).get(f),
                       r2.get("audit", {}).get(f)) for f in AUDIT_FIELDS]
            for name, a, b in pairs:
                if a is None or b is None:
                    continue
                n += 1
                if a == b:
                    exact += 1
                elif isinstance(a, float) and isinstance(b, float):
                    rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
                    if rel < 1e-12:
                        near_n += 1
                    if rel > worst[0]:
                        worst = (rel, name, r2["seed"])
                else:
                    print(f"  seed {r2['seed']} {name}: {a!r} != {b!r}")
        verdict = ("BIT-IDENTICAL" if exact == n else
                   "MATCH (rel<1e-12)" if exact + near_n == n else "DIFFER")
        print(f"[{alloy}] fields compared: {n}, exact: {exact}, "
              f"near: {near_n} -> {verdict}")
        if worst[0] > 0:
            print(f"  worst relative diff: {worst[0]:.3e} at {worst[1]} (seed {worst[2]})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
