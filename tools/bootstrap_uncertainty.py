#!/usr/bin/env python3
"""
bootstrap_uncertainty.py - Data-side uncertainty via parametric bootstrap.

NO GATING: family classification and audit are recorded as descriptive
metadata only and exclude nothing - different data returning different alpha
IS the measured uncertainty. Training seed is FIXED so the reported spread is
purely data-origin, orthogonal to the seed-spread of Table 2
(quadrature-combinable). Noise sigma defaults to the profile's plateau
scatter; pass --sigma to use a documented value.

--alpha-reference is REQUIRED: sign flips are counted against the declared
point estimate, never against the resample median (self-referential - a
distribution straddling zero would report the most optimistic rate).

Usage (from repo root):
  python tools/bootstrap_uncertainty.py --alloy 316SS --mode coupled --n 50 --alpha-reference -0.339 --sigma 1.25
  python tools/bootstrap_uncertainty.py --alloy HASTN --mode eds     --n 25 --alpha-reference -0.640
  python tools/bootstrap_uncertainty.py --alloy 316H  --mode eds     --n 25 --alpha-reference +0.707
"""
import argparse, importlib, json, sys, copy, statistics as st
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import alloy_configs as AC

def find_config(alloy):
    for name in dir(AC):
        obj = getattr(AC, name)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict) and alloy.lower() in str(k).lower():
                    return copy.deepcopy(v), f"{name}[{k!r}]"
            if alloy.lower() in name.lower():
                return copy.deepcopy(obj), name
    sys.exit(f"config for {alloy} not found in alloy_configs; containers: "
             f"{[n for n in dir(AC) if isinstance(getattr(AC,n), dict)]}")

def find_profile_key(cfg):
    cands = []
    for k, v in cfg.items():
        try:
            a = np.asarray(v, dtype=float)
        except Exception:
            continue
        if a.ndim == 1 and 15 <= a.size <= 80 and 0.0 < np.nanmax(a) < 30.0 and "x" not in k.lower():
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    sys.exit(f"profile key ambiguous or missing - candidates {cands}; "
             "set PROFILE_KEY manually at the top of this script")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", required=True)
    ap.add_argument("--mode", choices=["coupled", "eds"], required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="fixed training seed")
    ap.add_argument("--alpha-reference", type=float, required=True, help="declared point estimate whose sign defines a flip (e.g. -0.339, -0.640, +0.707)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="override noise sigma in wt%% (e.g. 1.25 for 316SS per the documented plateau scatter)")
    args = ap.parse_args()
    mod = importlib.import_module(
        "pinn_production_coupled" if args.mode == "coupled" else "pinn_eds_only")
    cfg0, src = find_config(args.alloy)
    pk = find_profile_key(cfg0)
    c0 = np.asarray(cfg0[pk], dtype=float)
    # depth key: same length as the profile, numeric, and monotonically increasing.
    # (The old heuristic looked for "x" in the key name, which never matched
    #  the actual key "depth_um" - the sigma screen below was silently skipped.)
    xk = None
    for k, v in cfg0.items():
        if k == pk:
            continue
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1 and arr.size == c0.size and np.all(np.diff(np.sort(arr)) >= 0) \
           and np.ptp(arr) > 0:
            xk = k
            break
    if xk is not None:
        x = np.asarray(cfg0[xk], dtype=float)
        pf = cfg0.get("plateau_from", x[0] + 2.0 * (x[-1] - x[0]) / 3.0)
        tail = c0[x > pf]
        m = np.median(tail); mad = 1.4826 * np.median(np.abs(tail - m))
        keep = tail > m - 3 * mad
        sigma = float(np.std(tail[keep]))
        sigma_source = f"far-field scatter beyond {pf:.1f} um ({int(keep.sum())} of {len(tail)} tail points)"
        sigma_excluded = int((~keep).sum())
        if (~keep).any():
            print(f"sigma screen: {int((~keep).sum())} tail point(s) more than 3 scaled-MAD below the "
                  f"tail median excluded (deep-attack dips are signal, not noise; same screen as delta_from_csv.py)")
    else:
        sigma = float(np.std(c0[-6:]))
        sigma_source = "FALLBACK: last 6 profile points (small-sample, likely low)"
        sigma_excluded = 0
        print("WARNING: no depth array found; sigma from the last 6 points only - pass --sigma.")
    if args.sigma is not None:
        sigma = args.sigma
        sigma_source = "explicit --sigma"
    print(f"config={src} profile_key={pk} n_pts={c0.size} plateau_sigma={sigma:.4f} wt%")
    # protocol-knob self-healing probe: alloy_configs is data-only by design,
    # so protocol constants (w_mass, alpha0, ...) are injected here from the module
    filled = {}
    for _ in range(6):
        try:
            _c = copy.deepcopy(cfg0)
            _c["__dry_probe__"] = True
            import inspect
            srccode = inspect.getsource(mod.run_pinn)
            import re as _re
            for key in _re.findall(r'cfg\["(\w+)"\]', srccode):
                if key not in cfg0:
                    const = getattr(mod, key.upper(), None)
                    if const is not None:
                        cfg0[key] = const; filled[key] = const
            break
        except Exception as e:
            print("probe warning:", e); break
    if filled: print("protocol knobs injected from module constants:", filled)
    still = [k for k in __import__('re').findall(r'cfg\["(\w+)"\]', __import__('inspect').getsource(mod.run_pinn)) if k not in cfg0]
    if still:
        sys.exit(f"missing protocol keys with no module constant: {still} - paste this list back")
    results = []
    for r in range(args.n):
        rng = np.random.default_rng(1000 + r)
        cfg = copy.deepcopy(cfg0)
        cfg[pk] = (c0 + rng.normal(0.0, sigma, c0.size)).tolist()
        out = mod.run_pinn(cfg, args.seed)
        out["resample"] = r
        results.append(out)
        print(f"  r={r}: D0={out['D0']:.3e} alpha={out['alpha']:+.3f} "
              f"audit={'pass' if out['audit']['audit_pass'] else 'FAIL'}")
    all_a=[o["alpha"] for o in results]; all_D0=[o["D0"] for o in results]
    all_Dw=[o["D_wall"] for o in results]
    all_mD=[0.5*(d0+dw) for d0,dw in zip(all_D0,all_Dw)]
    ref = args.alpha_reference
    if abs(ref) < 0.05:
        print("WARNING: |alpha-reference| < 0.05 - flip counting is ill-defined near zero.")
    flips=len([v for v in all_a if v*ref < 0])
    def smad(a):
        m=st.median(a); return 1.4826*st.median([abs(x-m) for x in a])
    rep = {
        "rule": "NO gating: family classification and audit are recorded as descriptive "
                "metadata only and exclude nothing - different data returning different "
                "alpha IS the measured uncertainty. Training seed fixed at %d; "
                "noise sigma = far-field scatter over the outer third, after a "
                "3 scaled-MAD screen (deep-attack dips are signal, not noise)." % args.seed,
        "n_resamples": args.n, "sigma_wt_pct": sigma,
        "sigma_source": sigma_source, "sigma_screen_excluded": sigma_excluded,
        "n_converged": len([o for o in results if o.get("converged",True)]),
        "n_audit_pass_descriptive": len([o for o in results if o["audit"]["audit_pass"]]),
        "alpha_reference_sign": ("+" if ref>0 else "-"), "sign_flip_count": flips, "sign_flip_fraction": flips/len(all_a),
        "alpha_median": st.median(all_a), "alpha_popSD": st.pstdev(all_a), "alpha_scaledMAD": smad(all_a),
        "D0_median": st.median(all_D0), "D0_popSD": st.pstdev(all_D0),
        "D_wall_median": st.median(all_Dw), "D_wall_popSD": st.pstdev(all_Dw),
        "meanD_median": st.median(all_mD), "meanD_popSD": st.pstdev(all_mD),
        "stability_note": "compare relative popSD of meanD < D_wall < D0/alpha (S6 hierarchy prediction)",
        "resamples": results,
    }
    # results/bootstrap/ - deliberately OUTSIDE the sealed ledger directory:
    # the 24 ledgers in results/ are the bit-identical world (make_seal.py
    # refuses to seal if unexpected files sit beside them); bootstrap records
    # are the distribution world, re-run under changing sigma and n.
    out = Path("results") / "bootstrap" / (f"bootstrap_{args.alloy}_{args.mode}"
                                          f"_n{args.n}_sig{sigma:.4f}_seedfixed.json")
    if out.exists():
        sys.exit(f"refusing to overwrite {out} - move it aside first")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"written: {out}  (data-origin spread: alpha popSD {rep['alpha_popSD']:.4f})")

if __name__ == "__main__":
    main()
