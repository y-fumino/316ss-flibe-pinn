#!/usr/bin/env python3
"""
delta_from_csv.py - Model-free mass-deficit screening for YOUR corrosion data.
Needs only two routine measurements: a depth profile (CSV: depth_um, C_wt_pct)
and a gravimetric mass loss. Returns the deficit Delta under the full 2x2
analysis-convention grid (baseline: nominal vs plateau; integrand: clipped vs
unclipped) and a regime reading in the sense of the two-axis diagnostic.
No model, no training, no fitting - arithmetic only.

Usage:
  python tools/delta_from_csv.py --profile my_profile.csv --mass-loss 0.72 \\
         --density 8.86 [--c-bulk 7.53] [--plateau-from 10] [--c-surface 0.82]

Reference: "A Two-Axis Diagnostic for Molten-Salt Corrosion: Mass Deficit and
Diffusion Regime via a Self-Auditing Inversion" (JNM, submitted).
"""
import argparse, csv, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="CSV with two columns: depth_um, C_wt_pct")
    ap.add_argument("--mass-loss", type=float, required=True, help="measured gravimetric loss, mg/cm^2")
    ap.add_argument("--density", type=float, required=True, help="alloy density, g/cm^3")
    ap.add_argument("--c-bulk", type=float, default=None, help="certified bulk composition, wt%% (default: plateau mean)")
    ap.add_argument("--plateau-from", type=float, default=None, help="depth (um) beyond which the profile is far-field (default: last third)")
    ap.add_argument("--c-surface", type=float, default=None, help="surface composition, wt%% (default: outermost point; informational)")
    a = ap.parse_args()
    x, c = [], []
    with open(a.profile, newline="") as f:
        for row in csv.reader(f):
            try:
                x.append(float(row[0])); c.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    x, c = np.asarray(x), np.asarray(c)
    i = np.argsort(x); x, c = x[i], c[i]
    if len(x) < 5:
        sys.exit("need at least 5 numeric (depth, concentration) rows")
    pf = a.plateau_from if a.plateau_from is not None else x[0] + 2.0 * (x[-1] - x[0]) / 3.0
    plateau = float(np.mean(c[x >= pf]))
    nominal = a.c_bulk if a.c_bulk is not None else plateau
    print(f"points: {len(x)}  extent: {x[0]:.2f}-{x[-1]:.2f} um  plateau(mean beyond {pf:.1f} um): {plateau:.3f} wt%")
    print(f"baselines: nominal {nominal:.3f} / plateau {plateau:.3f} wt%   measured loss: {a.mass_loss:.3f} mg/cm^2\n")
    print(f"{'baseline':<10}{'integrand':<11}{'explained':>11}{'Delta':>9}{'Delta/measured':>16}")
    fracs = []
    for bname, B in (("nominal", nominal), ("plateau", plateau)):
        for cname, clip in (("clipped", True), ("unclipped", False)):
            integ = B - c
            if clip:
                integ = np.clip(integ, 0.0, None)
            explained = a.density / 100.0 * np.trapezoid(integ, x) * 1e-4 * 1e3  # mg/cm^2
            delta = a.mass_loss - explained
            frac = delta / a.mass_loss
            fracs.append(frac)
            print(f"{bname:<10}{cname:<11}{explained:>11.3f}{delta:>+9.3f}{100*frac:>+15.1f} %")
    fmin, fmax = min(fracs), max(fracs)
    print()
    if fmin > 0 and (fmax - fmin) < 0.15:
        print(f"reading: CONVENTION-INVARIANT deficit (+{100*fmin:.0f} to +{100*fmax:.0f} %) - "
              "a substantial fraction of the measured loss leaves no record in the retained profile; "
              "unrecorded-pathway regime (cf. Section 4.4 / Table S16 of the reference).")
    elif fmin < 0 < fmax:
        print(f"reading: SIGN UNSTABLE under conventions ({100*fmin:+.0f} to {100*fmax:+.0f} %) - "
              "closed balance fluctuating within measurement systematics; homogenized reading adequate.")
    else:
        print(f"reading: convention range {100*fmin:+.0f} to {100*fmax:+.0f} % - intermediate case; "
              "inspect the grid and the profile quality before classifying.")
    if a.c_surface is None:
        print(f"(surface point: {c[0]:.3f} wt% at {x[0]:.2f} um)")

if __name__ == "__main__":
    main()
