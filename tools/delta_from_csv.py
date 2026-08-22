#!/usr/bin/env python3
"""
delta_from_csv.py - Step 0 of the two-axis diagnostic, for YOUR corrosion data.

Given two routine measurements - a depth profile (CSV: depth_um, C_wt_pct) and
a gravimetric mass loss - this reports the model-free deficit Delta under the
full 2x2 analysis-convention grid (baseline: nominal vs plateau; integrand:
clipped vs unclipped) and reads the regime.

What it decides:
  - whether the mass balance closes on the measured extent,
  - therefore whether the domain must be reduced before an inversion, and
  - the mass-anchor weight that follows (Section 5.6, Step 1).

What it does NOT decide:
  - HOW FAR to reduce the domain.  The reference paper sizes the reduced
    domain from the SEM-observed intragranular depletion depth (Section 3.3),
    an input this tool does not have.  Pass --domain-um to evaluate the grid
    on a domain you have sized yourself.

No model, no training, no fitting - arithmetic only.

Usage:
  python tools/delta_from_csv.py --profile my_profile.csv --mass-loss 0.724 \
         --density 8.86 --c-bulk 7.53 [--c-surface 0.82] [--exposure-h 1000]

Reference: "A Two-Axis Diagnostic for Molten-Salt Corrosion: Mass Deficit and
Diffusion Regime via a Self-Auditing Inversion" (JNM, submitted).
"""
import argparse, csv, sys
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

INVARIANCE_SPAN = 0.15   # convention span below which the deficit reads as invariant


def load_profile(path):
    x, c = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                x.append(float(row[0])); c.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    x, c = np.asarray(x), np.asarray(c)
    i = np.argsort(x)
    return x[i], c[i]


def explained_mass(rho, baseline, cg, xg, clip):
    """Profile-explained mass, mg/cm^2.  rho [g/cm^3] x wt-fraction x um."""
    integ = baseline - cg
    if clip:
        integ = np.clip(integ, 0.0, None)
    return rho / 100.0 * _trapz(integ, xg) * 1e-4 * 1e3


def grid(rho, mass_loss, x, c, nominal, plateau, printer, baselines=None):
    """Evaluate the convention grid.  Returns {(baseline, integrand): frac}.

    `baselines` selects which baseline rows to evaluate.  On the full measured
    extent only the plateau baseline is meaningful: the nominal-plateau offset
    is a baseline shift, not depletion, and integrating it over the whole
    far-field inflates the explained mass in proportion to the extent.  On a
    reduced analysis window the offset contributes negligibly and both
    baselines are exercised, as in Table S16.
    """
    if baselines is None:
        baselines = (("nominal", nominal), ("plateau", plateau))
    out = {}
    for bname, B in baselines:
        for cname, clip in (("clipped", True), ("unclipped", False)):
            e = explained_mass(rho, B, c, x, clip)
            d = mass_loss - e
            frac = d / mass_loss
            tag = ("   [invalid: negative explained - enrichment or outliers "
                   "dominate this convention]" if e < 0 else "")
            printer(f"{bname:<10}{cname:<11}{e:>11.3f}{d:>+9.3f}{100*frac:>+15.1f} %{tag}")
            out[(bname, cname)] = frac if e >= 0 else None
    return out


def read_regime(fracs):
    """(label, w_mass, message) from the valid convention fractions."""
    vals = [v for v in fracs.values() if v is not None]
    if not vals:
        return "invalid", None, ("every convention returned a negative explained mass; "
                                 "inspect the profile before proceeding.")
    lo, hi = min(vals), max(vals)
    if lo > 0 and (hi - lo) < INVARIANCE_SPAN:
        return "deficit", 0, (
            f"deficit invariant across the exercised conventions (+{100*lo:.0f} to +{100*hi:.0f} %) - "
            "a substantial fraction of the measured loss leaves no record in the retained profile; "
            "unrecorded-pathway regime (cf. Section 4.4 / Table S16 of the reference).")
    if lo < 0 < hi:
        return "consistent", 10, (
            f"SIGN UNSTABLE under conventions ({100*lo:+.0f} to {100*hi:+.0f} %) - "
            "closed balance fluctuating within measurement systematics; homogenized reading adequate.")
    return "intermediate", 10, (
        f"convention range {100*lo:+.0f} to {100*hi:+.0f} % - intermediate case; "
        "inspect the grid and the profile quality before classifying.")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="This tool decides WHETHER the domain must be reduced, not HOW FAR.")
    ap.add_argument("--profile", required=True, help="CSV with two columns: depth_um, C_wt_pct")
    ap.add_argument("--mass-loss", type=float, required=True, help="measured gravimetric loss, mg/cm^2")
    ap.add_argument("--density", type=float, required=True, help="alloy density, g/cm^3")
    ap.add_argument("--c-bulk", type=float, default=None,
                    help="certified bulk composition, wt%% (default: plateau mean; the baseline axis of the grid is then NOT exercised)")
    ap.add_argument("--c-surface", type=float, default=None,
                    help="surface composition, wt%% (default: outermost profile point; pass the preprocessed value to match the paper)")
    ap.add_argument("--exposure-h", type=float, default=None, help="exposure time, hours (emitted into the suggested config)")

    adv = ap.add_argument_group("advanced")
    adv.add_argument("--domain-um", type=float, default=None,
                     help="evaluate the grid additionally on a domain YOU have sized (e.g. from a micrograph)")
    adv.add_argument("--plateau-from", type=float, default=None,
                     help="depth (um) beyond which the profile is far-field (default: last third)")

    reg = ap.add_argument_group("regression (exit nonzero on mismatch)")
    reg.add_argument("--expect-delta-pct", type=float, default=None,
                     help="nominal-clipped Delta/measured (%%) on the reported domain")
    reg.add_argument("--expect-plateau-pct", type=float, default=None,
                     help="plateau-clipped Delta/measured (%%) on the reported domain")
    reg.add_argument("--expect-regime", choices=["deficit", "consistent", "intermediate"], default=None)
    a = ap.parse_args()

    x, c = load_profile(a.profile)
    if len(x) < 5:
        sys.exit("need at least 5 numeric (depth, concentration) rows")

    pf = a.plateau_from if a.plateau_from is not None else x[0] + 2.0 * (x[-1] - x[0]) / 3.0
    tail = x >= pf
    if not tail.any():
        sys.exit("--plateau-from lies beyond the profile")
    plateau = float(np.mean(c[tail]))
    sig_plat = float(np.std(c[tail]))

    c_surf = a.c_surface if a.c_surface is not None else float(c[0])
    c_bulk_given = a.c_bulk is not None
    nominal = a.c_bulk if c_bulk_given else plateau
    if not c_bulk_given:
        print("WARNING: --c-bulk not given; the nominal baseline defaults to the plateau mean, "
              "so the baseline axis of the convention grid is NOT exercised.")

    if not (0.01 <= a.mass_loss <= 50):
        print(f"WARNING: --mass-loss = {a.mass_loss} is unusual for mg/cm^2 "
              "(corrosion literature often reports mdd or mg/dm^2 - convert first).")
    if not (2.0 <= a.density <= 25.0):
        print(f"WARNING: --density = {a.density} is unusual for g/cm^3.")

    print(f"points: {len(x)}  extent: {x[0]:.2f}-{x[-1]:.2f} um  "
          f"plateau (mean beyond {pf:.1f} um): {plateau:.3f} wt%")
    print(f"baselines: nominal {nominal:.3f} / plateau {plateau:.3f} wt%   "
          f"measured loss: {a.mass_loss:.3f} mg/cm^2")

    amplitude = nominal - c_surf
    if amplitude <= 0:
        sys.exit("C_bulk must exceed C_surface")
    if sig_plat > 0.05 * amplitude:
        print(f"\nWARNING: far-field scatter ({sig_plat:.2f} wt%) is {100*sig_plat/amplitude:.0f} % of the "
              "depletion amplitude - the tail is not a quiet plateau (deep attack?). The plateau baseline "
              "is unreliable here; consider --plateau-from (cf. the deep intergranular attack of the 316H "
              "case in the reference).")

    extent = float(x[-1] - x[0])
    ceiling = a.density / 100.0 * amplitude * extent * 1e-4 * 1e3
    occ = a.mass_loss / ceiling
    print(f"\ndepletion budget rho*(C_bulk-C_surface)*L/100 over the measured extent "
          f"({extent:.1f} um): {ceiling:.3f} mg/cm^2")
    print(f"budget occupancy (measured / ceiling): {occ:.2f}")
    if occ > 1.0:
        print("  > 1: even complete depletion over the measured extent cannot supply the "
              "measured loss - the balance cannot close, before any inversion.")
    elif occ > 0.6:
        print("  near 1: little headroom - a deficit is likely.")

    header = f"{'baseline':<10}{'integrand':<11}{'explained':>11}{'Delta':>9}{'Delta/measured':>16}"
    print(f"\n--- convention grid on the measured extent ({extent:.1f} um) ---")
    if c_bulk_given and abs(nominal - plateau) > 1e-9:
        print("    (plateau baseline only: over the full extent the nominal-plateau offset "
              f"of {nominal - plateau:+.3f} wt% is a baseline shift, not depletion, and would\n"
              "     accumulate across the far field. Both baselines are exercised on a "
              "reduced window - pass --domain-um.)")
    print(header)
    fracs = grid(a.density, a.mass_loss, x, c, nominal, plateau, print,
                 baselines=(("plateau", plateau),))

    reported_fracs = fracs
    if a.domain_um is not None:
        m = x <= a.domain_um
        if m.sum() < 3:
            sys.exit("--domain-um leaves fewer than 3 profile points")
        print(f"\n--- convention grid on your domain (L = {a.domain_um:g} um), as in Table S16 ---")
        print(header)
        reported_fracs = grid(a.density, a.mass_loss, x[m], c[m], nominal, plateau, print)
        cw = a.density / 100.0 * amplitude * a.domain_um * 1e-4 * 1e3
        print(f"budget on this domain: {cw:.3f} mg/cm^2 -> occupancy {a.mass_loss/cw:.2f}"
              + ("  [> 1: cannot close within this domain]" if a.mass_loss / cw > 1 else ""))

    regime, _wmass, message = read_regime(reported_fracs)
    print(f"\nreading ({'your domain' if a.domain_um is not None else 'measured extent'}): {message}")
    if regime == "invalid":
        sys.exit(1)

    print("\n--- next step (Section 5.6) ---")
    if regime == "deficit":
        print("  The balance does NOT close: the profile cannot account for the measured loss.")
        print("  1. REDUCE THE DOMAIN before inverting. This tool does not size the reduced")
        print("     domain; the reference paper takes it from the SEM-observed intragranular")
        print("     depletion depth (Section 3.3, ~2.5x that depth). Re-run with --domain-um")
        print("     once you have sized it, to see the grid on that domain.")
        print("  2. INVERT WITHOUT THE MASS ANCHOR: module `pinn_eds_only`.")
        print("     The anchor cannot be satisfied, so a coupled run would spend the deficit")
        print("     as a boundary lift and fail the audit (Sections 4.2-4.3).")
    else:
        print("  The balance closes on the measured extent: no domain reduction is indicated;")
        print(f"  the measured extent ({extent:.1f} um) can serve as the computational domain.")
        print("  INVERT WITH THE MASS ANCHOR: module `pinn_production_coupled`.")
    print()
    print("  The alloy configuration itself (profile arrays, gravimetric record, temperature,")
    print("  outlier rule) is written by hand into alloy_configs.py - this tool reports the")
    print("  two decisions above, not the configuration.")

    checks, failed = [], False
    if a.expect_delta_pct is not None:
        got = reported_fracs[("nominal", "clipped")]
        checks.append(("nominal-clipped %", None if got is None else 100 * got, a.expect_delta_pct, 0.5))
    if a.expect_plateau_pct is not None:
        got = reported_fracs[("plateau", "clipped")]
        checks.append(("plateau-clipped %", None if got is None else 100 * got, a.expect_plateau_pct, 0.5))
    if checks or a.expect_regime:
        print()
    for label, got, want, tol in checks:
        ok = got is not None and abs(got - want) < tol
        shown = "n/a" if got is None else f"{got:+.2f}"
        print(f"regression: {label} {shown} vs expected {want:+.2f} -> {'OK' if ok else 'FAIL'}")
        failed |= not ok
    if a.expect_regime:
        ok = regime == a.expect_regime
        print(f"regression: regime '{regime}' vs expected '{a.expect_regime}' -> {'OK' if ok else 'FAIL'}")
        failed |= not ok
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
