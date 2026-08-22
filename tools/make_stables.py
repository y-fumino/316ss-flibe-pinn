#!/usr/bin/env python3
"""
make_stables.py - Generate Supplementary Tables S1-S20 (document order; S19 = convention-and-window grid, S20 = units) from the campaign
result ledgers, without retraining. S7 is the condensed synthetic summary;
S19 (the deficit-convention-and-window grid) is generated self-contained from frozen digitized-profile snapshots.
Table S20 (units verification) is maintained directly in the manuscript
source and is not generated here.

Sources: current-generation ledgers under results/ (history/ fallback for
inherited names). Quadratic-closure tables are emitted in endpoint-curvature
coordinates (D0, D_wall, beta, <D>); cross-references follow the current
supplement numbering (synthetic study: Section S4; two-basis study: Section S6).

The closing census counts the audit-passing runs of the study inventory
(Section 3.4): 12 ledgers, 100 runs, of which the two deficit-alloy coupled
campaigns fail the audit by design - expected output n=78, max 0.0253.
Robustness variants (weight sweeps, full-extent, L40) are disclosed
separately in the second sweep sentence of Section 2.3.

Usage (from re_run/): python tools/make_stables.py   [no shell redirect -
the script writes tables_supplementary.md itself]
"""
import json, sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "tools"))

def mad(x):
    x = np.asarray(x); return 1.4826 * np.median(np.abs(x - np.median(x)))

def J(fname):
    for d in (BASE / "results", BASE / "history"):
        p = d / fname
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    raise FileNotFoundError(f"{fname} not found in results/ or history/")

def seed_table(title, seeds, quad=False, raw_chart=False):
    lines = [f"**{title}**", ""]
    if quad:
        if raw_chart:
            lines.append("| seed | D₀ (cm²/s) | D~wall~ (cm²/s) | β (cm²/s) | ⟨D⟩ (cm²/s) | a1 | a2 | loss | RMSE (wt%) | w~gap~ | w(0,1) | audit |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| seed | D₀ (cm²/s) | D~wall~ (cm²/s) | β (cm²/s) | ⟨D⟩ (cm²/s) | loss | RMSE (wt%) | w~gap~ | w(0,1) | audit |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
    else:
        lines.append("| seed | D₀ (cm²/s) | α | D~wall~ (cm²/s) | loss | RMSE (wt%) | L-BFGS | w(1,1) | w(0,1) | ∫(1−u) | w~gap~ | audit |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in seeds:
        a = r["audit"]
        if quad:
            D0, a1, a2 = r["D0"], r["alpha1"], r["alpha2"]  # ledger fields; used for derived coords only
            Dw = r.get("D_wall", D0 * (1 + a1 + a2))
            beta = r.get("beta", -D0 * a2)
            mD = r.get("meanD", 0.5 * (D0 + Dw) + beta / 6.0)
            raw_cols = f"{a1:+.3f} | {a2:+.3f} | " if raw_chart else ""
            lines.append(f"| {r['seed']} | {D0:.2e} | {Dw:.2e} | {beta:+.2e} | {mD:.2e} | "
                         f"{raw_cols}"
                         f"{r['loss']:.5f} | {r['eds_rmse']:.2f} | "
                         f"{a['w_gap']:+.4f} | {a['w_pred_at_x0_t1']:+.4f} | "
                         f"{'pass' if a['audit_pass'] else 'FAIL'} |")
        else:
            lines.append(f"| {r['seed']} | {r['D0']:.2e} | {r['alpha']:+.3f} | {r['D_wall']:.2e} | "
                         f"{r['loss']:.5f} | {r['eds_rmse']:.2f} | {r['lbfgs_steps']} | "
                         f"{a['w_pred_at_x1_t1']:+.4f} | {a['w_pred_at_x0_t1']:+.4f} | "
                         f"{a['integral_1_minus_u']:+.4f} | {a['w_gap']:+.4f} | "
                         f"{'pass' if a['audit_pass'] else 'FAIL'} |")
    D0s = [r["D0"] for r in seeds]
    als = [r.get("alpha", r.get("alpha1", 0.0)) for r in seeds]
    Dws = [r.get("D_wall", r["D0"] * (1 + r.get("alpha", 0.0))) for r in seeds]
    lines.append("")
    if quad:
        lines.append(f"Ensemble (all listed seeds): D₀ median {np.median(D0s):.2e} [MAD {mad(D0s):.1e}]; "
                     f"D~wall~ median {np.median(Dws):.2e} [MAD {mad(Dws)/np.median(Dws)*100:.1f} %]")
    else:
        lines.append(f"Ensemble (all listed seeds; the values reported in Table 2 are filtered by the solution-family classification and audit of Section 3.4): D₀ median {np.median(D0s):.2e} [MAD {mad(D0s):.1e}]; "
                     f"α median {np.median(als):+.3f} [MAD {mad(als):.3f}]; "
                     f"D~wall~ median {np.median(Dws):.2e} [MAD {mad(Dws)/np.median(Dws)*100:.1f} %]")
    if quad:
        mDs = [r.get("meanD", 0.5*(r["D0"]+r.get("D_wall", r["D0"]*(1+r["alpha1"]+r["alpha2"])))+r.get("beta", -r["D0"]*r["alpha2"])/6.0) for r in seeds]
        lines.append(f"⟨D⟩: {np.mean(mDs):.2e} +/- {np.std(mDs):.1e} ({np.std(mDs)/np.mean(mDs)*100:.1f} %)")
    lines.append("")
    return lines

def pick(d, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default

def condensed_synth_table(fq, sw):
    """Table S7: eight-configuration summary of the synthetic campaign,
    aggregated from the same ledgers as Tables S8-S9 (three seeds per
    configuration; sign vs truth computed from the recovered alpha,
    ground truth α < 0 in every case)."""
    cases = {}
    for blk in (next(iter(fq.values())), next(iter(sw.values()))):
        for s_rec in (blk.get("seeds") or blk.get("cases") or []):
            for c in pick(s_rec, "cases", "results", default=[s_rec]):
                name = pick(c, "case", "name", "tag", default="?")
                k = float(pick(c, "inflation", "k", "infl", "infl_factor", default=1))
                regime = "Deep" if "Deep" in name else "Shallow"
                label = f"{regime}, consistent" if k == 1 else f"{regime}, inflated"
                cases.setdefault((regime, k, label), []).append(
                    (pick(c, "seed", default=pick(s_rec, "seed")),
                     pick(c, "alpha", default=float("nan")),
                     bool(c.get("audit", {}).get("audit_pass"))))
    lines = ["**Table S7. Synthetic verification summary: sign recovery and audit outcome for the eight configurations of the four-quadrant design and the inflation sweep (three seeds each; ground truth α < 0; complete per-seed listing in Tables S8-S9).**", ""]
    lines.append("| configuration | k | injected Δ | α (three seeds) | sign vs truth | audit |")
    lines.append("|---|---|---|---|---|---|")
    order = [("Deep", 1.0), ("Deep", 5.0), ("Deep", 10.0), ("Deep", 20.0),
             ("Shallow", 1.0), ("Shallow", 5.0), ("Shallow", 10.0), ("Shallow", 20.0)]
    for regime, k in order:
        key = next(kk for kk in cases if kk[0] == regime and kk[1] == k)
        rows = sorted(cases[key])
        delta = "0" if k == 1 else f"{100*(k-1)/k:.0f} %"
        alphas = " / ".join(f"{a:+.2f}" for _, a, _ in rows)
        n_neg = sum(1 for _, a, _ in rows if a < 0)
        sign = f"correct {n_neg}/3" if n_neg == 3 else (f"reversed {3-n_neg}/3" if n_neg == 0 else f"correct {n_neg}/3, reversed {3-n_neg}/3")
        n_pass = sum(1 for _, _, p in rows if p)
        aud = f"pass {n_pass}/3" if n_pass == 3 else (f"FAIL {3-n_pass}/3" if n_pass == 0 else f"pass {n_pass}/3, FAIL {3-n_pass}/3")
        lines.append(f"| {key[2]} | {k:.0f} | {delta} | {alphas} | {sign} | {aud} |")
    lines.append("")
    return lines



def synth_table(title, data):
    """Per-case listing for the synthetic campaign. Ground truth is alpha < 0
    in every case; sign-vs-truth is computed from the recovered alpha."""
    lines = [f"**{title}**", ""]
    lines.append("| seed | case | k | injected Δ | alpha | sign vs truth | w~gap~ | w(0,1) | audit |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    blk = next(iter(data.values()))
    for s_rec in (blk.get("seeds") or blk.get("cases") or []):
        seed = pick(s_rec, "seed")
        cases = pick(s_rec, "cases", "results", default=[s_rec])
        for c in cases:
            name = pick(c, "case", "name", "tag", default="?")
            k = pick(c, "inflation", "k", "infl", "infl_factor", default=1)
            try:
                kf = float(k); delta = f"{100*(kf-1)/kf:.0f} %"
            except (TypeError, ValueError):
                delta = "-"
            a = c.get("audit", {})
            alpha = pick(c, "alpha", default=float("nan"))
            svt = "negative (correct)" if alpha < 0 else "POSITIVE (spurious)"
            lines.append(f"| {seed} | {name} | {k} | {delta} | {alpha:+.3f} | {svt} | "
                         f"{a.get('w_gap', float('nan')):+.4f} | "
                         f"{a.get('w_pred_at_x0_t1', float('nan')):+.4f} | "
                         f"{'pass' if a.get('audit_pass') else 'FAIL'} |")
    lines.append("")
    return lines

def tolerance_table(ens):
    """Table S8: primary/secondary partitions vs classification tolerance."""
    # Load classify_basins from the tool file directly (single source),
    # independent of sys.path resolution quirks.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "make_tables", Path(__file__).resolve().parent / "make_tables.py")
    _mt = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mt)
    classify_basins = _mt.classify_basins
    lines = ["**Table S15. Solution-family classification tolerance sweep (production ensembles): "
             "n_primary / n_secondary at each tolerance (applied to both log₁₀ D₀ and α).**", ""]
    alloys = list(ens.keys())
    lines.append("| tolerance | " + " | ".join(alloys) + " |")
    lines.append("|---" * (len(alloys) + 1) + "|")
    for tol in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        cells = []
        for name in alloys:
            conv = [r for r in ens[name]["seeds"] if r.get("converged", True)]
            pri, sec = classify_basins(conv, tol_logD0=tol, tol_alpha=tol)
            cells.append(f"{len(pri)} / {len(sec)}")
        lines.append(f"| {tol:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Every reported (audit-passing) partition is invariant over [0.25, 0.45], "
                 "a plateau containing the reporting tolerance of 0.30; transitions at tighter "
                 "tolerances resolve intra-family spread (316SS coupled at 0.15; Hastelloy-N "
                 "EDS-only below 0.25), and all remaining variation lies in footnote-gated "
                 "(audit-FAIL) ensembles.")
    lines.append("")
    return lines

out = []

# ---- S1-S3: production campaign (current-generation per-alloy ledgers) ----
prod = {}
for fname in ("316SS_coupled.json", "HN_coupled.json", "316H_coupled.json"):
    prod.update(J(fname))
for name, tno in [("316SS", "S1"), ("Hastelloy-N", "S2"), ("316H", "S3")]:
    out += seed_table(f"Table {tno}. Production campaign (coupled, w~mass~ = 10): {name}", prod[name]["seeds"])
out.append("Note on Table S3: both same-time replicate anchors are fed raw; w(1,1) equals their constrained mean (2.4936, i.e. 3.184 mg/cm2) to four digits in every seed, and the mass RMSE equals the replicate half-spread (0.957 mg/cm2) exactly - the irreducible floor of Section 3.3. The recorded w_target field holds the last-fed anchor (specimen 2), not the constrained mean.")
out.append("")

# ---- S4-S6: EDS-only campaigns and initialization control ----
eds = {}
for fname in ("HN_eds_only.json", "316H_eds_only.json"):
    eds.update(J(fname))
out += seed_table("Table S4. EDS-only campaign: Hastelloy-N (reported ensemble)", eds["Hastelloy-N"]["seeds"])
out += seed_table("Table S5. EDS-only campaign: 316H (reported ensemble)", eds["316H"]["seeds"])
neg = J("316H_eds_only_alpha0_minus1.json")
key = list(neg.keys())[0]
out += seed_table("Table S6. Initialization-reversal control: 316H, EDS-only, α₀ = −1.0", neg[key]["seeds"])

# ---- S7-S9: condensed summary and synthetic verification campaign ----
out += condensed_synth_table(J("S2_four_quadrant.json"), J("S2_inflation_sweep.json"))
out += synth_table("Table S8. Synthetic verification: four-quadrant design (Section S4)", J("S2_four_quadrant.json"))
out += synth_table("Table S9. Synthetic verification: inflation sweep (Section S4)", J("S2_inflation_sweep.json"))
out.append("Note on Tables S8-S9: ground truth is α = −0.5 in every case; the sign-vs-truth column is computed from the recovered α. The overload factors are set at the deficit scales of the real records: the five- and tenfold overloads (80 % and 90 %) bracket the 84 % Hastelloy-N deficit, and the twentyfold overload (95 %) matches the 316H deficit.")
out.append("")

# ---- S10-S11: quadratic-closure study, dual-basis (physical primary, polynomial control) ----
cp = J("316SS_quad_physical_basis.json")
out += seed_table("Table S10. Quadratic-closure identifiability study: 316SS, coupled, trained in the endpoint-curvature chart of Section 2.1 (physical basis)", cp["316SS"]["seeds"], quad=True)
a2j = J("316SS_quad_polynomial_basis.json")
out += seed_table("Table S11. Parameterization-robustness control: 316SS, coupled, trained in a polynomial basis (raw coefficients a1, a2 alongside their endpoint-curvature transformation)", a2j["316SS"]["seeds"], quad=True, raw_chart=True)

# ---- S12-S14: quadratic-closure probes, EDS-only (current generation, batch N9) ----
qa = J("HN_quad_eds_only.json")
out += seed_table("Table S12. Quadratic-closure probe: Hastelloy-N, EDS-only", qa["Hastelloy-N"]["seeds"], quad=True)
qb = J("316H_quad_eds_only.json")
out += seed_table("Table S13. Quadratic-closure probe: 316H, EDS-only", qb["316H"]["seeds"], quad=True)
qc = J("316SS_quad_eds_only.json")
out += seed_table("Table S14. Quadratic-closure mode-matched disambiguation: 316SS, EDS-only", qc["316SS"]["seeds"], quad=True)

# ---- S15: tolerance sweep (coupled production + reported EDS-only ensembles) ----
ens = {f"{n} (coupled)": prod[n] for n in prod}
ens["Hastelloy-N (EDS-only)"] = eds["Hastelloy-N"]
ens["316H (EDS-only)"] = eds["316H"]
out += tolerance_table(ens)

# ---- S19: deficit-convention-and-window grid ----
# Single-source design: profiles from data/*_eds_profile.csv (the digitization
# canon), physics (rho, C_bulk, dW_data) from src/alloy_configs.py (the run
# canon). Only the ANALYSIS CHOICES live here: the three windows per alloy
# (depletion layer ~2.5x rule / production analysis window - for 316SS the
# 40 um ablation window of Section S5 / full digitized extent) and the plateau
# cutoff (first depth safely beyond the depletion zone). Measured-loss basis
# follows the paper: 316SS = audit-convention anchor (last of dW_data);
# Hastelloy-N = the single record; 316H = mean of the same-time replicates.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("alloy_configs", BASE / "src" / "alloy_configs.py")
_ac = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ac)

_S19_SPEC = [
    ("316SS",       "316SS_eds_profile.csv", (15, 40, 80), 16.0, "last"),
    ("Hastelloy-N", "HN_eds_profile.csv",    (6, 15, 70),  15.0, "first"),
    ("316H",        "316H_eds_profile.csv",  (5, 12, 50),  12.5, "mean"),
]

def _s16_load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        parts = line.replace(",", " ").replace(chr(9), " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue  # header
    rows.sort()
    dedup = [rows[0]]
    for r in rows[1:]:
        if r[0] != dedup[-1][0]:
            dedup.append(r)
    arr = np.array(dedup)
    return arr[:, 0], arr[:, 1]

out.append("**Table S19. The model-free deficit under analysis conventions and analysis windows: "
           "two baselines (nominal C_bulk of Table 1; measured plateau mean) x two integrands "
           "(clipped: max(C_bulk - C, 0); unclipped: C_bulk - C) x three windows per alloy. "
           "Each cell: profile-explained mass in mg/cm^2 (Delta as % of the measured loss). "
           "Windows per alloy: 316SS 0-15 (deficit-conditioned, ~2.5x the depletion depth), "
           "0-40 (the ablation window of Section S5), 0-80 (full extent); Hastelloy-N 0-6, 0-15 "
           "(deficit-conditioned = production analysis window), 0-70; 316H 0-5, 0-12 "
           "(deficit-conditioned = production analysis window), 0-50. Trapezoidal integration "
           "of the raw digitized points.**")
out.append("")
out.append("| alloy | window (um) | nominal / clipped | nominal / unclipped | plateau / clipped | plateau / unclipped |")
out.append("|---|---|---|---|---|---|")
_plat_note = []
for _al, _csv, _ws, _pf, _mrule in _S19_SPEC:
    _cfg = _ac.get_config(_al)
    _rho, _cbn = float(_cfg["rho"]), float(_cfg["C_bulk"])
    _dw = np.asarray(_cfg["dW_data"], dtype=float)
    _meas = {"last": _dw[-1], "first": _dw[0], "mean": float(np.mean(_dw))}[_mrule]
    _x, _C = _s16_load(BASE / "data" / _csv)
    _cbp = float(_C[_x >= _pf].mean())
    _plat_note.append(f"{_al} {_cbp:.2f} (x >= {_pf:g} um)")
    for _W in _ws:
        _m = _x <= _W + 1e-9
        _cells = []
        for _cb in (_cbn, _cbp):
            for _clip in (True, False):
                _y = np.maximum(_cb - _C[_m], 0.0) if _clip else (_cb - _C[_m])
                _ex = 1e3 * _rho / 100 * np.trapezoid(_y, _x[_m] * 1e-4)
                _cells.append(f"{_ex:.3f} ({100*(1-_ex/_meas):+.1f} %)")
        out.append(f"| {_al} | 0-{_W} | " + " | ".join(_cells) + " |")
out.append("")
out.append("Note to Table S19: the deficit alloys are sign-invariant across all twelve cells each; "
           "316SS is negative in every cell of its deficit-conditioned window and splits sign only on "
           "the full digitized extent (Section 4.4). Plateau means: " + "; ".join(_plat_note) + ".")
out.append("")

# ---- S16: data-origin (bootstrap) summary; S17: time-sweep maxima ----
bdir = BASE / "results" / "bootstrap"
if bdir.exists():
    out.append("**Table S16. Data-origin spread: parametric bootstrap at the measured noise of each profile "
               "(training seed fixed, no gating; Section 5.5). alpha quantiles are over the resamples; "
               "<D> = (D0 + D_wall)/2.**")
    out.append("")
    out.append("| alloy | n | sigma (wt%) | sign flips | alpha min / med / max | alpha scaled-MAD | <D> median (cm^2/s) | <D> robust rel. |")
    out.append("|---|---|---|---|---|---|---|---|")
    _ORD = {"316SS": 0, "Hastelloy-N": 1, "316H": 2}
    def _alloy_of(name):
        u = str(name).upper()
        if "316SS" in u:
            return "316SS"
        if "316H" in u:
            return "316H"
        if "HN" in u or "HAST" in u:
            return "Hastelloy-N"
        return str(name)
    _rows16 = []
    for p in sorted(bdir.glob("*.json")):
        d = json.load(open(p, encoding="utf-8"))
        rs = d["resamples"]
        al = sorted(r["alpha"] for r in rs)
        mD = sorted(0.5 * (r["D0"] + r["D_wall"]) for r in rs)
        smad_a = mad(al); med_m = float(np.median(mD)); smad_m = mad(mD)
        _name = _alloy_of(d.get("alloy", p.stem))
        _rows16.append((_ORD.get(_name, 9), f"| {_name} | {len(rs)} | {d.get('sigma_wt_pct', float('nan')):.3f} | "
                   f"{d.get('sign_flip_count', 0)}/{len(rs)} | "
                   f"{al[0]:+.3f} / {float(np.median(al)):+.3f} / {al[-1]:+.3f} | {smad_a:.3f} | "
                   f"{med_m:.2e} | {smad_m/med_m*100:.1f} % |"))
    for _o, _r in sorted(_rows16):
        out.append(_r)
    out.append("")
out.append("**Table S17. Time-sweep maxima per campaign (21-point grid together with the anchor times; "
           "max_t |w(0,t)| and max_t |w_gap(t)| over all seeds of each ledger; Section 2.3).**")
out.append("")
out.append("| campaign | max |w(0,t)| | max |w_gap(t)| |")
out.append("|---|---|---|")
_rows17 = []
for p in sorted((BASE / "results").glob("*.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    for key, blk in d.items():
        if not isinstance(blk, dict):
            continue
        recs = blk.get("seeds") or blk.get("cases") or []
        mw = [r["audit"]["time_sweep"]["max_abs_w0"] for r in recs
              if isinstance(r, dict) and "time_sweep" in r.get("audit", {})]
        mg = [r["audit"]["time_sweep"]["max_abs_w_gap"] for r in recs
              if isinstance(r, dict) and "time_sweep" in r.get("audit", {})]
        if mw:
            _tag = blk.get("run_tag", key)
            _rows17.append((_ORD.get(_alloy_of(_tag), 9), str(_tag),
                            f"| {_tag} ({p.stem}) | {max(mw):.4f} | {max(mg):.4f} |"))
for _o, _t, _r in sorted(_rows17):
    out.append(_r)
out.append("")

# ---- S18: 316SS anchor-free control (the symmetric-design completion) ----
try:
    _ss_eds = J("316SS_eds_only.json")
    _k20 = [k for k in _ss_eds if isinstance(_ss_eds[k], dict) and "seeds" in _ss_eds[k]][0]
    out += seed_table("Table S18. EDS-only control campaign: 316SS (anchor-free counterpart of the "
                      "production ensemble; Sections 3.4 and 5.5)", _ss_eds[_k20]["seeds"])
    out.append("Note on Table S18: the anchor-free 316SS ensemble reproduces the coupled estimates "
               "(D0 within 0.7 %, median alpha -0.318 versus -0.339, eleven of eleven seeds negative), "
               "establishing that the mass anchor does not bias the 316SS inversion - independently "
               "of the audit (Section 5.5).")
    out.append("")
except FileNotFoundError:
    out.append("(Table S18 requires results/316SS_eds_only.json - campaign N11.)")
    out.append("")

out.append("")

# ---- audit-gap census for the threshold sentence of Section 2.3 ----
gaps = []
for fname in ("316SS_coupled.json", "HN_coupled.json", "316H_coupled.json",
              "HN_eds_only.json", "316H_eds_only.json",
              "316H_eds_only_alpha0_minus1.json", "316SS_single_anchor.json",
              "316SS_quad_polynomial_basis.json",
              "HN_quad_eds_only.json", "316H_quad_eds_only.json",
              "316SS_quad_eds_only.json",
              "316SS_quad_physical_basis.json"):
    d = J(fname)
    for alloy in d.values():
        for r in alloy["seeds"]:
            a = r["audit"]
            if a["audit_pass"]:
                gaps.append(max(abs(a["w_gap"]), abs(a["w_pred_at_x0_t1"])))
print(f"passing runs: n={len(gaps)}, max audit quantity = {max(gaps):.4f}, "
      f"95th pct = {np.percentile(gaps,95):.4f}")

body = "\n".join(out)

# ---- presentation pass: scientific notation and units (D5) ----
# 4.17e-14 -> 4.17 x 10^-14 (unicode superscripts); mg/cm2 -> mg/cm^2.
# Applied to the rendered text only - every number above is computed first.
import re as _re
_SUP = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
def _sci(m):
    exp = m.group(2).lstrip("+")
    neg = exp.startswith("-")
    exp = exp.lstrip("-").lstrip("0") or "0"
    if neg:
        exp = "-" + exp
    return m.group(1) + " \u00d7 10" + exp.translate(_SUP)
body = _re.sub(r"(\d(?:\.\d+)?)[eE]([+-]?\d+)", _sci, body)
body = body.replace("mg/cm2", "mg/cm\u00b2")

with open(BASE / "tables_supplementary.md", "w", encoding="utf-8") as f:
    f.write(body)
s9_status = "FALLBACK - run tools/delta_sensitivity.py and paste manually" \
    if "run it separately" in body else "embedded"
print(f"tables_supplementary.md written: {body.count(chr(10))+1} file lines, "
      f"{sum(1 for L in body.splitlines() if L.startswith('**Table'))} tables, "
      f"Table S19: {s9_status}")
