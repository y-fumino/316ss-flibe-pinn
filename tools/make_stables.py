#!/usr/bin/env python3
"""
make_stables.py - Generate all Supplementary tables (S1-S16; Table S7 condensed summary; Table S16 embedded)
from the campaign result ledgers, without retraining. Table S9 (deficit
conventions) is generated separately by tools/delta_sensitivity.py.

Sources: current-generation ledgers under results/; inherited probe ledgers
(quadratic/chart closure studies, Sections S4/S10) under history/ with their
original filenames - the loader falls back automatically. Quadratic-closure
tables are emitted in endpoint-curvature coordinates (D0, D_wall, beta, <D>).

Usage (from re_run/): python tools/make_stables.py
Output: tables_supplementary.md in the repository root.
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
        for s_rec in blk["seeds"]:
            for c in pick(s_rec, "cases", "results", default=[s_rec]):
                name = pick(c, "case", "name", "tag", default="?")
                k = float(pick(c, "inflation", "k", "infl", "infl_factor", default=1))
                regime = "Deep" if "Deep" in name else "Shallow"
                label = f"{regime}, consistent" if k == 1 else f"{regime}, inflated"
                cases.setdefault((regime, k, label), []).append(
                    (pick(c, "seed", default=pick(s_rec, "seed")),
                     pick(c, "alpha", default=float("nan")),
                     bool(c.get("audit", {}).get("audit_pass"))))
    lines = ["**Table S7. Synthetic verification summary: sign recovery and audit "
             "outcome for the eight configurations of the four-quadrant design and "
             "the inflation sweep (three seeds each; ground truth α < 0; "
             "complete per-seed listing in Tables S8-S9).**", ""]
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
    for s_rec in blk["seeds"]:
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

# ---- S16: deficit-convention grid (embedded from tools/delta_sensitivity.py) ----
import subprocess, os
try:
    _env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, str(BASE / "tools" / "delta_sensitivity.py")],
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=BASE, env=_env)
    if r.returncode == 0 and ("Table S16" in r.stdout) or ("Table S9" in r.stdout):
        out.append(r.stdout.strip().replace("Table S9.", "Table S16."))
        out.append("")
    else:
        out.append("(Table S16 is generated by tools/delta_sensitivity.py; run it separately if this embed failed.)")
        out.append("")
except Exception as e:
    out.append(f"(Table S16 embed failed: {e}; run tools/delta_sensitivity.py separately.)")
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
with open(BASE / "tables_supplementary.md", "w", encoding="utf-8") as f:
    f.write(body)
s9_status = "FALLBACK - run tools/delta_sensitivity.py and paste manually" \
    if "run it separately" in body else "embedded"
print(f"tables_supplementary.md written: {body.count(chr(10))+1} file lines, "
      f"{sum(1 for L in body.splitlines() if L.startswith('**Table'))} tables, "
      f"Table S16: {s9_status}")
