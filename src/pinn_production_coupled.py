"""
Multi-Physics Coupled PINN for Molten FLiBe Corrosion — Production Run (full)
==============================================================================
Inverse estimation of concentration-dependent chromium diffusion parameters
from spatial EDS concentration profiles and gravimetric mass loss data.

Three alloy systems:
  - 316 Stainless Steel / FLiBe / 700°C / 3000h
  - Hastelloy-N / FLiBe / 750°C / 1000h
  - Type 316H Stainless Steel / FLiBe / 750°C / 1000h

Training protocol (unified for all alloys, float64 precision):
  Phase 1: Adam 15k @ lr=1e-3
  Phase 2: Adam 10k @ lr=1e-4 (settling)
  Phase 3: L-BFGS (conditional restart <=2 if < 50 steps)

Post-processing:
  Stage 1: Validity by optimizer state (L-BFGS steps >= 50)
  Stage 2: Basin membership by parameter-space clustering in
           (log10 D0, alpha); loss used only for ordering and
           primary-basin designation.
  A-PINN consistency audit (dense grid at t=1):
    w_gap = w(1,1) - w(0,1) - integral(1-u)dx  ==> 0 iff dw/dx = 1-u holds
    audit_pass = |w_gap| < 0.05 and |w(0,1)| < 0.05
    Expectation: pass where the profile and the gravimetric record are
    mutually consistent (316SS); FAIL where a mass deficit exists
    (Hastelloy-N and 316H — the audit quantifies the unaccounted mass).

Reference:
  G. Zheng et al., J. Nucl. Mater. 482 (2016) 147-155
  G. Zheng, Ph.D. Thesis, Univ. of Wisconsin-Madison (2015)
  K.M. Sankar et al., Nucl. Technol. 210 (2024) 391-408
"""
import deepxde as dde
import numpy as np
import torch
import json
import sys

# ============================================
# Run Configuration
# ============================================
# Full production run: all 11 seeds, all three alloys.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]

# Tag appended to the output JSON filename so earlier runs (pilot,
# audit316H) are never overwritten.
RUN_TAG = "production_standalone"
# Standalone execution writes all three alloys under this single tag.
# Canonical campaign results are produced per-job via run_all.py + campaigns.json.

PROTOCOL_STR = ("Adam 15k@1e-3 + 10k@1e-4 → L-BFGS (restart ≤2), "
                "float64, α₀=+1.0")

# ============================================================
# Alloy configurations — imported from the single source of truth.
# REFACTORED (2026-07): no script defines its own alloy dicts anymore;
# see alloy_configs.py for data and provenance. This script declares
# only PROTOCOL: W_MASS is a property of the run, not of an alloy.
# ============================================================
import alloy_configs as AC

W_MASS = 10   # mass-anchor loss weight (protocol knob)

CFG_316SS = {**AC.get_config("316SS"), "w_mass": W_MASS}
CFG_HASTN = {**AC.get_config("Hastelloy-N"), "w_mass": W_MASS}
CFG_316H = {**AC.get_config("316H"), "w_mass": W_MASS}

ALL_ALLOYS = [CFG_316SS, CFG_HASTN, CFG_316H]

# ============================================
# Coupled PDE + PINN Runner
# ============================================
def run_pinn(cfg, seed, return_model=False):
    """
    Run a single coupled PINN estimation for one alloy and one seed.

    Args:
        cfg: Alloy configuration dictionary
        seed: Random seed for reproducibility

    Returns:
        Dictionary with D0, alpha, loss, EDS RMSE, mass ratio, audit, etc.
    """
    dde.config.set_random_seed(seed)
    dde.config.set_default_float("float64")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Unpack configuration ---
    C_bulk = cfg["C_bulk"]
    C_surface = cfg["C_surface"]
    C_range = C_bulk - C_surface
    rho = cfg["rho"]
    L_um = cfg["domain_um"]
    L_depth = L_um * 1e-4          # domain depth in cm
    T_max_hr = cfg["T_max_hr"]
    T_max_s = T_max_hr * 3600      # max time in seconds
    w_mass = cfg["w_mass"]

    # --- Prepare EDS data (sort, remove outliers, truncate to domain) ---
    x_all = cfg["depth_um"].copy()
    Cr_all = cfg["Cr_wt"].copy()
    # Pair-preserving sort by depth + removal of exact duplicate
    # digitization clicks (identical x AND Cr to machine precision)
    x_all, Cr_all = np.unique(np.column_stack([x_all, Cr_all]), axis=0).T
    # Outlier removal (316H only)
    if cfg["outlier_fn"] is not None:
        keep = [cfg["outlier_fn"](x, cr) for x, cr in zip(x_all, Cr_all)]
        x_all = x_all[keep]
        Cr_all = Cr_all[keep]
    # Clamp negative depths to zero
    x_all = np.maximum(x_all, 0.0)
    # Truncate to computational domain
    mask = x_all <= L_um
    x_um = x_all[mask]
    Cr_meas = Cr_all[mask]
    # Normalize: x to [0,1], C to [0,1]
    x_norm = x_um * 1e-4 / L_depth
    C_norm = np.clip((Cr_meas - C_surface) / C_range, 0, 1)

    # --- Prepare mass loss data ---
    conversion_factor = rho * (C_range / 100) * L_depth * 1000  # mg/cm²
    dW = cfg["dW_data"]
    t_hr = cfg["mass_times_hr"]
    w_target = dW / conversion_factor
    t_norm = t_hr * 3600 / T_max_s

    # Observation points for EDS: (x, t=1.0)
    n_eds = len(x_norm)
    obs_sp = np.hstack([x_norm.reshape(-1, 1), np.full((n_eds, 1), 1.0)])
    obs_val = C_norm.reshape(-1, 1)

    # Observation points for mass: (x=1.0, t)
    n_mass = len(t_hr)
    obs_mass = np.hstack([np.full((n_mass, 1), 1.0), t_norm.reshape(-1, 1)])
    obs_mass_val = w_target.reshape(-1, 1)

    # --- Trainable parameters ---
    log10_D_raw = dde.Variable(-14.0)     # log10(D0) in cm²/s
    alpha_est = dde.Variable(1.0)         # concentration-dependence parameter

    # --- Coupled PDE system ---
    def pde_coupled(x, y):
        u = y[:, 0:1]
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_x = dde.grad.jacobian(y, x, i=1, j=0)
        log10_D = torch.clamp(log10_D_raw, min=-15.0)
        D_base = 10.0 ** log10_D
        D_C = D_base * (1.0 + alpha_est * (1.0 - u))
        scale = T_max_s / (L_depth ** 2)
        D_norm = D_C * scale
        dD_dC = -D_base * alpha_est * scale
        # Eq1: Nonlinear diffusion
        eq_diff = du_t - (D_norm * du_xx + dD_dC * du_x * du_x)
        # Eq2: Integral definition (dw/dx = 1 - u)
        eq_int = dw_x - (1.0 - u)
        return [eq_diff, eq_int]

    # --- Domain, BCs, IC ---
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    bcs = [
        dde.icbc.DirichletBC(geomtime, lambda _: 0,
            lambda x, on_b: on_b and np.isclose(x[0], 0), component=0),
        dde.icbc.DirichletBC(geomtime, lambda _: 1,
            lambda x, on_b: on_b and np.isclose(x[0], 1), component=0),
        dde.icbc.IC(geomtime, lambda _: 1.0,
            lambda _, on_i: on_i, component=0),
        dde.icbc.DirichletBC(geomtime, lambda _: 0,
            lambda x, on_b: on_b and np.isclose(x[0], 0), component=1),
        dde.icbc.PointSetBC(obs_sp, obs_val, component=0),
        dde.icbc.PointSetBC(obs_mass, obs_mass_val, component=1),
    ]

    loss_weights = [1, 1, 1, 1, 1, 1, 1.0, float(w_mass)]
    data = dde.data.TimePDE(
        geomtime, pde_coupled, bcs,
        num_domain=1000, num_boundary=100, num_initial=100,
        anchors=np.vstack([obs_sp, obs_mass])
    )
    net = dde.nn.FNN([2] + [40] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # --- Training: Phase 1 (Adam lr=1e-3) + Phase 2 (Adam lr=1e-4) + Phase 3 (L-BFGS) ---
    # Phase 1: Large steps to approach the valley (15k iterations)
    model.compile("adam", lr=1e-3, loss_weights=loss_weights,
                  external_trainable_variables=[log10_D_raw, alpha_est])
    model.train(iterations=15000, display_every=5000)

    # Phase 2: Small steps to settle into the valley floor (10k iterations)
    model.compile("adam", lr=1e-4, loss_weights=loss_weights,
                  external_trainable_variables=[log10_D_raw, alpha_est])
    model.train(iterations=10000, display_every=5000)

    # Phase 3: L-BFGS with convergence check + conditional restart
    adam_total = 25000
    model.compile("L-BFGS", loss_weights=loss_weights,
                  external_trainable_variables=[log10_D_raw, alpha_est])
    losshistory, train_state = model.train()
    lbfgs_steps = losshistory.steps[-1] - adam_total

    retries = 0
    while lbfgs_steps < 50 and retries < 2:
        model.compile("adam", lr=1e-4, loss_weights=loss_weights,
                      external_trainable_variables=[log10_D_raw, alpha_est])
        model.train(iterations=5000, display_every=5000)
        adam_total += 5000
        model.compile("L-BFGS", loss_weights=loss_weights,
                      external_trainable_variables=[log10_D_raw, alpha_est])
        losshistory, train_state = model.train()
        lbfgs_steps = losshistory.steps[-1] - adam_total
        retries += 1

    converged = lbfgs_steps >= 50

    # --- Extract estimated parameters ---
    D0 = 10 ** torch.clamp(log10_D_raw, min=-15.0).detach().cpu().numpy().item()
    alpha = alpha_est.detach().cpu().numpy().item()
    D_wall = D0 * (1 + alpha)
    loss = float(sum(train_state.loss_test))

    # --- Compute EDS RMSE ---
    query_eds = np.hstack([x_norm.reshape(-1, 1), np.full((n_eds, 1), 1.0)])
    y_pred_eds = model.predict(query_eds)
    u_pred = y_pred_eds[:, 0]
    Cr_pred = u_pred * C_range + C_surface
    eds_rmse = np.sqrt(np.mean((Cr_pred - Cr_meas) ** 2))
    eds_nrmse = eds_rmse / C_range * 100

    # --- Compute mass prediction ---
    query_mass = np.hstack([np.full((n_mass, 1), 1.0), t_norm.reshape(-1, 1)])
    y_pred_mass = model.predict(query_mass)
    w_pred = y_pred_mass[:, 1]
    dW_pred = w_pred * conversion_factor
    mass_rmse = np.sqrt(np.mean((dW_pred - dW) ** 2))
    mass_ratio = np.mean(dW_pred / dW) if np.all(dW > 0) else 0.0

    # --- A-PINN consistency audit (dense grid, t = 1) ---
    x_dense = np.linspace(0.0, 1.0, 2001).reshape(-1, 1)
    grid = np.hstack([x_dense, np.ones_like(x_dense)])
    y_d = model.predict(grid)
    u_d, w_d = y_d[:, 0], y_d[:, 1]
    w_from_u = float(np.trapezoid(1.0 - u_d, x_dense.ravel()))  # requires numpy >= 2.0  # ∫(1−u)dx at t=1
    w_gap = float(w_d[-1] - w_d[0] - w_from_u)   # ≈0 iff dw/dx=1−u holds
    audit_pass = bool(abs(w_gap) < 0.05 and abs(float(w_d[0])) < 0.05)
    audit = {
        "w_pred_at_x1_t1": float(w_d[-1]),
        "w_pred_at_x0_t1": float(w_d[0]),
        "integral_1_minus_u": w_from_u,
        "w_gap": w_gap,
        "w_target": float(w_target[-1]),
        "u_min_on_grid": float(u_d.min()),
        "u_max_on_grid": float(u_d.max()),
        "audit_pass": audit_pass,
    }

    result = {
        "seed": seed,
        "D0": float(D0),
        "alpha": float(alpha),
        "D_wall": float(D_wall),
        "loss": float(loss),
        "eds_rmse": float(eds_rmse),
        "eds_nrmse": float(eds_nrmse),
        "mass_rmse": float(mass_rmse),
        "mass_ratio": float(mass_ratio),
        "n_eds_points": int(n_eds),
        "lbfgs_steps": int(lbfgs_steps),
        "retries": int(retries),
        "converged": bool(converged),
        "audit": audit,
    }
    if return_model:
        result["model"] = model
    return result

# ============================================
# Summary Output with Optimizer-State Filtering
# ============================================
# Basin-classification tolerances — PROTOCOL CONSTANTS for THIS ledger.
# They are not universal: a tolerance is only meaningful as the midpoint of a
# stability plateau — a band of tolerances over which the classification is
# invariant. For the datasets of this study the plateau is [0.15, 0.45] for
# every decisive separation (measured with basin_stability_sweep below); the
# single borderline case (one Hastelloy-N coupled seed at 0.30 vs 0.35) lies
# in a footnote-gated ensemble and is disclosed in the supplementary material.
# 0.3 in log10(D0) is a factor-of-two "same physical scale" criterion.
# FOR NEW ALLOYS OR DATASETS: do not inherit these numbers. Run
# basin_stability_sweep on your ensemble and re-establish the plateau; if no
# plateau exists, the ensemble has no discrete family structure (a continuum
# of equal-loss solutions) and should be reported as such.
BASIN_TOL_LOGD0 = 0.3
BASIN_TOL_ALPHA = 0.3


def basin_stability_sweep(conv, tols=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)):
    """Tolerance-sweep audit of the family classification.

    Returns [(tol, n_primary, n_secondary), ...] across the tolerance grid.
    A plateau (consecutive tolerances with identical partitions) justifies
    the protocol constants for a given dataset; its absence indicates a
    continuum rather than discrete families."""
    rows = []
    for t in tols:
        primary, secondary = classify_basins(conv, tol_logD0=t, tol_alpha=t)
        rows.append((t, len(primary), len(secondary)))
    return rows


def classify_basins(conv, tol_logD0=BASIN_TOL_LOGD0, tol_alpha=BASIN_TOL_ALPHA):
    """Greedy clustering of converged runs in (log10 D0, alpha) space.
    Loss is used only for ordering and for designating the primary basin."""
    basins = []
    for r in sorted(conv, key=lambda r: r["loss"]):   # ascending loss
        placed = False
        for b in basins:
            c = b[0]                                   # basin representative = lowest-loss run
            if (abs(np.log10(r["D0"]) - np.log10(c["D0"])) < tol_logD0
                    and abs(r["alpha"] - c["alpha"]) < tol_alpha):
                b.append(r); placed = True; break
        if not placed:
            basins.append([r])                          # new basin
    primary = basins[0] if basins else []               # the basin containing the best loss is primary
    secondary = [r for b in basins[1:] for r in b]
    return primary, secondary


def print_summary(alloy_name, temperature, duration, results, expected_sign):
    """Print detailed summary with two-stage filtering:
       Stage 1: Convergence by optimizer state (L-BFGS steps >= 50)
       Stage 2: Basin membership by parameter-space clustering in
                (log10 D0, alpha); loss used only for ordering and
                primary-basin designation.
    """
    n_total = len(results)

    # Stage 1: Validity by optimizer state
    conv = [r for r in results if r["converged"]]
    stalled = [r for r in results if not r["converged"]]

    # Stage 2: Basin selection among converged runs (parameter-space clustering)
    primary, secondary = classify_basins(conv)

    n_pri = len(primary)
    n_sec = len(secondary)
    n_stl = len(stalled)

    print(f"\n  {'='*62}")
    print(f"  {alloy_name} ({temperature}, {duration}) — Summary")
    print(f"  Converged: {len(conv)}/{n_total}, Stalled: {n_stl}/{n_total}, "
          f"Primary: {n_pri}, Secondary-basin: {n_sec}")
    print(f"  {'='*62}")

    # Per-seed table (with audit columns)
    all_sorted = primary + secondary + sorted(stalled, key=lambda r: r["loss"])
    print(f"  {'Seed':<6} {'α':>8} {'D0':>10} {'Loss':>10} "
          f"{'w(1,1)':>8} {'∫(1-u)':>8} {'w_gap':>8} {'Audit':>6} "
          f"{'L-BFGS':>7} {'Retry':>5} {'Status':>10}")
    print(f"  {'-'*105}")
    for r in all_sorted:
        if r in primary:
            status = "PRIMARY"
        elif r in secondary:
            status = "sec-basin"
        else:
            status = "STALLED"
        a = r["audit"]
        audit_str = "ok" if a["audit_pass"] else "FAIL"
        print(f"  {r['seed']:<6} {r['alpha']:>+8.3f} {r['D0']:>10.1e} "
              f"{r['loss']:>10.5f} "
              f"{a['w_pred_at_x1_t1']:>+8.3f} {a['integral_1_minus_u']:>+8.3f} "
              f"{a['w_gap']:>+8.3f} {audit_str:>6} "
              f"{r['lbfgs_steps']:>7d} {r['retries']:>5d} {status:>10}")

    # Aggregate statistics (primary only)
    if primary:
        D0s = [r["D0"] for r in primary]
        alphas = [r["alpha"] for r in primary]
        D_walls = [r["D_wall"] for r in primary]
        rmses = [r["eds_rmse"] for r in primary]
        nrmses = [r["eds_nrmse"] for r in primary]
        mass_rmses = [r["mass_rmse"] for r in primary]
        mass_ratios = [r["mass_ratio"] for r in primary]
        n_audit_ok = sum(1 for r in primary if r["audit"]["audit_pass"])
        if expected_sign == "negative":
            n_correct = sum(1 for a in alphas if a < 0)
            sign_label = "α<0"
        else:
            n_correct = sum(1 for a in alphas if a > 0)
            sign_label = "α>0"
        print(f"\n  D0,PINN: ({np.mean(D0s):.1e} ± {np.std(D0s):.1e}) cm²/s")
        print(f"  α:        {np.mean(alphas):+.3f} ± {np.std(alphas):.3f}")
        print(f"  D_wall:  ({np.mean(D_walls):.1e} ± {np.std(D_walls):.1e}) cm²/s")
        print(f"  EDS RMSE:  {np.mean(rmses):.2f} ± {np.std(rmses):.2f} wt% Cr")
        print(f"  NRMSE:     {np.mean(nrmses):.1f} ± {np.std(nrmses):.1f} %")
        print(f"  Mass RMSE: {np.mean(mass_rmses):.4f} ± {np.std(mass_rmses):.4f} mg/cm²")
        print(f"  Mass ratio:{np.mean(mass_ratios):.4f} ± {np.std(mass_ratios):.4f}")
        print(f"  Audit:     {n_audit_ok}/{n_pri} primary pass")
        print(f"  Sign:      {sign_label} in {n_correct}/{n_pri} primary")
    return primary, secondary, stalled

# ============================================
# Main Execution
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Coupled PINN Production Run — Three Alloys (with A-PINN audit)")
    print(f"  Protocol: {PROTOCOL_STR}")
    print(f"  Run tag: {RUN_TAG}")
    print(f"  Seeds: {len(SEEDS)} {SEEDS}")
    print("=" * 70)

    all_results = {}
    for cfg in ALL_ALLOYS:
        alloy = cfg["name"]
        all_results[alloy] = []
        print(f"\n\n{'#'*70}")
        print(f"  {alloy} / FLiBe / {cfg['temperature']} / {cfg['duration']}")
        print(f"  Domain: {cfg['domain_um']:.0f} μm, "
              f"C_bulk={cfg['C_bulk']}, C_surface={cfg['C_surface']}")
        print(f"{'#'*70}")
        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            r = run_pinn(cfg, seed)
            all_results[alloy].append(r)
            a = r["audit"]
            print(f"  D0={r['D0']:.1e}, α={r['alpha']:+.3f}, "
                  f"loss={r['loss']:.5f}, "
                  f"RMSE={r['eds_rmse']:.2f} wt%, "
                  f"L-BFGS={r['lbfgs_steps']}, retries={r['retries']}, "
                  f"conv={'Y' if r['converged'] else 'N'}")
            print(f"  AUDIT[{'ok' if a['audit_pass'] else 'FAIL'}]: "
                  f"w(1,1)={a['w_pred_at_x1_t1']:+.4f} vs "
                  f"∫(1-u)dx={a['integral_1_minus_u']:+.4f} "
                  f"(target {a['w_target']:.3f}) | "
                  f"w(0,1)={a['w_pred_at_x0_t1']:+.4f}, "
                  f"w_gap={a['w_gap']:+.4f}, "
                  f"u∈[{a['u_min_on_grid']:+.2f}, {a['u_max_on_grid']:+.2f}]")

    # ============================================
    # Summary
    # ============================================
    print(f"\n\n{'='*70}")
    print(f"  PRODUCTION RUN — COMPLETE SUMMARY")
    print(f"{'='*70}")
    expected_signs = {
        "316SS": "negative",
        "Hastelloy-N": "negative",
        "316H": "positive",
    }
    summary_data = {}
    for cfg in ALL_ALLOYS:
        alloy = cfg["name"]
        pri, sec, stl = print_summary(
            alloy, cfg["temperature"], cfg["duration"],
            all_results[alloy], expected_signs[alloy]
        )
        summary_data[alloy] = {"primary": pri, "secondary": sec, "stalled": stl}

    # Cross-alloy comparison table
    print(f"\n\n  {'='*70}")
    print(f"  CROSS-ALLOY COMPARISON (Primary Seeds Only)")
    print(f"  {'='*70}")
    print(f"  {'Alloy':<15} {'T':>5} {'α mean':>10} {'D0 mean':>12} "
          f"{'Sign':>10} {'Audit':>7} {'NRMSE':>8}")
    print(f"  {'-'*75}")
    for cfg in ALL_ALLOYS:
        alloy = cfg["name"]
        pri = summary_data[alloy]["primary"]
        if pri:
            alphas = [r["alpha"] for r in pri]
            D0s = [r["D0"] for r in pri]
            nrmses = [r["eds_nrmse"] for r in pri]
            n_ok = sum(1 for r in pri if r["audit"]["audit_pass"])
            exp = expected_signs[alloy]
            if exp == "negative":
                n_sign = sum(1 for a in alphas if a < 0)
                sign_str = f"{n_sign}/{len(pri)} α<0"
            else:
                n_sign = sum(1 for a in alphas if a > 0)
                sign_str = f"{n_sign}/{len(pri)} α>0"
            print(f"  {alloy:<15} {cfg['temperature']:>5} "
                  f"{np.mean(alphas):>+10.3f} {np.mean(D0s):>12.1e} "
                  f"{sign_str:>10} {n_ok}/{len(pri):>4} {np.mean(nrmses):>7.1f}%")
    print(f"  {'='*70}")

    # Save results as JSON
    try:
        output = {}
        for cfg in ALL_ALLOYS:
            alloy = cfg["name"]
            output[alloy] = {
                "temperature": cfg["temperature"],
                "duration": cfg["duration"],
                "domain_um": cfg["domain_um"],
                "C_bulk": cfg["C_bulk"],
                "C_surface": cfg["C_surface"],
                "protocol": PROTOCOL_STR,
                "float_precision": "float64",
                "run_tag": RUN_TAG,
                "seeds_run": SEEDS,
                "seeds": all_results[alloy],
            }
        out_name = f"production_results_float64_{RUN_TAG}.json"
        with open(out_name, "w") as f:
            json.dump(output, f, indent=2, default=float)
        print(f"\n  Results saved to {out_name}")
    except Exception as e:
        print(f"\n  Warning: Could not save JSON: {e}")
