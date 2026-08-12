"""
Coupled PINN — EDS-Only Final Campaign + Initial-Value Control
==========================================================================
Two campaigns in one script (each writes its own tagged JSON):

  Campaign 1  "eds_only_standalone":
      Hastelloy-N + 316H, all 11 seeds, α₀ = +1.0.
      Purpose: promote the EDS-only estimates to FINAL REPORTED VALUES for
      the two audit-FAIL alloys (gated reporting: coupled-run parameters
      for these alloys carry residual mass-channel pressure and are quoted
      only as footnotes).

  Campaign 2  "eds_only_316H_alpha0_minus1":
      316H only, seeds [2, 3, 4], α₀ = −1.0.
      Purpose: decisive initial-value-bias control.

Pre-registered predictions (2026-07):
  P4. 316H started from α₀ = −1.0 climbs back to α ≈ +0.70 (the profile-
      intrinsic optimum), crossing zero. If it instead remains negative,
      the positive sign in Campaign 1 would be attributable to initial-
      value bias / a secondary basin — falsifying the profile-intrinsic
      interpretation.
  P5. Campaign 1 seed statistics remain as tight as the 3-seed pilot
      (316H: α = +0.70 with ≲0.01 scatter; HN: α ≈ −0.72 ± 0.03), and
      audit passes 22/22.

Protocol identical to production except the mass anchor is absent and,
in Campaign 2, the α initial value:
  Adam 15k@1e-3 + 10k@1e-4 → L-BFGS (restart ≤2), float64.
Loss vector (7 components): [eq_diff, eq_int, bc_u0, bc_u1, ic, bc_w0, eds]
`mass_rmse` / `mass_ratio` are post-hoc PREDICTIONS (not fitted).
JSON schema matches production; process with tools/make_tables.py.
"""
import deepxde as dde
import numpy as np
import torch
import json
import sys

# ============================================
# Campaign Configuration
# ============================================
SEEDS_FULL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
SEEDS_CTRL = [2, 3, 4]

# ============================================================
# Alloy configurations — imported from the single source of truth.
# REFACTORED (2026-07): see alloy_configs.py. EDS-only mode has no
# mass anchor, so no w_mass exists in these runtime configurations;
# mass_rmse / mass_ratio remain post-hoc PREDICTIONS computed against
# the raw gravimetric records (for 316H, the raw replicate pair).
# ============================================================
import alloy_configs as AC

ALLOY_HASTN = AC.get_config("Hastelloy-N")
ALLOY_316H = AC.get_config("316H")

CAMPAIGNS = [
    {
        "tag": "eds_only_standalone",
        "alloys": [ALLOY_HASTN, ALLOY_316H],
        "seeds": SEEDS_FULL,
        "alpha0": +1.0,
    },
    {
        "tag": "eds_only_316H_alpha0_minus1",
        "alloys": [ALLOY_316H],
        "seeds": SEEDS_CTRL,
        "alpha0": -1.0,
    },
]

# ============================================
# EDS-only PINN Runner (α₀ parameterized)
# ============================================
def run_pinn(cfg, seed, alpha0=1.0, return_model=False):
    """Single EDS-only inverse estimation; alpha0 sets the α initial value."""
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
    L_depth = L_um * 1e-4          # cm
    T_max_hr = cfg["T_max_hr"]
    T_max_s = T_max_hr * 3600

    # --- Prepare EDS data (sort/dedup, remove outliers, truncate) ---
    x_all = cfg["depth_um"].copy()
    Cr_all = cfg["Cr_wt"].copy()
    x_all, Cr_all = np.unique(np.column_stack([x_all, Cr_all]), axis=0).T
    if cfg["outlier_fn"] is not None:
        keep = [cfg["outlier_fn"](x, cr) for x, cr in zip(x_all, Cr_all)]
        x_all = x_all[keep]
        Cr_all = Cr_all[keep]
    x_all = np.maximum(x_all, 0.0)
    mask = x_all <= L_um
    x_um = x_all[mask]
    Cr_meas = Cr_all[mask]
    x_norm = x_um * 1e-4 / L_depth
    C_norm = np.clip((Cr_meas - C_surface) / C_range, 0, 1)

    # --- Mass reference (PREDICTION target only; NOT fitted) ---
    conversion_factor = rho * (C_range / 100) * L_depth * 1000  # mg/cm²
    dW = cfg["dW_data"]
    t_hr = cfg["mass_times_hr"]
    w_target = dW / conversion_factor
    t_norm = t_hr * 3600 / T_max_s

    n_eds = len(x_norm)
    obs_sp = np.hstack([x_norm.reshape(-1, 1), np.full((n_eds, 1), 1.0)])
    obs_val = C_norm.reshape(-1, 1)
    n_mass = len(t_hr)

    # --- Trainable parameters ---
    log10_D_raw = dde.Variable(-14.0)
    alpha_est = dde.Variable(float(alpha0))

    # --- Coupled PDE system (w channel retained for the audit) ---
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
        eq_diff = du_t - (D_norm * du_xx + dD_dC * du_x * du_x)
        eq_int = dw_x - (1.0 - u)
        return [eq_diff, eq_int]

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
    ]

    # 7-component loss: [eq_diff, eq_int, bc_u0, bc_u1, ic, bc_w0, eds]
    loss_weights = [1, 1, 1, 1, 1, 1, 1.0]
    data = dde.data.TimePDE(
        geomtime, pde_coupled, bcs,
        num_domain=1000, num_boundary=100, num_initial=100,
        anchors=obs_sp
    )
    net = dde.nn.FNN([2] + [40] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # --- Three-phase protocol ---
    model.compile("adam", lr=1e-3, loss_weights=loss_weights,
                  external_trainable_variables=[log10_D_raw, alpha_est])
    model.train(iterations=15000, display_every=5000)

    model.compile("adam", lr=1e-4, loss_weights=loss_weights,
                  external_trainable_variables=[log10_D_raw, alpha_est])
    model.train(iterations=10000, display_every=5000)

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

    # --- EDS RMSE ---
    query_eds = np.hstack([x_norm.reshape(-1, 1), np.full((n_eds, 1), 1.0)])
    y_pred_eds = model.predict(query_eds)
    u_pred = y_pred_eds[:, 0]
    Cr_pred = u_pred * C_range + C_surface
    eds_rmse = np.sqrt(np.mean((Cr_pred - Cr_meas) ** 2))
    eds_nrmse = eds_rmse / C_range * 100

    # --- Mass PREDICTION (not fitted) ---
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
    w_from_u = float(np.trapezoid(1.0 - u_d, x_dense.ravel()))
    w_gap = float(w_d[-1] - w_d[0] - w_from_u)
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
        "alpha0": float(alpha0),
        "D0": float(D0),
        "alpha": float(alpha),
        "D_wall": float(D_wall),
        "loss": float(loss),
        "eds_rmse": float(eds_rmse),
        "eds_nrmse": float(eds_nrmse),
        "mass_rmse": float(mass_rmse),      # prediction error, NOT a fit
        "mass_ratio": float(mass_ratio),    # predicted/measured, NOT a fit
        "n_eds_points": int(n_eds),
        "lbfgs_steps": int(lbfgs_steps),
        "retries": int(retries),
        "converged": bool(converged),
        "audit": audit,
    }
    if return_model:
        result["model"] = model
    return result


if __name__ == "__main__":
    for camp in CAMPAIGNS:
        tag = camp["tag"]
        alpha0 = camp["alpha0"]
        seeds = camp["seeds"]
        protocol = (f"EDS-only (no mass anchor): Adam 15k@1e-3 + 10k@1e-4 → "
                    f"L-BFGS (restart ≤2), float64, α₀={alpha0:+.1f}")
        print("=" * 70)
        print(f"  CAMPAIGN: {tag}")
        print(f"  Protocol: {protocol}")
        print(f"  Seeds: {len(seeds)} {seeds}")
        print("=" * 70)

        all_results = {}
        for cfg in camp["alloys"]:
            alloy = cfg["name"]
            all_results[alloy] = []
            print(f"\n\n{'#'*70}")
            print(f"  {alloy} / FLiBe / {cfg['temperature']} / "
                  f"{cfg['duration']} — EDS ONLY, α₀={alpha0:+.1f}")
            print(f"{'#'*70}")
            for seed in seeds:
                print(f"\n  --- Seed {seed} ---")
                r = run_pinn(cfg, seed, alpha0=alpha0)
                all_results[alloy].append(r)
                a = r["audit"]
                print(f"  D0={r['D0']:.1e}, α={r['alpha']:+.3f}, "
                      f"loss={r['loss']:.5f}, RMSE={r['eds_rmse']:.2f} wt%, "
                      f"L-BFGS={r['lbfgs_steps']}, retries={r['retries']}, "
                      f"conv={'Y' if r['converged'] else 'N'}")
                print(f"  AUDIT[{'ok' if a['audit_pass'] else 'FAIL'}]: "
                      f"w(1,1)={a['w_pred_at_x1_t1']:+.4f} vs "
                      f"∫(1-u)dx={a['integral_1_minus_u']:+.4f} "
                      f"(gravimetric ref {a['w_target']:.3f}) | "
                      f"w(0,1)={a['w_pred_at_x0_t1']:+.4f}, "
                      f"w_gap={a['w_gap']:+.4f}")
                print(f"  MASS PREDICTION (not fitted): "
                      f"predicted/measured = {r['mass_ratio']:.3f}")

        # Save per-campaign JSON (schema-compatible with make_tables.py)
        try:
            output = {}
            for cfg in camp["alloys"]:
                alloy = cfg["name"]
                output[alloy] = {
                    "temperature": cfg["temperature"],
                    "duration": cfg["duration"],
                    "domain_um": cfg["domain_um"],
                    "C_bulk": cfg["C_bulk"],
                    "C_surface": cfg["C_surface"],
                    "protocol": protocol,
                    "float_precision": "float64",
                    "run_tag": tag,
                    "alpha0": alpha0,
                    "seeds_run": seeds,
                    "seeds": all_results[alloy],
                }
            out_name = f"production_results_float64_{tag}.json"
            with open(out_name, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=float)
            print(f"\n  Results saved to {out_name}")
        except Exception as e:
            print(f"\n  Warning: Could not save JSON: {e}")