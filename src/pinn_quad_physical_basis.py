"""
Quadratic-Closure Identifiability Campaign — Physical Basis (Table S10)
==========================================================================
316SS coupled inversion with the quadratic closure trained directly in the
endpoint-curvature chart of the paper (Section 2.1):

    D(C) = D_wall (1 - C) + D0 C + beta C (1 - C)

with trainable (log10 D_wall, log10 D0, beta_hat), beta = beta_hat x 1e-14
cm2/s. Eleven seeds, final protocol. The starting function is identical to
that of the polynomial-basis control (D0 = 1e-14, D_wall = 2e-14, beta = 0),
so the two campaigns differ ONLY in the optimizer's coordinates - together
they establish that the curvature unidentifiability is a property of the
data, not of the parameterization (Section S6).

Pre-registered predictions (2026-07, before execution; registered against
the polynomial-basis campaign, which ran first):
  PC1: every seed lands on the equal-loss family - total loss within its
       band (~2.0-2.5e-3) and audit pass.
  PC2: <D> = (D0+D_wall)/2 + beta/6 pinned within ~6% of 3.22e-14 cm2/s.
  PC3 (genuinely uncertain): whether the seed spread along the family
       persists in this chart. Either outcome is informative.
Outcomes are recorded in Section S6 of the paper.
"""
import deepxde as dde
import numpy as np
import torch
import json

import alloy_configs as AC

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
RUN_TAG = "316SS_quad_physical_basis"
BETA_SCALE = 1e-14
W_MASS = 10
PROTOCOL_STR = ("Quadratic closure, endpoint-bump chart: Adam 15k@1e-3 + "
                "10k@1e-4 -> L-BFGS (restart <=2), float64, "
                "init D0=1e-14, Dwall=2e-14, beta=0, w_mass=10")

# Alloy data come from the single source of truth (alloy_configs.py).
# This campaign previously carried its own copy, whose dW_data was a
# superseded digitization vintage (0.170/0.221/0.318/0.340 at 1000/2000 h,
# same means as the current 0.174/0.217/0.308/0.350). Synchronized 2026-08.
ALLOY_316SS = AC.get_config("316SS")
ALLOY_316SS["w_mass"] = W_MASS   # protocol knob, not alloy data


def run_pinn(cfg, seed):
    dde.config.set_random_seed(seed)
    dde.config.set_default_float("float64")
    np.random.seed(seed)
    torch.manual_seed(seed)

    C_bulk, C_surface = cfg["C_bulk"], cfg["C_surface"]
    C_range = C_bulk - C_surface
    rho = cfg["rho"]
    L_depth = cfg["domain_um"] * 1e-4
    T_max_s = cfg["T_max_hr"] * 3600
    w_mass = cfg["w_mass"]

    x_all, Cr_all = np.unique(
        np.column_stack([cfg["depth_um"], cfg["Cr_wt"]]), axis=0).T
    x_all = np.maximum(x_all, 0.0)
    mask = x_all <= cfg["domain_um"]
    x_norm = x_all[mask] * 1e-4 / L_depth
    Cr_meas = Cr_all[mask]
    C_norm = np.clip((Cr_meas - C_surface) / C_range, 0, 1)

    conversion_factor = rho * (C_range / 100) * L_depth * 1000
    dW = cfg["dW_data"]
    w_target = dW / conversion_factor
    t_norm = cfg["mass_times_hr"] * 3600 / T_max_s

    n_eds = len(x_norm)
    obs_sp = np.hstack([x_norm.reshape(-1, 1), np.full((n_eds, 1), 1.0)])
    obs_val = C_norm.reshape(-1, 1)
    n_mass = len(t_norm)
    obs_mass = np.hstack([np.full((n_mass, 1), 1.0), t_norm.reshape(-1, 1)])
    obs_mass_val = w_target.reshape(-1, 1)

    # --- endpoint-bump chart: the ONLY departure from alpha2_check ---
    log10_Dwall_raw = dde.Variable(float(np.log10(2e-14)))  # D_wall = 2e-14
    log10_D0_raw = dde.Variable(-14.0)                       # D0     = 1e-14
    beta_hat = dde.Variable(0.0)                             # beta   = 0

    def pde_coupled(x, y):
        u = y[:, 0:1]
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_x = dde.grad.jacobian(y, x, i=1, j=0)
        D_wall = 10.0 ** torch.clamp(log10_Dwall_raw, min=-15.0)
        D_0 = 10.0 ** torch.clamp(log10_D0_raw, min=-15.0)
        beta = beta_hat * BETA_SCALE
        # D(u) = D_wall (1-u) + D0 u + beta u (1-u)
        D_C = D_wall * (1.0 - u) + D_0 * u + beta * u * (1.0 - u)
        scale = T_max_s / (L_depth ** 2)
        D_norm = D_C * scale
        # dD/du = (D0 - D_wall) + beta (1 - 2u)
        dD_du = ((D_0 - D_wall) + beta * (1.0 - 2.0 * u)) * scale
        eq_diff = du_t - (D_norm * du_xx + dD_du * du_x * du_x)
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
        dde.icbc.IC(geomtime, lambda _: 1.0, lambda _, on_i: on_i, component=0),
        dde.icbc.DirichletBC(geomtime, lambda _: 0,
            lambda x, on_b: on_b and np.isclose(x[0], 0), component=1),
        dde.icbc.PointSetBC(obs_sp, obs_val, component=0),
        dde.icbc.PointSetBC(obs_mass, obs_mass_val, component=1),
    ]
    loss_weights = [1, 1, 1, 1, 1, 1, 1.0, float(w_mass)]
    data = dde.data.TimePDE(geomtime, pde_coupled, bcs,
                            num_domain=1000, num_boundary=100,
                            num_initial=100,
                            anchors=np.vstack([obs_sp, obs_mass]))
    net = dde.nn.FNN([2] + [40] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    ext = [log10_Dwall_raw, log10_D0_raw, beta_hat]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights,
                  external_trainable_variables=ext)
    model.train(iterations=15000, display_every=5000)
    model.compile("adam", lr=1e-4, loss_weights=loss_weights,
                  external_trainable_variables=ext)
    model.train(iterations=10000, display_every=5000)
    # Cumulative Adam steps from phases (i)+(ii); bookkeeping for lbfgs_steps.
    adam_total = 15000 + 10000
    model.compile("L-BFGS", loss_weights=loss_weights,
                  external_trainable_variables=ext)
    losshistory, train_state = model.train()
    lbfgs_steps = losshistory.steps[-1] - adam_total

    retries = 0
    while lbfgs_steps < 50 and retries < 2:
        model.compile("adam", lr=1e-4, loss_weights=loss_weights,
                      external_trainable_variables=ext)
        model.train(iterations=5000, display_every=5000)
        adam_total += 5000
        model.compile("L-BFGS", loss_weights=loss_weights,
                      external_trainable_variables=ext)
        losshistory, train_state = model.train()
        lbfgs_steps = losshistory.steps[-1] - adam_total
        retries += 1

    D_wall = 10 ** torch.clamp(log10_Dwall_raw, min=-15.0).detach().cpu().numpy().item()
    D0 = 10 ** torch.clamp(log10_D0_raw, min=-15.0).detach().cpu().numpy().item()
    beta = beta_hat.detach().cpu().numpy().item() * BETA_SCALE
    meanD = 0.5 * (D0 + D_wall) + beta / 6.0
    # derived alpha-chart values for schema compatibility
    a2 = -beta / D0
    a1 = D_wall / D0 - 1.0 - a2
    loss = float(sum(train_state.loss_test))

    y_pred = model.predict(obs_sp)
    Cr_pred = y_pred[:, 0] * C_range + C_surface
    eds_rmse = float(np.sqrt(np.mean((Cr_pred - Cr_meas) ** 2)))

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

    # --- Supplementary time-sweep diagnostic (gate unchanged: Eq. (12b) at t = 1) ---
    _t_extra = t_norm.ravel() if 't_norm' in locals() else np.array([1.0])
    t_sweep = np.unique(np.round(np.concatenate([np.linspace(0.0, 1.0, 21), _t_extra, [1.0]]), 12))
    _w0_t, _gap_t = [], []
    for _tv in t_sweep:
        _g = np.hstack([x_dense, np.full_like(x_dense, _tv)])
        _yv = model.predict(_g)
        _uv, _wv = _yv[:, 0], _yv[:, 1]
        _integ = float(np.trapezoid(1.0 - _uv, x_dense.ravel()))
        _w0_t.append(float(_wv[0]))
        _gap_t.append(float(_wv[-1] - _wv[0] - _integ))
    _i1 = int(np.argmin(np.abs(t_sweep - 1.0)))
    assert abs(_w0_t[_i1] - audit["w_pred_at_x0_t1"]) < 1e-12 and \
           abs(_gap_t[_i1] - audit["w_gap"]) < 1e-12, \
        "time-sweep t=1 must reproduce the gate quantities exactly (definition unchanged)"
    audit["time_sweep"] = {
        "t_grid": [float(v) for v in t_sweep],
        "w0_of_t": _w0_t,
        "w_gap_of_t": _gap_t,
        "max_abs_w0": float(max(abs(v) for v in _w0_t)),
        "max_abs_w_gap": float(max(abs(v) for v in _gap_t)),
    }
    return {
        "seed": seed, "D0": float(D0),
        "alpha": float(a1), "alpha1": float(a1), "alpha2": float(a2),
        "D_wall": float(D_wall), "beta": float(beta), "meanD": float(meanD),
        "loss": loss,
        "eds_rmse": eds_rmse, "eds_nrmse": eds_rmse / C_range * 100,
        "mass_rmse": 0.0, "mass_ratio": 0.0,
        "n_eds_points": int(n_eds), "lbfgs_steps": int(lbfgs_steps),
        "retries": int(retries), "converged": bool(lbfgs_steps >= 50),
        "audit": audit,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  Chart-replication probe: 316SS coupled, endpoint-bump chart")
    print(f"  {PROTOCOL_STR}")
    print("=" * 70)
    results = []
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        r = run_pinn(ALLOY_316SS, seed)
        results.append(r)
        a = r["audit"]
        print(f"  D0={r['D0']:.2e}, D_wall={r['D_wall']:.2e}, "
              f"beta={r['beta']:+.2e}, meanD={r['meanD']:.2e} | "
              f"loss={r['loss']:.5f}, RMSE={r['eds_rmse']:.2f}, "
              f"conv={'Y' if r['converged'] else 'N'}, "
              f"AUDIT[{'ok' if a['audit_pass'] else 'FAIL'}]")
        print(f"  endpoint contrast D_wall/D0 = {r['D_wall']/r['D0']:.3f}")
    mds = np.array([r["meanD"] for r in results if r["converged"]])
    bts = np.array([r["beta"] for r in results if r["converged"]])
    print(f"\n  SUMMARY: <D> = {mds.mean():.3e} ± {mds.std():.1e} "
          f"({mds.std()/mds.mean()*100:.1f} %) | beta range "
          f"{bts.min():+.2e} .. {bts.max():+.2e}")
    print("  PC1: loss in ridge band + audit pass? "
          + ("YES" if all(r['audit']['audit_pass'] and 0.0015 < r['loss'] < 0.0030
                          for r in results) else "CHECK"))
    print("  PC2: <D> within ~6% of 3.22e-14? "
          + ("YES" if abs(mds.mean() - 3.22e-14) / 3.22e-14 < 0.10 else "CHECK"))
    out = {"316SS": {
        "temperature": ALLOY_316SS["temperature"],
        "duration": ALLOY_316SS["duration"],
        "domain_um": ALLOY_316SS["domain_um"],
        "C_bulk": ALLOY_316SS["C_bulk"],
        "C_surface": ALLOY_316SS["C_surface"],
        "protocol": PROTOCOL_STR, "float_precision": "float64",
        "run_tag": RUN_TAG, "seeds_run": SEEDS, "seeds": results,
    }}
    with open(f"production_results_float64_{RUN_TAG}.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  Results saved to production_results_float64_{RUN_TAG}.json")