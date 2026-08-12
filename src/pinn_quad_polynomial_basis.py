"""
Parameterization-Robustness Control — Polynomial Basis (Table S4b)
==========================================================================
316SS coupled inversion with the quadratic closure trained in a standard
polynomial expansion,

    D(C) = D0 [1 + a1 (1 - C) + a2 (1 - C)^2],

as the basis-independence control for the physical-basis campaign of
Table S4a (pinn_quad_physical_basis.py): identical starting function
(a1 = +1, a2 = 0  <=>  D_wall = 2e-14, beta = 0), identical protocol,
different optimizer coordinates. The raw coefficients (a1, a2) are
transformed to the endpoint-curvature set (D0, D_wall, beta) for
tabulation: D_wall = D0 (1 + a1 + a2), beta = -D0 a2.

Pre-registered expectation (2026-07): a2 statistically indistinguishable
from zero (legacy-protocol result: -0.040 +/- 0.156); sign of a1 negative
in 11/11. The outcome - a continuous equal-loss family rather than a
collapse to zero - is recorded in Section S4 of the paper.
Final protocol: float64, unified three-phase training, audit; w_mass = 10.
"""
import deepxde as dde
import numpy as np
import torch
import json

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]
RUN_TAG = "316SS_quad_polynomial_basis"
PROTOCOL_STR = ("Quadratic closure check: Adam 15k@1e-3 + 10k@1e-4 → L-BFGS "
                "(restart ≤2), float64, α₁₀=+1.0, α₂₀=0.0, w_mass=10")

ALLOY_316SS = {
    "name": "316SS",
    "temperature": "700°C",
    "duration": "3000h",
    "C_bulk": 17.0,
    "C_surface": 1.2,
    "rho": 8.0,
    "domain_um": 80.0,
    "T_max_hr": 3000.0,
    "w_mass": 10,
    "depth_um": np.array([
        0.000, 1.170, 3.029, 4.301, 6.259, 8.017, 9.683, 11.346, 13.106,
        14.669, 16.336, 18.000, 19.763, 21.426, 23.092, 24.757, 26.420,
        27.989, 29.848, 31.316, 33.175, 34.840, 36.607, 38.072, 39.739,
        41.403, 43.166, 44.829, 46.591, 49.725, 51.489, 53.252, 54.914,
        56.579, 58.245, 59.910, 61.671, 63.337, 65.000, 66.567, 68.525,
        70.093, 71.759, 73.325, 74.990, 76.752, 78.517, 79.985
    ]),
    "Cr_wt": np.array([
        1.187, 5.479, 6.814, 7.704, 8.595, 12.297, 11.264, 13.338, 15.709,
        19.262, 16.602, 17.789, 17.496, 18.682, 17.206, 17.505, 18.248,
        16.772, 17.958, 18.257, 19.888, 19.595, 16.944, 18.713, 16.645,
        16.652, 18.134, 18.729, 14.864, 18.143, 16.815, 16.670, 18.744,
        19.339, 17.271, 17.570, 18.461, 17.281, 19.355, 18.618, 18.770,
        17.589, 16.707, 16.705, 16.119, 17.601, 14.942, 15.832
    ]),
    "mass_times_hr": np.array([1000.0, 1000.0, 2000.0, 2000.0, 3000.0, 3000.0]),
    "dW_data": np.array([0.170, 0.221, 0.318, 0.340, 0.456, 0.547]),
}


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

    log10_D_raw = dde.Variable(-14.0)
    alpha1_est = dde.Variable(1.0)
    alpha2_est = dde.Variable(0.0)

    def pde_coupled(x, y):
        u = y[:, 0:1]
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_x = dde.grad.jacobian(y, x, i=1, j=0)
        log10_D = torch.clamp(log10_D_raw, min=-15.0)
        D_base = 10.0 ** log10_D
        one_m_u = 1.0 - u
        D_C = D_base * (1.0 + alpha1_est * one_m_u + alpha2_est * one_m_u ** 2)
        scale = T_max_s / (L_depth ** 2)
        D_norm = D_C * scale
        # dD/du = -D_base (alpha1 + 2 alpha2 (1-u))  — u-dependent
        dD_du = -D_base * (alpha1_est + 2.0 * alpha2_est * one_m_u) * scale
        eq_diff = du_t - (D_norm * du_xx + dD_du * du_x * du_x)
        eq_int = dw_x - one_m_u
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

    ext = [log10_D_raw, alpha1_est, alpha2_est]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights,
                  external_trainable_variables=ext)
    model.train(iterations=15000, display_every=5000)
    model.compile("adam", lr=1e-4, loss_weights=loss_weights,
                  external_trainable_variables=ext)
    model.train(iterations=10000, display_every=5000)
    adam_total = 25000
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

    D0 = 10 ** torch.clamp(log10_D_raw, min=-15.0).detach().cpu().numpy().item()
    a1 = alpha1_est.detach().cpu().numpy().item()
    a2 = alpha2_est.detach().cpu().numpy().item()
    D_wall = D0 * (1 + a1 + a2)
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
    return {
        "seed": seed, "D0": float(D0),
        "alpha": float(a1), "alpha1": float(a1), "alpha2": float(a2),
        "D_wall": float(D_wall), "loss": loss,
        "eds_rmse": eds_rmse, "eds_nrmse": eds_rmse / C_range * 100,
        "mass_rmse": 0.0, "mass_ratio": 0.0,   # recomputed by tables if needed
        "n_eds_points": int(n_eds), "lbfgs_steps": int(lbfgs_steps),
        "retries": int(retries), "converged": bool(lbfgs_steps >= 50),
        "audit": audit,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  Quadratic-closure adequacy check (316SS, final protocol)")
    print(f"  {PROTOCOL_STR}")
    print("=" * 70)
    results = []
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        r = run_pinn(ALLOY_316SS, seed)
        results.append(r)
        a = r["audit"]
        print(f"  D0={r['D0']:.2e}, α1={r['alpha1']:+.3f}, "
              f"α2={r['alpha2']:+.3f}, D_wall={r['D_wall']:.2e}, "
              f"loss={r['loss']:.5f}, RMSE={r['eds_rmse']:.2f}, "
              f"conv={'Y' if r['converged'] else 'N'}, "
              f"AUDIT[{'ok' if a['audit_pass'] else 'FAIL'}]")
    a2s = np.array([r["alpha2"] for r in results if r["converged"]])
    a1s = np.array([r["alpha1"] for r in results if r["converged"]])
    print(f"\n  SUMMARY: α1 = {a1s.mean():+.3f} ± {a1s.std():.3f} | "
          f"α2 = {a2s.mean():+.3f} ± {a2s.std():.3f} "
          f"({'consistent with zero' if abs(a2s.mean()) < 2*a2s.std() else 'NON-ZERO — investigate'})")
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