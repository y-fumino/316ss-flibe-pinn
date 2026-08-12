"""
Quadratic-Closure Probes, EDS-Only — Physical Basis (Tables S10a–S10c)
==========================================================================
Derived verbatim from the certified pinn_quad_physical_basis.py (Table S4a
campaign); the ONLY departures are (1) the gravimetric anchor is absent —
the probes ask what survives when the EDS profile is the only data channel
— and (2) alloy configurations come from alloy_configs for the three-alloy
batch N9. Everything else (closure, chart, starting function, three-phase
training, audit) is identical.

Closure, trained directly in the endpoint-curvature chart of the paper:

    D(C) = D_wall (1 - C) + D0 C + beta C (1 - C)

with trainable (log10 D_wall, log10 D0, beta_hat), beta = beta_hat x 1e-14
cm2/s; starting function D0 = 1e-14, D_wall = 2e-14, beta = 0 (common to
every quadratic campaign of this study). The polynomial basis appears in
this study only as the declared robustness control of Table S4b.

Purpose: Tables S10a–S10b (Hastelloy-N, 316H) test whether the equal-loss
curvature freedom survives without the mass channel; Table S10c (316SS)
is the mode-matched disambiguation — whether removing the anchor dissolves
or preserves the degeneracy. mass_ratio is a post-hoc PREDICTION against
the unfitted gravimetric record (never fitted).

Run via: python src/run_all.py --batch N9   (seeds [2, 3, 4] per job)
"""
import deepxde as dde
import numpy as np
import torch

import alloy_configs as AC

ALLOY_316SS = AC.get_config("316SS")
ALLOY_HASTN = AC.get_config("Hastelloy-N")
ALLOY_316H = AC.get_config("316H")

BETA_SCALE = 1e-14
PROTOCOL_STR = ("Quadratic closure, endpoint-curvature chart, EDS-only: "
                "Adam 15k@1e-3 + 10k@1e-4 -> L-BFGS (restart <=2), float64, "
                "init D0=1e-14, Dwall=2e-14, beta=0, no mass anchor")


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

    x_all, Cr_all = np.unique(
        np.column_stack([cfg["depth_um"], cfg["Cr_wt"]]), axis=0).T
    x_all = np.maximum(x_all, 0.0)
    if cfg.get("outlier_fn") is not None:
        keep = [cfg["outlier_fn"](x, cr) for x, cr in zip(x_all, Cr_all)]
        x_all = x_all[keep]
        Cr_all = Cr_all[keep]
    mask = x_all <= cfg["domain_um"]
    x_norm = x_all[mask] * 1e-4 / L_depth
    Cr_meas = Cr_all[mask]
    C_norm = np.clip((Cr_meas - C_surface) / C_range, 0, 1)

    # gravimetric record retained for the post-hoc prediction ONLY (not fitted)
    conversion_factor = rho * (C_range / 100) * L_depth * 1000
    w_target = np.asarray(cfg["dW_data"]) / conversion_factor

    n_eds = len(x_norm)
    obs_sp = np.hstack([x_norm.reshape(-1, 1), np.full((n_eds, 1), 1.0)])
    obs_val = C_norm.reshape(-1, 1)

    # --- endpoint-curvature chart (identical to pinn_quad_physical_basis) ---
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
    # --- EDS-only: the mass PointSetBC of the coupled campaign is absent ---
    bcs = [
        dde.icbc.DirichletBC(geomtime, lambda _: 0,
            lambda x, on_b: on_b and np.isclose(x[0], 0), component=0),
        dde.icbc.DirichletBC(geomtime, lambda _: 1,
            lambda x, on_b: on_b and np.isclose(x[0], 1), component=0),
        dde.icbc.IC(geomtime, lambda _: 1.0, lambda _, on_i: on_i, component=0),
        dde.icbc.DirichletBC(geomtime, lambda _: 0,
            lambda x, on_b: on_b and np.isclose(x[0], 0), component=1),
        dde.icbc.PointSetBC(obs_sp, obs_val, component=0),
    ]
    loss_weights = [1, 1, 1, 1, 1, 1, 1.0]
    data = dde.data.TimePDE(geomtime, pde_coupled, bcs,
                            num_domain=1000, num_boundary=100,
                            num_initial=100, anchors=obs_sp)
    net = dde.nn.FNN([2] + [40] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    ext = [log10_Dwall_raw, log10_D0_raw, beta_hat]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights,
                  external_trainable_variables=ext)
    model.train(iterations=15000, display_every=5000)
    model.compile("adam", lr=1e-4, loss_weights=loss_weights,
                  external_trainable_variables=ext)
    model.train(iterations=10000, display_every=5000)
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
    mass_ratio = float(w_d[-1]) / float(w_target[-1]) if len(w_target) else 0.0
    audit = {
        "w_pred_at_x1_t1": float(w_d[-1]),
        "w_pred_at_x0_t1": float(w_d[0]),
        "integral_1_minus_u": w_from_u,
        "w_gap": w_gap,
        "w_target": float(w_target[-1]) if len(w_target) else 0.0,
        "u_min_on_grid": float(u_d.min()),
        "u_max_on_grid": float(u_d.max()),
        "audit_pass": audit_pass,
    }
    return {
        "seed": seed, "D0": float(D0),
        "alpha": float(a1), "alpha1": float(a1), "alpha2": float(a2),
        "D_wall": float(D_wall), "beta": float(beta), "meanD": float(meanD),
        "loss": loss,
        "eds_rmse": eds_rmse, "eds_nrmse": eds_rmse / C_range * 100,
        "mass_rmse": 0.0, "mass_ratio": mass_ratio,
        "n_eds_points": int(n_eds), "lbfgs_steps": int(lbfgs_steps),
        "retries": int(retries), "converged": bool(lbfgs_steps >= 50),
        "audit": audit,
    }


if __name__ == "__main__":
    print("Run via the orchestrator: python src/run_all.py --batch N9")