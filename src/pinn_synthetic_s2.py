"""
Synthetic S2 Verification Harness — Four Quadrants and Inflation Sweep
==========================================================================
Re-establishes the synthetic sign-integrity results (Section S2) under the
final production protocol.

Design:
  - DATA GENERATION is transplanted verbatim from the original synthetic
    study: an FDM forward solve of the same nonlinear diffusion law with
    known ground truth (D0 = 4.3e-14 cm^2/s, alpha = -0.5), subsampled and
    perturbed with a fixed noise realization (np.random.seed(99)) so that
    the synthetic dataset is identical for every seed and every case size,
    exactly as a real measured dataset would be.
  - THE INVERSE SOLVER IS NOT DUPLICATED HERE. Each case is packed into a
    standard configuration dictionary and solved by
    pinn_production_coupled.run_pinn — the same verified solver, protocol
    (float64, Adam 15k@1e-3 + 10k@1e-4 -> L-BFGS with restart gate,
    W_MASS = 10), clamp, and A-PINN audit as every other campaign run.
  - The expected signs recorded per case are the findings of the original
    study and serve as pre-registered predictions for the re-establishment;
    a departure under the final protocol would itself be a finding.

Called by run_all.py via the SYNTH_4Q and SYNTH_INFLATION configurations;
one orchestrator "seed" runs all sub-cases of its grid for that seed.
"""
import numpy as np

from pinn_production_coupled import run_pinn as production_run_pinn, W_MASS

# ============================================
# Ground truth and synthetic-material constants
# ============================================
D0_TRUE = 4.3e-14
ALPHA_TRUE = -0.5
C_BULK = 17.0
C_SURFACE = 1.2
RHO_ALLOY = 8.0
NOISE_SEED = 99          # fixed noise realization: the dataset, not the run
NOISE_REL = 0.03         # 3 % of the concentration range

# Configurations discovered by run_all.find_cfg (data generated on demand).
SYNTH_4Q = {"name": "SYNTH_4Q"}
SYNTH_INFLATION = {"name": "SYNTH_INFLATION"}

CASES = {
    "SYNTH_4Q": [
        {"name": "Q1_Deep_Consistent",    "L_um": 80.0, "T_max_hr": 3000.0, "n_eds": 25, "times": [1000.0, 2000.0, 3000.0], "infl": 1.0,  "exp_sign": "neg"},
        {"name": "Q2_Deep_Inflated",      "L_um": 80.0, "T_max_hr": 3000.0, "n_eds": 25, "times": [1000.0, 2000.0, 3000.0], "infl": 5.0,  "exp_sign": "neg"},
        {"name": "Q3_Shallow_Consistent", "L_um": 12.0, "T_max_hr": 1000.0, "n_eds": 12, "times": [1000.0],                 "infl": 1.0,  "exp_sign": "neg"},
        {"name": "Q4_Shallow_Inflated",   "L_um": 12.0, "T_max_hr": 1000.0, "n_eds": 12, "times": [1000.0],                 "infl": 10.0, "exp_sign": "pos"},
    ],
    "SYNTH_INFLATION": [
        {"name": "Deep_10x",    "L_um": 80.0, "T_max_hr": 3000.0, "n_eds": 25, "times": [1000.0, 2000.0, 3000.0], "infl": 10.0, "exp_sign": "neg"},
        {"name": "Deep_20x",    "L_um": 80.0, "T_max_hr": 3000.0, "n_eds": 25, "times": [1000.0, 2000.0, 3000.0], "infl": 20.0, "exp_sign": "neg"},
        {"name": "Shallow_5x",  "L_um": 12.0, "T_max_hr": 1000.0, "n_eds": 12, "times": [1000.0],                 "infl": 5.0,  "exp_sign": "pos"},
        {"name": "Shallow_20x", "L_um": 12.0, "T_max_hr": 1000.0, "n_eds": 12, "times": [1000.0],                 "infl": 20.0, "exp_sign": "pos"},
    ],
}

# ============================================
# FDM synthetic data generator (verbatim transplant from the v1 study)
# ============================================
def solve_diffusion_fdm(D0, alpha, C_bulk, C_surface, L_cm, t_max_s,
                        mass_times_s, nx=200, nt=50000):
    dx = L_cm / nx
    dt = t_max_s / nt
    C_range = C_bulk - C_surface

    D_max = D0 * (1 + abs(alpha))
    if D_max * dt / dx**2 > 0.5:
        nt = int(t_max_s * D_max / (0.4 * dx**2)) + 1
        dt = t_max_s / nt

    C = np.ones(nx + 1) * C_bulk
    C[0] = C_surface
    x_cm = np.linspace(0, L_cm, nx + 1)
    x_um = x_cm * 1e4

    mass_loss = {}
    current_time = 0.0

    for step in range(nt):
        C_norm = (C - C_surface) / C_range
        C_norm = np.clip(C_norm, 0, 1)
        D = D0 * (1 + alpha * (1 - C_norm))

        C_new = C.copy()
        for i in range(1, nx):
            D_hp = 0.5 * (D[i] + D[i + 1])
            D_hm = 0.5 * (D[i] + D[i - 1])
            C_new[i] = C[i] + dt * (D_hp * (C[i + 1] - C[i])
                                    - D_hm * (C[i] - C[i - 1])) / dx**2

        C = C_new
        C[0] = C_surface
        C[-1] = C_bulk
        current_time += dt

        for t_s in mass_times_s:
            t_hr = t_s / 3600
            if abs(current_time - t_s) < dt and t_hr not in mass_loss:
                try:
                    dep = np.trapezoid(C_bulk - C, x_cm)
                except AttributeError:
                    dep = np.trapz(C_bulk - C, x_cm)
                mass_loss[t_hr] = RHO_ALLOY * dep / 100 * 1000

    return x_um, C, mass_loss


def make_case_config(case):
    """Generate the synthetic dataset for one case and pack it as a
    standard configuration dictionary for the production solver."""
    x_um, C_profile, mass_con = solve_diffusion_fdm(
        D0_TRUE, ALPHA_TRUE, C_BULK, C_SURFACE,
        case["L_um"] * 1e-4, case["T_max_hr"] * 3600,
        [t * 3600 for t in case["times"]]
    )

    idx = np.linspace(0, len(x_um) - 1, case["n_eds"], dtype=int)
    eds_x = x_um[idx]
    eds_Cr = C_profile[idx]

    np.random.seed(NOISE_SEED)
    noise = np.random.normal(0, NOISE_REL * (C_BULK - C_SURFACE), case["n_eds"])
    eds_Cr = np.clip(eds_Cr + noise, C_SURFACE, C_BULK)

    times = sorted(mass_con)
    dW = np.array([mass_con[t] * case["infl"] for t in times])

    return {
        "name": case["name"],
        "temperature": "synthetic",
        "duration": f"{case['T_max_hr']:.0f}h",
        "C_bulk": C_BULK,
        "C_surface": C_SURFACE,
        "rho": RHO_ALLOY,
        "domain_um": case["L_um"],
        "T_max_hr": case["T_max_hr"],
        "w_mass": W_MASS,
        "depth_um": eds_x,
        "Cr_wt": eds_Cr,
        "mass_times_hr": np.array(times, dtype=float),
        "dW_data": dW,
        "outlier_fn": None,
        "D0_ref_label": "synthetic ground truth",
    }


# ============================================
# run_all.py hook: one orchestrator seed = all sub-cases of the grid
# ============================================
def run_pinn(cfg, seed):
    grid = CASES[cfg["name"]]
    sub_cases = []
    for case in grid:
        print(f"    [seed {seed}] {case['name']} (inflation {case['infl']}x)",
              flush=True)
        case_cfg = make_case_config(case)
        r = production_run_pinn(case_cfg, seed)
        sign_ok = (r["alpha"] < 0) if case["exp_sign"] == "neg" else (r["alpha"] > 0)
        entry = {
            "case": case["name"],
            "inflation": case["infl"],
            "D0_true": D0_TRUE,
            "alpha_true": ALPHA_TRUE,
            "expected_sign": case["exp_sign"],
            "sign_ok": bool(sign_ok),
        }
        entry.update({k: v for k, v in r.items() if k != "seed"})
        a = r["audit"]
        print(f"      D0={r['D0']:.2e} (true {D0_TRUE:.1e}), "
              f"alpha={r['alpha']:+.3f} (true {ALPHA_TRUE:+.1f}), "
              f"sign {'OK' if sign_ok else 'MISS'}, "
              f"audit {'ok' if a['audit_pass'] else 'FAIL'}, "
              f"conv={'Y' if r['converged'] else 'N'}", flush=True)
        sub_cases.append(entry)

    return {
        "seed": seed,
        "grid": cfg["name"],
        "n_cases": len(sub_cases),
        "n_sign_ok": int(sum(e["sign_ok"] for e in sub_cases)),
        "n_audit_pass": int(sum(e["audit"]["audit_pass"] for e in sub_cases)),
        "cases": sub_cases,
    }
