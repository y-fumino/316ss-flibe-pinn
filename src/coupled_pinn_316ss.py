"""
Multi-Physics Coupled PINN for 316SS/FLiBe Corrosion
=====================================================

Inverse estimation of concentration-dependent chromium diffusion parameters
from spatial EDS concentration profile and gravimetric mass loss data.

Reference:
    Y. Fumino, "Quantifying Grain Boundary and Bulk Diffusion Contributions
    in 316 Stainless Steel / FLiBe Corrosion via Multi-Physics Physics-Informed
    Neural Networks", Journal of Nuclear Materials (submitted).

The neural network has two outputs: [u(x,t), w(x,t)]
    u: Normalized Cr concentration
    w: Auxiliary variable for integral mass conservation

Governing equations (PDEs):
    Eq1 (diffusion):  du/dt = d/dx[D(C) * du/dx]
    Eq2 (integral):   dw/dx = 1 - u

where D(C) = D0 * (1 + alpha * (1 - C_norm))

By construction, w(1,t) = integral_0^1 (1-u) dx, which is directly
proportional to the mass loss (see Yuan et al., J. Comput. Phys. 462, 2022).
"""

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================
# Experimental Data
# ============================================

# --- Spatial concentration profile ---
# Source: Zheng et al., J. Nucl. Mater. 482 (2016) 147-155, Figure 3(a)
# Conditions: 316SS in 316SS crucible, purified FLiBe, 700 C, 3000 h
# Extracted using WebPlotDigitizer (48 data points)

depth_from_surface = np.array([
    0.000, 1.170, 3.029, 4.301, 6.259, 8.017, 9.683,
    11.346, 13.106, 14.669, 16.336, 18.000, 19.763,
    21.426, 23.092, 24.757, 26.420, 27.989, 29.848,
    31.316, 33.175, 34.840, 36.607, 38.072, 39.739,
    41.403, 43.166, 44.829, 46.591, 49.725, 51.489,
    53.252, 54.914, 56.579, 58.245, 59.910, 61.671,
    63.337, 65.000, 66.567, 68.525, 70.093, 71.759,
    73.325, 74.990, 76.752, 78.517, 79.985
])  # depth from wall surface (um)

Cr_measured = np.array([
    1.187, 5.479, 6.814, 7.704, 8.595, 12.297, 11.264,
    13.338, 15.709, 19.262, 16.602, 17.789, 17.496,
    18.682, 17.206, 17.505, 18.248, 16.772, 17.958,
    18.257, 19.888, 19.595, 16.944, 18.713, 16.645,
    16.652, 18.134, 18.729, 14.864, 18.143, 16.815,
    16.670, 18.744, 19.339, 17.271, 17.570, 18.461,
    17.281, 19.355, 18.618, 18.770, 17.589, 16.707,
    16.705, 16.119, 17.601, 14.942, 15.832
])  # Cr concentration (wt%)

# --- Gravimetric mass loss data ---
# Source: Zheng, Ph.D. Thesis, Univ. of Wisconsin-Madison (2015), Figure 75
# Conditions: 316SS in 316SS crucible, purified FLiBe, 700 C
# Two replicate specimens per time point (6 data points total)

time_hr_data = np.array([1000.0, 1000.0, 2000.0, 2000.0, 3000.0, 3000.0])
dW_data = np.array([0.174, 0.217, 0.308, 0.350, 0.456, 0.547])  # mg/cm^2


# ============================================
# Physical Parameters
# ============================================

C_bulk = 17.0       # Bulk Cr concentration (wt%)
C_surface = 1.2     # Surface Cr concentration at equilibrium with FLiBe (wt%)
C_range = C_bulk - C_surface
rho_alloy = 8.0     # Alloy density (g/cm^3)
L_depth = 80.0e-4   # Computational domain depth: 80 um = 80e-4 cm
T_max_hr = 3000.0   # Maximum exposure time (h)
T_max_s = T_max_hr * 3600  # Maximum exposure time (s)


# ============================================
# Normalization
# ============================================

# Spatial profile: normalize x to [0,1] and C to [0,1]
x_norm_sp = depth_from_surface * 1e-4 / L_depth
C_norm_sp = np.clip((Cr_measured - C_surface) / C_range, 0, 1)

# Mass loss: convert delta_W (mg/cm^2) to target w(1,t) values
# delta_W = rho_alloy * (C_range/100) * L_depth * w(1,t) * 1000
conversion_factor = rho_alloy * (C_range / 100) * L_depth * 1000
w_target_data = dW_data / conversion_factor
t_norm_mass = time_hr_data * 3600 / T_max_s

print("=== Mass conservation conversion ===")
print(f"Conversion factor: {conversion_factor:.6f} mg/cm^2 per unit integral")
for i in [0, 2, 4]:
    print(f"  t={time_hr_data[i]:.0f}h: dW={dW_data[i]:.3f} mg/cm^2 -> w(1,t) = {w_target_data[i]:.4f}")


# ============================================
# Trainable Parameters
# ============================================

log10_D_est = dde.Variable(-14.0)  # log10(D0) in cm^2/s
alpha_est = dde.Variable(1.0)      # concentration-dependence parameter


# ============================================
# Coupled PDE Definition
# ============================================

def pde_coupled(x, y):
    """
    Coupled PDE system for nonlinear diffusion + integral mass conservation.

    Args:
        x: Input tensor [spatial_coordinate, time]
        y: Output tensor [u (normalized concentration), w (auxiliary variable)]

    Returns:
        List of PDE residuals [diffusion_eq, integral_eq]
    """
    u = y[:, 0:1]
    w = y[:, 1:2]

    # Derivatives of u
    du_t = dde.grad.jacobian(y, x, i=0, j=1)
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)

    # Derivative of w
    dw_x = dde.grad.jacobian(y, x, i=1, j=0)

    # Recover physical parameters
    D_base = 10.0 ** log10_D_est
    D_C = D_base * (1.0 + alpha_est * (1.0 - u))
    scale = T_max_s / (L_depth ** 2)
    D_norm = D_C * scale
    dD_dC = -D_base * alpha_est * scale

    # Eq 1: Nonlinear diffusion equation
    #   du/dt = d/dx[D(C) * du/dx] = D(C)*du_xx + dD/dC*(du/dx)^2
    eq_diffusion = du_t - (D_norm * du_xx + dD_dC * du_x * du_x)

    # Eq 2: Integral definition (dw/dx = 1 - u)
    eq_integral = dw_x - (1.0 - u)

    return [eq_diffusion, eq_integral]


# ============================================
# Computational Domain, BCs, and IC
# ============================================

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Boundary conditions for u (component 0)
bc_u_surface = dde.icbc.DirichletBC(
    geomtime, lambda _: 0,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0),
    component=0
)
bc_u_bulk = dde.icbc.DirichletBC(
    geomtime, lambda _: 1,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
    component=0
)
ic_u = dde.icbc.IC(
    geomtime, lambda _: 1.0,
    lambda _, on_initial: on_initial,
    component=0
)

# Boundary condition for w (component 1)
# Integral starts from x=0, so w(0,t) = 0
bc_w_start = dde.icbc.DirichletBC(
    geomtime, lambda _: 0,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0),
    component=1
)


# ============================================
# Observation Data (incorporated into loss)
# ============================================

# 1. Spatial profile: u(x, t=1.0) at 25 subsampled points
idx_sp = np.linspace(0, len(x_norm_sp) - 1, 25, dtype=int)
obs_points_sp = np.hstack([x_norm_sp[idx_sp].reshape(-1, 1), np.full((25, 1), 1.0)])
obs_values_sp = C_norm_sp[idx_sp].reshape(-1, 1)
observe_u = dde.icbc.PointSetBC(obs_points_sp, obs_values_sp, component=0)

# 2. Mass loss: w(x=1, t) at 3 time points x 2 replicates
obs_points_mass = np.hstack([np.full((6, 1), 1.0), t_norm_mass.reshape(-1, 1)])
obs_values_mass = w_target_data.reshape(-1, 1)
observe_w = dde.icbc.PointSetBC(obs_points_mass, obs_values_mass, component=1)


# ============================================
# Model Construction and Training
# ============================================

# Loss weights: mass data points are fewer, so assign higher weight
loss_weights = [
    1, 1,      # PDE residuals: diffusion, integral
    1, 1, 1,   # BC/IC for u: surface, bulk, initial
    1,         # BC for w: w(0,t)=0
    1.0,       # Spatial profile data (u)
    10.0       # Mass loss data (w) - weighted 10x
]

data = dde.data.TimePDE(
    geomtime, pde_coupled,
    [bc_u_surface, bc_u_bulk, ic_u, bc_w_start, observe_u, observe_w],
    num_domain=1000, num_boundary=100, num_initial=100,
    anchors=np.vstack([obs_points_sp, obs_points_mass])
)

# Neural network: 2 inputs -> 3 hidden layers (40 neurons each) -> 2 outputs
net = dde.nn.FNN([2] + [40] * 3 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# Phase 1: Adam optimizer (15,000 iterations, lr=1e-3)
print("\n=== Starting coupled PINN training ===")
model.compile(
    "adam", lr=1e-3,
    loss_weights=loss_weights,
    external_trainable_variables=[log10_D_est, alpha_est]
)
losshistory, train_state = model.train(iterations=15000, display_every=3000)

# Phase 2: L-BFGS refinement until convergence
model.compile(
    "L-BFGS",
    loss_weights=loss_weights,
    external_trainable_variables=[log10_D_est, alpha_est]
)
losshistory, train_state = model.train()


# ============================================
# Results
# ============================================

log10_D_val = log10_D_est.detach().cpu().numpy().item()
alpha_val = alpha_est.detach().cpu().numpy().item()
D_base = 10 ** log10_D_val

print(f"\n{'=' * 60}")
print(f"Coupled PINN Estimation Results")
print(f"  D_0,PINN = {D_base:.3e} cm^2/s")
print(f"  alpha    = {alpha_val:.3f}")
print(f"{'=' * 60}")


# ============================================
# Validation: Mass Loss Prediction
# ============================================

print("\n=== Mass loss prediction ===")
t_check_hr = np.array([1000, 2000, 3000])
t_check_norm = t_check_hr * 3600 / T_max_s

x_query = np.ones((3, 1))
t_query = t_check_norm.reshape(3, 1)
query_points = np.hstack([x_query, t_query])

y_pred = model.predict(query_points)
w_pred = y_pred[:, 1]
dW_predicted = w_pred * conversion_factor

dW_measured_avg = np.array([
    np.mean(dW_data[:2]), np.mean(dW_data[2:4]), np.mean(dW_data[4:])
])

print(f"{'Time (h)':>10} {'Measured (mg/cm2)':>18} {'Predicted (mg/cm2)':>18} {'Ratio':>8}")
print("-" * 58)
for i in range(3):
    ratio = dW_predicted[i] / dW_measured_avg[i] if dW_measured_avg[i] > 0 else 0
    print(f"{t_check_hr[i]:10d} {dW_measured_avg[i]:18.3f} {dW_predicted[i]:18.3f} {ratio:8.2f}")
