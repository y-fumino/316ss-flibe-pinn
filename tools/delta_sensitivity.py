#!/usr/bin/env python3
"""
delta_sensitivity.py — Model-free Δ sensitivity table (Supplementary Table S16)
================================================================================
Computes the mass deficit Δ = ΔW_measured − ΔW_profile directly from the raw
digitized data (trapezoidal integration; no trained model involved) under a
2×2 grid of analysis conventions per alloy:
  baseline B ∈ {nominal C_bulk, plateau mean} × integrand ∈ {clipped, unclipped}
where "clipped" replicates the preprocessing clip max(B − Cr, 0).
Domain-limited integrals mirror the audit domain of the main text.
Dependencies: numpy only.
"""
import numpy as np

ALLOYS = {
  "316SS": dict(x=np.array([
      0.000,1.170,3.029,4.301,6.259,8.017,9.683,11.346,13.106,14.669,
      16.336,18.000,19.763,21.426,23.092,24.757,26.420,27.989,29.848,
      31.316,33.175,34.840,36.607,38.072,39.739,41.403,43.166,44.829,
      46.591,49.725,51.489,53.252,54.914,56.579,58.245,59.910,61.671,
      63.337,65.000,66.567,68.525,70.093,71.759,73.325,74.990,76.752,
      78.517,79.985
    ]), c=np.array([
          1.187,5.479,6.814,7.704,8.595,12.297,11.264,13.338,15.709,
          19.262,16.602,17.789,17.496,18.682,17.206,17.505,18.248,
          16.772,17.958,18.257,19.888,19.595,16.944,18.713,16.645,
          16.652,18.134,18.729,14.864,18.143,16.815,16.670,18.744,
          19.339,17.271,17.570,18.461,17.281,19.355,18.618,18.770,
          17.589,16.707,16.705,16.119,17.601,14.942,15.832
    ]),
        rho=8.0,  Cs=1.2,  Bnom=17.0, L=80.0, plateau_from=25.0,
        dW=0.547, dW_note="last 3000 h replicate (audit convention); replicate mean 0.502"),
  "Hastelloy-N": dict(x=np.array([
        0.010, 0.996, 2.133, 3.098, 4.219, 5.179, 6.216, 7.255, 8.292,
        9.330, 10.368, 10.368, 11.405, 12.362, 13.559, 14.518, 15.477,
        16.514, 17.632, 18.587, 19.546, 20.665, 21.622, 22.660, 23.857,
        24.655, 25.772, 26.810, 27.849, 28.886, 29.843, 31.040, 31.918,
        33.037, 34.074, 35.032, 36.070, 37.028, 38.224, 39.182, 40.141,
        41.258, 44.370, 42.295, 43.253, 45.408, 46.366, 47.404, 48.442,
        49.478, 50.437, 51.633, 52.592, 53.628, 54.668, 55.625, 56.662,
        57.701, 58.817, 59.775, 60.813, 61.851, 62.808, 63.926, 64.964,
        66.001, 66.960, 67.997, 69.114
    ]), c=np.array([
        0.984, 3.866, 5.782, 6.552, 6.839, 7.072, 7.037, 7.198, 7.109,
        7.110, 7.182, 7.182, 7.075, 7.058, 6.969, 7.166, 7.274,
        7.185, 7.222, 7.007, 7.061, 7.223, 7.152, 7.170, 7.189,
        7.189, 7.154, 7.155, 7.298, 7.263, 7.174, 7.121, 7.086,
        7.229, 7.158, 7.230, 7.231, 7.267, 7.107, 7.179, 7.269,
        7.234, 7.164, 7.145, 7.217, 7.164, 7.236, 7.255, 7.273,
        7.130, 7.256, 7.167, 7.204, 7.115, 7.276, 7.259, 7.134,
        7.296, 7.153, 7.208, 7.172, 7.262, 7.173, 7.245, 7.282,
        7.210, 7.300, 7.211, 7.176
    ]),
        rho=8.86, Cs=0.82, Bnom=7.53, L=15.0, plateau_from=6.0,
        dW=1.34/1.85, dW_note="single specimen"),
  "316H":        dict(x=np.array([
        -0.108, 0.921, 1.950, 2.979, 4.009, 5.038, 6.013, 7.096, 8.072,
        9.047, 10.076, 11.051, 12.080, 13.001, 14.139, 15.168, 16.143,
        17.172, 18.147, 19.177, 20.206, 21.073, 23.131, 24.269, 22.156,
        25.298, 26.273, 27.302, 28.223, 29.307, 30.282, 31.257, 32.340,
        33.315, 34.345, 35.428, 36.349, 37.432, 38.407, 39.491, 40.412,
        41.441, 42.362, 43.391, 44.475, 45.558, 46.533, 47.562, 48.537,
        49.567
    ]), c=np.array([
        3.625, 7.739, 12.502, 15.317, 15.968, 16.293, 16.511, 16.458, 16.351,
        16.622, 16.732, 16.625, 16.517, 16.573, 16.520, 16.683, 16.738,
        16.739, 16.957, 16.687, 16.689, 16.852, 16.529, 16.314, 16.799,
        21.510, 16.533, 16.426, 16.535, 16.861, 16.916, 16.809, 16.594,
        16.595, 15.405, 12.430, 10.754, 14.380, 16.221, 16.709, 16.656,
        16.874, 16.767, 16.714, 16.823, 16.987, 16.988, 16.935, 16.936,
        17.045
    ]),
        rho=8.0,  Cs=3.3,  Bnom=16.6, L=12.0, plateau_from=5.0,
        dW=5.89/1.85, dW_note="mean of two replicates",
        outlier=lambda x, cr: not (cr > 20.0 or (34.0 < x < 38.0 and cr < 15.5))),
}

def explained_mg(x_um, cr, B, rho, L, clip):
    m = x_um <= L
    x_cm = x_um[m] * 1e-4
    d = B - cr[m]
    if clip: d = np.clip(d, 0, None)
    return rho * 1000.0 / 100.0 * np.trapezoid(d, x_cm)   # mg/cm^2

rows = []
for name, a in ALLOYS.items():
    x, c = np.unique(np.column_stack([a["x"], a["c"]]), axis=0).T
    if "outlier" in a:
        keep = [a["outlier"](xi, ci) for xi, ci in zip(x, c)]
        x, c = x[keep], c[keep]
    x = np.maximum(x, 0.0)
    Bplat = float(c[x > a["plateau_from"]].mean())
    for Bname, B in [("nominal", a["Bnom"]), ("plateau mean", Bplat)]:
        for clip in (True, False):
            e = explained_mg(x, c, B, a["rho"], a["L"], clip)
            d = a["dW"] - e
            rows.append((name, f"{Bname} ({B:.2f})", "clipped" if clip else "unclipped",
                         e, d, 100 * d / a["dW"]))

print("**Table S16. Sensitivity of the model-free deficit Δ to analysis conventions.**\n")
print("| Alloy | Baseline (wt%) | Integrand | Explained (mg/cm²) | Δ (mg/cm²) | Δ / measured |")
print("|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:+.3f} | {r[5]:+.1f} % |")
print()
for name, a in ALLOYS.items():
    print(f"Measured ΔW ({name}): {a['dW']:.3f} mg/cm² — {a['dW_note']}")
