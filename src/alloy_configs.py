"""
alloy_configs.py — Single source of truth for all alloy data configurations.
==============================================================================
Every script in this repository (production, EDS-only, probes) imports its
alloy dictionaries from here. No script may define its own copy.

SCOPE RULE: this module contains DATA ONLY —
  measurements (EDS profiles, gravimetric records), physical constants,
  computational domain, outlier criteria, and provenance notes.
PROTOCOL KNOBS DO NOT LIVE HERE: mass-anchor weight (w_mass), seeds,
  initial values (alpha0), and closure choice are properties of a run
  protocol, not of an alloy, and are declared by each script or by
  campaigns.json overrides. (The historical duplication of these dicts
  across scripts, each carrying its own w_mass vintage, was the source
  of the fossil-field confusion eliminated by this refactor, 2026-07.)

Data provenance:
  316SS      — Zheng et al., J. Nucl. Mater. 482 (2016) 147-155, Fig. 3(a);
               mass: Zheng, Ph.D. Thesis, UW-Madison (2015), Fig. 75
               (three times × two replicate specimens).
  Hastelloy-N— Sankar et al., Nucl. Technol. 210 (2024), Fig. 2 (Cr K);
               ordinates labeled at.% but numerically wt.% (units note in
               manuscript). Raw digitization CSV verified point-by-point
               (2026-07). 69 rows include one exact duplicate click at
               x = 10.368 um (removed by np.unique preprocessing -> 68).
               Mass: Table II, 1.34 mg / 185 mm^2.
  316H       — Sankar et al., Nucl. Technol. 210 (2024), Fig. 16 (Cr K);
               same units note. 50 rows; outliers: Cr > 20 wt% (carbide
               spike ~25 um) and localized GB attack zone (34-38 um,
               Cr < 15.5 wt%). Mass: Table II, two specimens 7.66 and
               4.12 mg / 185 mm^2, fed as raw same-time replicate anchors
               (they constrain only their mean, 3.184 mg/cm^2).
"""
import numpy as np

ALLOY_316SS = {
    "name": "316SS",
    "temperature": "700°C",
    "duration": "3000h",
    "C_bulk": 17.0,
    "C_surface": 1.2,
    "rho": 8.0,
    "domain_um": 80.0,
    "T_max_hr": 3000.0,
    "depth_um": np.array([
        0.000, 1.170, 3.029, 4.301, 6.259, 8.017, 9.683,
        11.346, 13.106, 14.669, 16.336, 18.000, 19.763,
        21.426, 23.092, 24.757, 26.420, 27.989, 29.848,
        31.316, 33.175, 34.840, 36.607, 38.072, 39.739,
        41.403, 43.166, 44.829, 46.591, 49.725, 51.489,
        53.252, 54.914, 56.579, 58.245, 59.910, 61.671,
        63.337, 65.000, 66.567, 68.525, 70.093, 71.759,
        73.325, 74.990, 76.752, 78.517, 79.985
    ]),
    "Cr_wt": np.array([
        1.187, 5.479, 6.814, 7.704, 8.595, 12.297, 11.264,
        13.338, 15.709, 19.262, 16.602, 17.789, 17.496,
        18.682, 17.206, 17.505, 18.248, 16.772, 17.958,
        18.257, 19.888, 19.595, 16.944, 18.713, 16.645,
        16.652, 18.134, 18.729, 14.864, 18.143, 16.815,
        16.670, 18.744, 19.339, 17.271, 17.570, 18.461,
        17.281, 19.355, 18.618, 18.770, 17.589, 16.707,
        16.705, 16.119, 17.601, 14.942, 15.832
    ]),
    "mass_times_hr": np.array([1000.0, 1000.0, 2000.0, 2000.0, 3000.0, 3000.0]),
    "dW_data": np.array([0.174, 0.217, 0.308, 0.350, 0.456, 0.547]),
    "outlier_fn": None,
    "D0_ref_label": "Zheng Deff=4.2e-15 cm²/s",
}

ALLOY_HASTN = {
    "name": "Hastelloy-N",
    "temperature": "750°C",
    "duration": "1000h",
    "C_bulk": 7.53,
    "C_surface": 0.82,
    "rho": 8.86,
    "domain_um": 15.0,
    "T_max_hr": 1000.0,
    "depth_um": np.array([
        0.010, 0.996, 2.133, 3.098, 4.219, 5.179, 6.216, 7.255, 8.292,
        9.330, 10.368, 10.368, 11.405, 12.362, 13.559, 14.518, 15.477,
        16.514, 17.632, 18.587, 19.546, 20.665, 21.622, 22.660, 23.857,
        24.655, 25.772, 26.810, 27.849, 28.886, 29.843, 31.040, 31.918,
        33.037, 34.074, 35.032, 36.070, 37.028, 38.224, 39.182, 40.141,
        41.258, 44.370, 42.295, 43.253, 45.408, 46.366, 47.404, 48.442,
        49.478, 50.437, 51.633, 52.592, 53.628, 54.668, 55.625, 56.662,
        57.701, 58.817, 59.775, 60.813, 61.851, 62.808, 63.926, 64.964,
        66.001, 66.960, 67.997, 69.114
    ]),
    "Cr_wt": np.array([
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
    "mass_times_hr": np.array([1000.0]),
    "dW_data": np.array([1.34 / 1.85]),
    "outlier_fn": None,
    "D0_ref_label": "Sankar (2024)",
}

ALLOY_316H = {
    "name": "316H",
    "temperature": "750°C",
    "duration": "1000h",
    "C_bulk": 16.6,
    "C_surface": 3.3,
    "rho": 8.0,
    "domain_um": 12.0,
    "T_max_hr": 1000.0,
    "depth_um": np.array([
        -0.108, 0.921, 1.950, 2.979, 4.009, 5.038, 6.013, 7.096, 8.072,
        9.047, 10.076, 11.051, 12.080, 13.001, 14.139, 15.168, 16.143,
        17.172, 18.147, 19.177, 20.206, 21.073, 23.131, 24.269, 22.156,
        25.298, 26.273, 27.302, 28.223, 29.307, 30.282, 31.257, 32.340,
        33.315, 34.345, 35.428, 36.349, 37.432, 38.407, 39.491, 40.412,
        41.441, 42.362, 43.391, 44.475, 45.558, 46.533, 47.562, 48.537,
        49.567
    ]),
    "Cr_wt": np.array([
        3.625, 7.739, 12.502, 15.317, 15.968, 16.293, 16.511, 16.458, 16.351,
        16.622, 16.732, 16.625, 16.517, 16.573, 16.520, 16.683, 16.738,
        16.739, 16.957, 16.687, 16.689, 16.852, 16.529, 16.314, 16.799,
        21.510, 16.533, 16.426, 16.535, 16.861, 16.916, 16.809, 16.594,
        16.595, 15.405, 12.430, 10.754, 14.380, 16.221, 16.709, 16.656,
        16.874, 16.767, 16.714, 16.823, 16.987, 16.988, 16.935, 16.936,
        17.045
    ]),
    "mass_times_hr": np.array([1000.0, 1000.0]),
    "dW_data": np.array([7.66 / 1.85, 4.12 / 1.85]),
    "outlier_fn": lambda x, cr: not (cr > 20.0 or (34.0 < x < 38.0 and cr < 15.5)),
    "D0_ref_label": "Sankar (2024)",
}

ALL_ALLOYS = [ALLOY_316SS, ALLOY_HASTN, ALLOY_316H]
BY_NAME = {c["name"]: c for c in ALL_ALLOYS}


def get_config(name):
    """Return a shallow copy of the named alloy configuration."""
    return dict(BY_NAME[name])


# ==============================================================
# Self-test: data invariants. Run `python alloy_configs.py`.
# These pin the preprocessing outcomes that the published ledger
# was generated under; any transcription error breaks them.
# ==============================================================
def _preprocess(cfg, L=None):
    L = L if L is not None else cfg["domain_um"]
    x, cr = np.unique(np.column_stack([cfg["depth_um"], cfg["Cr_wt"]]), axis=0).T
    if cfg["outlier_fn"] is not None:
        keep = [cfg["outlier_fn"](a, b) for a, b in zip(x, cr)]
        x, cr = x[keep], cr[keep]
    x = np.maximum(x, 0.0)
    m = x <= L
    return x[m], cr[m]


def self_test():
    checks = []
    hn, h, ss = ALLOY_HASTN, ALLOY_316H, ALLOY_316SS

    def wt(cfg):
        L = cfg["domain_um"] * 1e-4
        cf = cfg["rho"] * ((cfg["C_bulk"] - cfg["C_surface"]) / 100) * L * 1000
        return (cfg["dW_data"] / cf)[-1]

    checks.append(("HN raw rows 69 (one duplicate click)", len(hn["depth_um"]) == 69))
    checks.append(("HN deduplicated 68", len(np.unique(np.column_stack(
        [hn["depth_um"], hn["Cr_wt"]]), axis=0)) == 68))
    checks.append(("HN in-domain anchors 15", len(_preprocess(hn)[0]) == 15))
    checks.append(("316H raw rows 50", len(h["depth_um"]) == 50))
    checks.append(("316H cleaned full-extent 45", len(_preprocess(h, 1e9)[0]) == 45))
    checks.append(("316H in-domain anchors 12", len(_preprocess(h)[0]) == 12))
    checks.append(("316SS anchors 48 (all retained)", len(_preprocess(ss)[0]) == 48))
    checks.append(("HN w_target ledger value",
                   wt(hn) == 0.8122422362144084))
    checks.append(("316H w_target ledger value (raw replicate pair)",
                   wt(h) == 1.7442254284359544))
    checks.append(("316SS w_target ledger value",
                   wt(ss) == 0.05409414556962026))
    ok = all(v for _, v in checks)
    for name, v in checks:
        print(("PASS  " if v else "FAIL  ") + name)
    print("=== alloy_configs self-test:", "ALL PASS ===" if ok else "FAILURES ===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
