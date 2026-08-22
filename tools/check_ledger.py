from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
EXPECTED = [
    "316SS_coupled", "316SS_L40_insensitivity", "316SS_wm20", "316SS_wm30", "316SS_wm40",
    "HN_coupled", "HN_eds_only", "HN_wm20", "HN_fullextent",
    "316H_coupled", "316SS_eds_only", "316H_eds_only", "316H_eds_only_alpha0_minus1",
    "316H_wm20", "316H_avg_anchor", "316H_fullextent", "316H_outliers_included",
    "S2_four_quadrant", "S2_inflation_sweep", "316SS_single_anchor",
    "316SS_quad_physical_basis", "316SS_quad_polynomial_basis",
    "HN_quad_eds_only", "316H_quad_eds_only", "316SS_quad_eds_only",
]
missing = [t for t in EXPECTED if not (BASE / "results" / f"{t}.json").exists()]
print("MISSING:", missing) if missing else print(f"ledger complete: {len(EXPECTED)} files")
