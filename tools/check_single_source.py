#!/usr/bin/env python3
"""
check_single_source.py - Guard the single-source-of-truth rule for alloy data.

WHY THIS EXISTS
---------------
Every run module must take its alloy data (EDS profile, gravimetric record,
composition, domain, outlier rule) from src/alloy_configs.py. Before the
2026-07 refactor each script carried its own copy of the alloy dictionaries,
and the copies drifted: two quadratic-closure modules were still training
against a superseded digitization of the 316SS gravimetric series
(0.170/0.221/0.318/0.340 mg/cm^2 at 1000 and 2000 h, against the current
0.174/0.217/0.308/0.350 - same means, different points). The drift was
invisible in every summary statistic and surfaced only on a line-by-line
read of the modules, one command before those campaigns would have been
re-run and sealed into the ledger under a current-generation seal. A ledger
whose provenance is advertised as "one code generation" cannot afford that.

WHAT IT CHECKS
--------------
1. No run module defines alloy measurement arrays or the gravimetric record
   inline; the fields below may appear only as reads (cfg["..."]).
2. Every run module actually imports alloy_configs.
3. Protocol knobs (w_mass, alpha0, seeds) do NOT appear inside
   alloy_configs.py: they are properties of a protocol, not of an alloy, and
   live in the modules or in campaigns.json overrides. This is the same rule
   read from the other direction - it is what keeps a single alloy record
   from re-acquiring a per-script vintage.

The check is textual and deliberately so: it must fail on a module that has
not been imported or run, which is exactly the state a fossil lives in.

Exit status is 1 on any violation, so this can gate a commit or a campaign.
Usage:  python tools/check_single_source.py
"""
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "src"

# Data fields that belong to alloy_configs.py alone.
DATA_FIELDS = ["depth_um", "Cr_wt", "dW_data", "mass_times_hr",
               "C_bulk", "C_surface", "rho", "domain_um", "T_max_hr",
               "outlier_fn"]
# Protocol knobs that must NOT appear in alloy_configs.py.
PROTOCOL_KNOBS = ["w_mass", "alpha0", "seeds"]

# Modules exempt from BOTH rules below. These do not consume alloy records:
#   pinn_synthetic_s2.py  manufactures its own ground-truth fields (Section S4),
#                         so defining T_max_hr / mass_times_hr inline is correct;
#   run_all.py            is the orchestrator and holds no physics.
# An exemption is a claim that the module has no alloy to get wrong. Add one
# only for a module that generates or dispatches data, never for one that
# merely happens to fail the check.
EXEMPT = {"run_all.py", "pinn_synthetic_s2.py"}


def literal_assignments(text, field):
    """Find `"field": <literal>` - a definition, not a read."""
    pat = rf'["\']{re.escape(field)}["\']\s*:\s*(np\.array|np\.asarray|\[|\d|-|\+|"|\')'
    return [text[:m.start()].count("\n") + 1 for m in re.finditer(pat, text)]


def main():
    if not SRC.is_dir():
        sys.exit(f"check_single_source: no src/ directory at {SRC}")

    violations, exempted = [], []

    # --- 1 & 2: run modules must read, not define ---------------------------
    modules = sorted(p for p in SRC.glob("*.py") if p.name != "alloy_configs.py")
    for p in modules:
        if p.name in EXEMPT:
            exempted.append(p.name)
            continue
        text = p.read_text(encoding="utf-8")
        for field in DATA_FIELDS:
            for line in sorted(literal_assignments(text, field)):
                violations.append(f"{p.name}:{line}: defines alloy data {field!r} "
                                  f"inline - import it from alloy_configs instead")
        if not re.search(r"^\s*import alloy_configs", text, re.M):
            violations.append(f"{p.name}: does not import alloy_configs")

    # --- 3: alloy_configs must stay free of protocol knobs ------------------
    cfg_path = SRC / "alloy_configs.py"
    if cfg_path.exists():
        cfg_text = cfg_path.read_text(encoding="utf-8")
        # Ignore the module docstring, which names the knobs to explain the rule.
        body = cfg_text.split('"""', 2)[-1] if cfg_text.lstrip().startswith('"""') else cfg_text
        for knob in PROTOCOL_KNOBS:
            for line in literal_assignments(body, knob):
                off = cfg_text.index(body)
                violations.append(f"alloy_configs.py:{cfg_text[:off].count(chr(10)) + line}: "
                                  f"carries protocol knob {knob!r} - it belongs to the run "
                                  f"module or to campaigns.json")
    else:
        violations.append("src/alloy_configs.py is missing")

    if violations:
        print("single source of truth: VIOLATED")
        for v in violations:
            print("  " + v)
        return 1
    checked = len(modules) - len(exempted)
    note = f"; exempt: {', '.join(sorted(exempted))}" if exempted else ""
    print(f"single source of truth: OK ({checked} run modules checked{note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())