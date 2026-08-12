#!/usr/bin/env python3
"""
run_all.py — Campaign orchestrator: runs the declarative batches (N1–N9) of campaigns.json.

One batch = one night. Configurations are frozen in campaigns.json;
this script only executes them. Results are written per-job in the same
JSON schema as the existing results/ files, so all downstream tools
(assemble_production_json.py, make_stables.py, dump_profiles.py) work unchanged.

Usage:
    python src/run_all.py --list                 # show batches and their jobs
    python src/run_all.py --batch N1             # run one night batch
    python src/run_all.py --batch N1 --pilot     # first 3 seeds only (preview)
    python src/run_all.py --batch N1 --dry-run   # print the plan, run nothing

Resume: a job whose output JSON already exists in results/ is skipped,
so an interrupted night can be restarted with the same command.
"""

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np

# Force UTF-8 stdout (Windows cp932 consoles crash on the unicode in job banners)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Dual-layout resolution: flat (campaigns.json beside run_all.py) or repo (src/ + root)
for _base in (HERE, ROOT):
    if os.path.exists(os.path.join(_base, "campaigns.json")):
        CAMPAIGNS = os.path.join(_base, "campaigns.json")
        RESULTS_DIR = os.path.join(_base, "results")
        break
else:
    raise FileNotFoundError("campaigns.json not found beside run_all.py or one level up")

sys.path.insert(0, HERE)


def load_campaigns():
    with open(CAMPAIGNS, encoding="utf-8") as f:
        return json.load(f)


def find_cfg(module, alloy_name):
    """Locate the alloy config dict inside a run module by its 'name' field."""
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, dict) and obj.get("name") == alloy_name:
            return obj
    raise KeyError(f"config named {alloy_name!r} not found in {module.__name__}")


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"override key {k!r} not present in base config "
                           f"(refusing to invent new fields)")
        if isinstance(cfg[k], np.ndarray):
            v = np.asarray(v, dtype=float)
        cfg[k] = v
    return cfg


def out_path(tag):
    return os.path.join(RESULTS_DIR, f"{tag}.json")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_job(job, seeds_default, pilot=False, dry=False):
    tag = job["tag"]
    dest = out_path(tag)
    if os.path.exists(dest):
        print(f"  [skip] {tag}: {os.path.basename(dest)} already exists (resume mode)")
        return dest

    seeds = job.get("seeds", seeds_default)
    if pilot:
        seeds = seeds[:3]
    alpha0 = job.get("alpha0", None)

    print(f"  [job ] {tag}: {job['module']}.run_pinn, alloy={job['alloy']}, "
          f"seeds={seeds}, overrides={job.get('overrides', {})}"
          + (f", alpha0={alpha0}" if alpha0 is not None else ""))
    if dry:
        return None

    module = importlib.import_module(job["module"])
    cfg = apply_overrides(find_cfg(module, job["alloy"]), job.get("overrides", {}))

    partial_dir = os.path.join(RESULTS_DIR, "partial")
    os.makedirs(partial_dir, exist_ok=True)

    runs = []
    t0 = time.time()
    for seed in seeds:
        pp = os.path.join(partial_dir, f"{tag}_seed{seed}.json")
        if os.path.exists(pp):
            with open(pp, encoding="utf-8") as f:
                r = json.load(f)
            runs.append(r)
            print(f"    --- seed {seed}: restored from checkpoint ---", flush=True)
            continue
        print(f"    --- seed {seed} ---", flush=True)
        if alpha0 is not None:
            r = module.run_pinn(cfg, seed, alpha0=alpha0)
        else:
            r = module.run_pinn(cfg, seed)
        with open(pp, "w", encoding="utf-8") as f:
            json.dump(r, f, default=float)
        runs.append(r)
        a = r.get("audit", {})
        print(f"    D0={r.get('D0', float('nan')):.2e}  "
              f"alpha={r.get('alpha', float('nan')):+.3f}  "
              f"audit={'ok' if a.get('audit_pass') else 'FAIL'}  "
              f"w_gap={a.get('w_gap', float('nan')):+.4f}", flush=True)

    output = {job["alloy"]: {
        "temperature": cfg.get("temperature"),
        "duration": cfg.get("duration"),
        "domain_um": cfg.get("domain_um"),
        "C_bulk": cfg.get("C_bulk"),
        "C_surface": cfg.get("C_surface"),
        "w_mass": cfg.get("w_mass"),
        "protocol": getattr(module, "PROTOCOL_STR", "n/a"),
        "float_precision": "float64",
        "run_tag": tag,
        "campaign_file": "campaigns.json",
        "overrides": job.get("overrides", {}),
        "alpha0": alpha0,
        "seeds_run": list(seeds),
        "pilot": bool(pilot),
        "elapsed_s": round(time.time() - t0, 1),
        "seeds": runs,
    }}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=float)
    with open(os.path.join(RESULTS_DIR, "SHA256SUMS"), "a", encoding="utf-8") as f:
        f.write(f"{sha256_of(dest)}  {os.path.basename(dest)}\n")
    for seed in seeds:
        pp = os.path.join(partial_dir, f"{tag}_seed{seed}.json")
        if os.path.exists(pp):
            os.remove(pp)
    print(f"  [done] {tag} -> {os.path.basename(dest)} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", help="batch name from campaigns.json (e.g. N1)")
    ap.add_argument("--list", action="store_true", help="list batches and exit")
    ap.add_argument("--pilot", action="store_true", help="first 3 seeds per job")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no runs")
    args = ap.parse_args()

    camp = load_campaigns()
    seeds_default = camp["seeds_full"]

    if args.list or not args.batch:
        print("Batches:")
        for name, b in camp["batches"].items():
            n_runs = sum(len(j.get("seeds", seeds_default)) for j in b["jobs"])
            print(f"  {name}: {len(b['jobs'])} jobs, {n_runs} runs — {b['comment']}")
        return

    batch = camp["batches"][args.batch]
    print(f"=== batch {args.batch}: {batch['comment']} ===", flush=True)
    for job in batch["jobs"]:
        run_job(job, seeds_default, pilot=args.pilot, dry=args.dry_run)
    print(f"=== batch {args.batch} complete ===", flush=True)


if __name__ == "__main__":
    main()
