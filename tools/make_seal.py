#!/usr/bin/env python3
"""
make_seal.py - Regenerate results/SHA256SUMS from the canonical ledger list.
The official ledger list is read from tools/check_ledger.py (single source);
this tool verifies that every expected ledger exists on disk, refuses to seal
if any is missing or if unexpected ledgers are present, then hashes each file
and writes results/SHA256SUMS in sorted sha256sum format.
Usage:  python tools/make_seal.py        (no arguments)
"""
import hashlib, re, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"

src = (Path(__file__).parent / "check_ledger.py").read_text(encoding="utf-8")
m = re.search(r"EXPECTED = \[(.*?)\]", src, re.S)
if not m:
    sys.exit("make_seal: could not read EXPECTED from check_ledger.py")
expected = sorted(re.findall(r'"(\w+)"', m.group(1)))

missing = [t for t in expected if not (RES / f"{t}.json").exists()]
if missing:
    sys.exit(f"make_seal: REFUSING to seal - missing ledgers: {missing}")
on_disk = sorted(p.stem for p in RES.glob("*.json"))
extra = [t for t in on_disk if t not in expected]
if extra:
    sys.exit(f"make_seal: REFUSING to seal - unexpected ledgers on disk (add to EXPECTED or remove): {extra}")

lines = []
for t in expected:
    p = RES / f"{t}.json"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    lines.append(f"{h}  {p.name}")
(RES / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
print(f"SHA256SUMS written: {len(lines)} ledgers sealed (canonical list from check_ledger.py)")
