"""
splice_tables.py — insert the machine-generated supplementary tables into the
manuscript at their Section S4 catalogue positions.

Usage:  python splice_tables.py manuscript_integrated_v5.md tables_supplementary.md
Writes: manuscript_integrated_v5_full.md

The tables file is the verbatim output of tools/make_stables.py (Table S9
embedded). Tables are never retyped by hand: this script distributes the
generated blocks mechanically, so the printed tables remain a frozen snapshot
of machine output.
"""
import re, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if len(sys.argv) < 3:
    sys.exit("Usage: python tools/splice_tables.py <manuscript.md> <tables_supplementary.md>\n(normally invoked by make_docx.py; no need to run directly)")
man_path, tab_path = Path(sys.argv[1]), Path(sys.argv[2])
man = man_path.read_text(encoding="utf-8")
tab = tab_path.read_text(encoding="utf-8")

# ---- split the tables file into ID-keyed blocks (trailing notes travel with their table)
parts = re.split(r"(?=^\*\*Table S\d+[abc]?\.)", tab, flags=re.M)
blocks = {}
order = []
for p in parts:
    m = re.match(r"\*\*Table (S\d+[abc]?)\.", p)
    if m:
        blocks[m.group(1)] = p.strip()
        order.append(m.group(1))
print("blocks found:", ", ".join(order))

# ---- anchor map: catalogue paragraph prefix -> table ids to insert after it
PLAN = [
    ("The production coupled campaigns are listed in Tables S1\u2013S3.", ["S1", "S2", "S3"]),
    ("are listed in Tables S4\u2013S5.", ["S4", "S5"]),
    ("is listed in Table S6.", ["S6"]),
    ("the complete per-seed listing is given in Tables S8\u2013S9.", ["S7", "S8", "S9"]),
    ("The two campaigns are tabulated in Tables S10\u2013S11.", ["S10", "S11"]),
    ("The probes are listed in Tables S12\u2013S14.", ["S12", "S13", "S14"]),
    ("The sweep is tabulated in Table S15.", ["S15"]),
    ("is given in Table S16.", ["S16"]),
]

missing = [i for _, ids in PLAN for i in ids if i not in blocks]
if missing:
    sys.exit(f"ERROR: tables file lacks: {missing} — regenerate with the latest tools/make_stables.py")

for prefix, ids in PLAN:
    i = man.find(prefix)
    if i < 0:
        sys.exit(f"ERROR: anchor not found in manuscript: {prefix[:50]}")
    j = man.find("\n", i)          # end of the anchor LINE (bullets are a
    if j < 0:                        # single-newline list; a paragraph search
        j = len(man)                 # would slide to the end of the block)
    ins = "\n\n" + "\n\n".join(blocks[k] for k in ids) + "\n"
    man = man[:j] + ins + man[j:]
    print(f"  spliced {ids} after '{prefix[:44]}...'")

out = man_path.with_name(man_path.stem + "_full.md")
out.write_text(man, encoding="utf-8")
n_tables = len(re.findall(r"^\*\*Table S", man, flags=re.M))
print(f"written: {out.name}  |  tables in document: {n_tables}  |  file lines: {man.count(chr(10))+1}")
print("document-order audit (anchors and tables as they appear):")
for line in man.splitlines():
    if line.startswith("## Section S") or line.startswith("**Table S"):
        print("   ", line[:72])
