"""
make_docx.py — one-command submission build:
  splice tables -> pandoc (Times New Roman, justified) -> centering post-process

Usage:  python tools/make_docx.py manuscript_integrated_v5.md tables_supplementary.md
Writes: manuscript_integrated_v5_full.md  and  manuscript_JNM.docx

Centering rules applied to the rendered docx (the source md is never edited):
  1. Display equations — standalone paragraphs ending with an equation tag
     "(N)" / "(Na)" and containing a math signal — are centered.
  2. The author paragraph (style "Author") and the affiliation paragraph
     are centered. The corresponding-author line is left as body text.
"""
import os, re, subprocess, sys, zipfile, shutil
from pathlib import Path

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
if len(sys.argv) < 3:
    sys.exit("Usage: python tools/make_docx.py <manuscript.md> <tables_supplementary.md>")
man_path, tab_path = Path(sys.argv[1]), Path(sys.argv[2])

# ---- step 1: splice tables (reuse the audited splicer) ----
r = subprocess.run([sys.executable, str(HERE / "splice_tables.py"),
                    str(man_path), str(tab_path)], capture_output=True, text=True,
                   encoding="utf-8", env=ENV)
print(r.stdout, end="")
if r.returncode != 0:
    sys.exit(r.stderr or "splice failed")
full_md = man_path.with_name(man_path.stem + "_full.md")

# ---- step 2: pandoc with the TNR justified reference ----
if shutil.which("pandoc") is None:
    sys.exit("pandoc not found on PATH. Install it (winget install --id JohnMacFarlane.Pandoc), "
             "then reopen the terminal so PATH refreshes, and rerun this command. "
             "The spliced markdown has already been written and is preserved.")
out_docx = Path("manuscript_JNM.docx")
ref = HERE.parent / "reference_tnr.docx"
if not ref.exists():
    ref = HERE / "reference_tnr.docx"
r = subprocess.run(["pandoc", str(full_md), "-o", str(out_docx),
                    f"--reference-doc={ref}"], capture_output=True, text=True,
                   encoding="utf-8", env=ENV)
if r.returncode != 0:
    sys.exit(r.stderr or "pandoc failed")
print(f"pandoc: {out_docx} written")

# ---- step 3: centering post-process on document.xml ----
EQ_TAIL = re.compile(r"\(\d{1,2}[ab]?\)\s*$")
MATH_SIGNALS = "=\u2261\u2248\u2202\u222b\u2212\u00d7"
AFFIL = "ORCANESH Inc., Japan"

tmp = out_docx.with_suffix(".tmp")
zin = zipfile.ZipFile(out_docx)
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
n_eq = n_meta = 0
for item in zin.namelist():
    data = zin.read(item)
    if item == "word/document.xml":
        doc = data.decode("utf-8")
        paras = re.split(r"(<w:p\b.*?</w:p>)", doc, flags=re.S)
        for i, p in enumerate(paras):
            if not p.startswith("<w:p"):
                continue
            text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
            is_eq = bool(EQ_TAIL.search(text.strip())) and any(c in text for c in MATH_SIGNALS)
            is_meta = 'w:val="Author"' in p or text.strip() == AFFIL
            if not (is_eq or is_meta):
                continue
            if "<w:jc " in p:
                p2 = re.sub(r"<w:jc [^/]*/>", '<w:jc w:val="center"/>', p, count=1)
            elif "<w:pPr>" in p:
                p2 = p.replace("<w:pPr>", '<w:pPr><w:jc w:val="center"/>', 1)
            else:
                p2 = p.replace(">", '><w:pPr><w:jc w:val="center"/></w:pPr>', 1)
            paras[i] = p2
            if is_eq: n_eq += 1
            else: n_meta += 1
        data = "".join(paras).encode("utf-8")
    zout.writestr(item, data)
zin.close(); zout.close()
shutil.move(tmp, out_docx)
print(f"centered: {n_eq} equations, {n_meta} author/affiliation paragraphs")
print(f"done: {out_docx}")