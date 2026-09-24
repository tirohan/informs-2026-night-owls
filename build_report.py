#!/usr/bin/env python3
"""Compile the supplied report source twice; needs pdflatex installed locally.

The source and figures are copied into a temporary build directory, so auxiliary
LaTeX files do not pollute the team workspace. No TeX packages are downloaded here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--figures-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    source, output = a.source_dir.resolve(), a.output.resolve()
    if not (source / "report.tex").is_file():
        ap.error(f"Missing source: {source / 'report.tex'}")
    engine = shutil.which("pdflatex")
    if engine is None:
        ap.error("pdflatex was not found. Prediction generation does not need LaTeX.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="night_owls_report_") as temporary:
        work = Path(temporary)
        shutil.copytree(source, work, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.aux", "*.log", "*.out", "*.synctex.gz", "report.pdf"))
        if a.figures_dir:
            shutil.copytree(a.figures_dir.resolve(), work / "figures", dirs_exist_ok=True)
        for stage in (1, 2):
            completed = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                                       cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace")
            output.with_name(output.stem + f"_build_pass{stage}.log").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"LaTeX build failed; inspect {output.stem}_build_pass{stage}.log")
        shutil.copy2(work / "report.pdf", output)
    print(f"Report built: {output}")


if __name__ == "__main__":
    main()
