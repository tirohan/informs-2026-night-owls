"""Run tests, the tree stack, the sequence model, the average, and the submission check."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--with-experiments", action="store_true")
    parser.add_argument("--with-figures", action="store_true")
    parser.add_argument("--with-report", action="store_true")
    parser.add_argument("--report-source", type=Path, default=ROOT / "report_source")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    for name in ("DM_Train.csv", "DM_Test.csv", "sample_submission.csv"):
        if not (data / name).is_file():
            parser.error(f"Required competition file not found: {data / name}")
    if args.with_report and not shutil.which("pdflatex"):
        parser.error("pdflatex is required only for --with-report.")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["NIGHT_OWLS_DATA_DIR"] = str(data)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = str(args.threads)
    py = sys.executable
    # LightGBM and PyTorch each ship libomp. Loading both in one process aborts on macOS,
    # so the tree tests and the sequence tests run as separate interpreters.
    commands = [
        ("tests_tree", [py, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-p", "test_[ap]*.py", "-v"]),
        ("tests_sequence", [py, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-p", "test_sequence_model.py", "-v"]),
        ("train", [py, "-m", "night_owls", "train", "--data-dir", str(data), "--output-dir", str(output),
                   "--ablations", "--threads", str(args.threads)]),
        ("sequence", [py, "-m", "night_owls", "sequence", "--data-dir", str(data), "--output-dir", str(output)]),
        ("combine", [py, "-m", "night_owls", "combine", "--data-dir", str(data), "--output-dir", str(output)]),
        ("diagnostics", [py, str(ROOT / "shape_diagnostics.py"), "--data-dir", str(data), "--results-dir", str(output)]),
        ("submission_check", [py, "-m", "night_owls", "validate", "--template", str(data / "sample_submission.csv"),
                              "--predictions", str(output / "predictions.csv")]),
    ]
    if args.with_experiments:
        commands.append(("experiments", [py, str(ROOT / "experiments.py"), "--data-dir", str(data),
                                         "--output-dir", str(output), "--threads", str(args.threads)]))
    figure_dir = output / "report_figures"
    if args.with_figures or args.with_report:
        commands.append(("figures", [py, str(ROOT / "make_figures.py"), "--data-dir", str(data),
                                    "--results-dir", str(output), "--output-dir", str(figure_dir)]))
    if args.with_report:
        commands.append(("report", [py, str(ROOT / "build_report.py"), "--source-dir", str(args.report_source),
                                    "--figures-dir", str(figure_dir), "--output", str(output / "report.pdf")]))
    records = []
    for label, command in commands:
        print(f"\n[{label}] Running; log: {logs / (label + '.log')}", flush=True)
        started = time.monotonic()
        with (logs / (label + ".log")).open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            handle.flush()
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        records.append({"stage": label, "returncode": completed.returncode,
                        "elapsed_seconds": round(time.monotonic() - started, 2)})
        (output / "launcher_manifest.json").write_text(json.dumps(records, indent=2))
        if completed.returncode:
            print(f"Stage {label} failed. See {logs / (label + '.log')}.", file=sys.stderr)
            return completed.returncode
        print(f"[{label}] Passed ({records[-1]['elapsed_seconds']} seconds)", flush=True)
    print(f"\nPredictions: {output / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
