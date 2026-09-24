"""Command line for the Night Owls package.

    python -m night_owls train --data-dir data --output-dir results
    python -m night_owls sequence --data-dir data --output-dir results
    python -m night_owls combine --data-dir data --output-dir results
    python -m night_owls offline --data-dir data --output-dir results
"""
from __future__ import annotations

import sys

USAGE = """\
Night Owls — INFORMS 2026 DMS challenge

  night-owls train     Fit the three-member tree stack and write predictions.csv
  night-owls sequence  Fit the selective state-space model
  night-owls combine   Average the stack and the sequence model into predictions.csv
  night-owls offline   Tests, train, sequence, combine, diagnostics, validation
  night-owls validate  Check a prediction file against the template

Each command accepts the same flags as the corresponding module (--data-dir, --output-dir).
"""


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return
    command, rest = argv[0], argv[1:]
    sys.argv = [f"night_owls {command}", *rest]
    if command == "train":
        from night_owls.pipeline import main as run
    elif command == "sequence":
        from night_owls.sequence import main as run
    elif command == "combine":
        from night_owls.blend import main as run
    elif command == "offline":
        from night_owls.offline import main as run
    elif command == "validate":
        from night_owls.validate_cli import main as run
    else:
        raise SystemExit(f"Unknown command {command!r}.\n{USAGE}")
    code = run()
    if isinstance(code, int) and code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
