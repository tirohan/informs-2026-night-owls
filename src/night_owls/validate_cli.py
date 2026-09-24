"""python -m night_owls validate --template ... --predictions ..."""
from __future__ import annotations

import argparse
import json

from night_owls.submission import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.template, args.predictions), indent=2))


if __name__ == "__main__":
    main()
