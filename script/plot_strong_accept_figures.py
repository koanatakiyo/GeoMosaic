#!/usr/bin/env python3
"""Generate Strong-Accept support figures and summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.strong_accept_figures import build_strong_accept_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_strong_accept_artifacts(args.reports_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
