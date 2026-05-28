#!/usr/bin/env python3
"""Create block-replicated synthetic stress tiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.paths import BENCH_DIR
from geomosaic_hg.synthetic import replicate_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--blocks", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = replicate_tables(args.input_dir, args.output_dir, args.blocks)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
