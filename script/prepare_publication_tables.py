#!/usr/bin/env python3
"""Prepare compact publication-facing CSV tables from completed experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.publication_tables import prepare_publication_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=Path("data/enriched_full_bench"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--direct-scoring-dir", type=Path, default=Path("data/1_intermediate/direct_scoring"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports/publication_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_publication_tables(
        bench_dir=args.bench_dir,
        reports_dir=args.reports_dir,
        direct_scoring_dir=args.direct_scoring_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
