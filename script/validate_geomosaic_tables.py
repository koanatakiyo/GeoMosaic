#!/usr/bin/env python3
"""Validate core JSONL tables and check path locality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.build import load_tables
from geomosaic_hg.paths import BENCH_DIR
from geomosaic_hg.validation import validate_core_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_tables(args.bench_dir)
    errors = validate_core_tables(tables)
    payload = {
        "bench_dir": args.bench_dir.as_posix(),
        "counts": {name: len(rows) for name, rows in tables.items()},
        "errors": errors,
        "ok": not errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
