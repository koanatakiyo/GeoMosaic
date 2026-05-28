#!/usr/bin/env python3
"""Build one-event smoke JSONL tables and validate them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.build import build_all, load_tables
from geomosaic_hg.events import EVENTS, normalize_event
from geomosaic_hg.paths import DATA_DIR, RAW_DIR, SCORE_DIR
from geomosaic_hg.validation import validate_core_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="crimea")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "smoke_bench")
    args = parser.parse_args()
    args.event = normalize_event(args.event)
    if args.event not in EVENTS:
        choices = ", ".join(sorted(EVENTS))
        parser.error(f"unknown event {args.event!r}; choose from: {choices}")
    return args


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / args.event
    summary = build_all(
        RAW_DIR,
        out_dir,
        [SCORE_DIR / "combined_official", SCORE_DIR / "combined_zh_news"],
        {args.event},
    )
    tables = load_tables(out_dir)
    errors = validate_core_tables(tables)
    payload = {"summary": summary, "errors": errors, "ok": not errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
