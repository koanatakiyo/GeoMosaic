#!/usr/bin/env python3
"""Build project-local GeoMosaic-Bench JSONL tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.build import build_all
from geomosaic_hg.events import EVENTS, normalize_event
from geomosaic_hg.paths import BENCH_DIR, DATA_DIR, RAW_DIR, SCORE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--score-dir", type=Path, action="append", default=None, help="Claim-audit directory. Can be passed more than once.")
    parser.add_argument("--external-dir", type=Path, default=None, help="Directory of external EvidenceAsset JSONL files.")
    parser.add_argument("--no-external-assets", action="store_true", help="Build without data/0_external assets.")
    parser.add_argument("--event", action="append", default=None, help="Optional event id filter; can be passed more than once.")
    args = parser.parse_args()
    if args.no_external_assets and args.external_dir:
        parser.error("--no-external-assets cannot be combined with --external-dir")
    if args.event:
        args.event = [normalize_event(event) for event in args.event]
        unknown = sorted(set(args.event) - set(EVENTS))
        if unknown:
            choices = ", ".join(sorted(EVENTS))
            parser.error(f"unknown event(s) {', '.join(unknown)}; choose from: {choices}")
    return args


def main() -> None:
    args = parse_args()
    score_dirs = args.score_dir or [SCORE_DIR / "combined_official", SCORE_DIR / "combined_zh_news"]
    events = set(args.event) if args.event else None
    external_dir = DATA_DIR / "__no_external_assets__" if args.no_external_assets else args.external_dir
    summary = build_all(args.raw_dir, args.output_dir, score_dirs, events, external_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
