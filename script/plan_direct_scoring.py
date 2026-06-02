#!/usr/bin/env python3
"""Plan E4 GeoGround-style direct scoring tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.direct_scoring import build_direct_scoring_tasks  # noqa: E402


def csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", default="data/0_external/official_doc_parsed")
    parser.add_argument("--manifest", default="data/0_external/official_doc_materialized/manifest.jsonl")
    parser.add_argument("--claims", default="data/enriched_full_bench/claim_evidence_hyperedges.jsonl")
    parser.add_argument("--output-dir", default="data/1_intermediate/direct_scoring")
    parser.add_argument("--source-layers", default="official")
    parser.add_argument(
        "--score-dir",
        action="append",
        dest="score_dirs",
        default=["data/3_direct_scores/combined_official"],
        help="Directory containing existing *_claim_audit.jsonl files for claim max values. Repeatable.",
    )
    parser.add_argument("--max-bundle-chars", type=int, default=40000)
    parser.add_argument("--events", default=None, help="Comma-separated event ids to plan; default all.")
    args = parser.parse_args()

    summary = build_direct_scoring_tasks(
        parsed_dir=args.parsed_dir,
        manifest_path=args.manifest,
        claims_path=args.claims,
        output_dir=args.output_dir,
        source_layers=csv_arg(args.source_layers),
        score_dirs=args.score_dirs,
        max_bundle_chars=args.max_bundle_chars,
        events=csv_arg(args.events),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
