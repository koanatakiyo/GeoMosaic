#!/usr/bin/env python3
"""Plan Stage C claim-grounding tasks from fixed claims and parsed official passages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geomosaic_hg.claim_grounding import plan_claim_grounding_tasks


def csv_arg(value: str | None) -> set[str] | None:
    if not value:
        return None
    out = {item.strip() for item in value.split(",") if item.strip()}
    return out or None


def parse_source_layers_arg(value: str | None) -> set[str] | None:
    if value is None:
        return {"official"}
    if not value.strip():
        print(
            "warning: empty --source-layers would disable filtering; preserving default 'official'. "
            "Use --source-layers all to include every claim source layer.",
            file=sys.stderr,
        )
        return {"official"}
    if value.strip().lower() == "all":
        return None
    return csv_arg(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("data/enriched_full_bench/claim_evidence_hyperedges.jsonl"),
        help="JSONL table containing fixed GeoGround claim_id/claim_text rows.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("data/0_external/official_doc_parsed"),
        help="Directory containing official_doc_text.jsonl and passages.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding"),
        help="Directory for claim_grounding_tasks.jsonl and summary.",
    )
    parser.add_argument("--events", default="", help="Comma-separated event IDs. Default: all events.")
    parser.add_argument("--source-layers", default="official", help="Comma-separated claim source layers. Default: official.")
    parser.add_argument("--languages", default="", help="Comma-separated passage languages. Default: all parsed languages.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Plan only the first N tasks for a pilot run.")
    parser.add_argument("--audit-sample-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = plan_claim_grounding_tasks(
        claims_path=args.claims,
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir,
        events=csv_arg(args.events),
        source_layers=parse_source_layers_arg(args.source_layers),
        languages=csv_arg(args.languages),
        top_k=args.top_k,
        limit=args.limit,
        audit_sample_rate=args.audit_sample_rate,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
