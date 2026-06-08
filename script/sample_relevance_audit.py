#!/usr/bin/env python3
"""Sample top-k retrieval pairs for human relevance audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.events import EVENTS, normalize_event
from geomosaic_hg.relevance_audit import (
    method_aliases,
    parse_csv_list,
    run_audit_sampling,
    write_audit_html,
    write_csv,
)
from geomosaic_hg.smpi import SMPI


def parse_event_ids(value: str | None) -> list[str]:
    if not value:
        return sorted(EVENTS)
    event_ids = [normalize_event(part) for part in parse_csv_list(value)]
    unknown = sorted(set(event_ids) - set(EVENTS))
    if unknown:
        raise ValueError(f"unknown event(s): {', '.join(unknown)}")
    return event_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=Path("data/enriched_full_bench"))
    parser.add_argument("--events", default="", help="Comma-separated event ids; default is all events.")
    parser.add_argument("--methods", default="bpe,mmr,metadata", help="Comma-separated method aliases or names.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--cutoff", default="2026-01-01T00:00:00Z")
    parser.add_argument("--ann-k", type=int, default=100)
    parser.add_argument("--expansion-depth", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports/relevance_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_ids = parse_event_ids(args.events)
    methods = method_aliases(args.methods)
    index = SMPI.from_dir(args.bench_dir)
    audit_rows, membership_rows, metadata = run_audit_sampling(
        index,
        event_ids=event_ids,
        methods=methods,
        top_k=args.top_k,
        max_pairs=args.max_pairs,
        cutoff=args.cutoff,
        budget=args.budget,
        ann_k=args.ann_k,
        expansion_depth=args.expansion_depth,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = args.output_dir / "relevance_audit_sample.csv"
    pairs_csv = args.output_dir / "relevance_audit_method_pairs.csv"
    sample_html = args.output_dir / "relevance_audit_sample.html"
    metadata_json = args.output_dir / "relevance_audit_sample_metadata.json"

    write_csv(sample_csv, audit_rows)
    write_csv(pairs_csv, membership_rows)
    write_audit_html(sample_html, audit_rows, metadata)
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                **metadata,
                "sample_csv": sample_csv.as_posix(),
                "method_pairs_csv": pairs_csv.as_posix(),
                "sample_html": sample_html.as_posix(),
                "metadata_json": metadata_json.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
