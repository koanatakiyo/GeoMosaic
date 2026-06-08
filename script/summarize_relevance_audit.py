#!/usr/bin/env python3
"""Summarize human relevance audit labels into P@K, nDCG@K, and mean relevance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.relevance_audit import (
    aggregate_by_method,
    read_csv_dict,
    summarize_relevance,
    write_csv,
    write_summary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, default=Path("data/reports/relevance_audit/relevance_audit_sample.csv"))
    parser.add_argument("--method-pairs-csv", type=Path, default=Path("data/reports/relevance_audit/relevance_audit_method_pairs.csv"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--relevance-threshold", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports/relevance_audit"))
    return parser.parse_args()


def json_safe_rows(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items()} for row in rows]


def main() -> None:
    args = parse_args()
    labels = {row["audit_id"]: row for row in read_csv_dict(args.sample_csv)}
    memberships = read_csv_dict(args.method_pairs_csv)
    rows = summarize_relevance(labels, memberships, k=args.k, relevance_threshold=args.relevance_threshold)
    aggregate = aggregate_by_method(rows)
    all_rows = [*aggregate, *rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "relevance_audit_summary.csv"
    output_json = args.output_dir / "relevance_audit_summary.json"
    output_md = args.output_dir / "relevance_audit_summary.md"
    write_csv(output_csv, all_rows)
    output_json.write_text(
        json.dumps(
            {
                "k": args.k,
                "relevance_threshold": args.relevance_threshold,
                "aggregate_by_method": json_safe_rows(aggregate),
                "per_event_method": json_safe_rows(rows),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_summary_markdown(output_md, rows, aggregate)
    print(
        json.dumps(
            {
                "k": args.k,
                "relevance_threshold": args.relevance_threshold,
                "aggregate_rows": len(aggregate),
                "per_event_rows": len(rows),
                "output_csv": output_csv.as_posix(),
                "output_json": output_json.as_posix(),
                "output_md": output_md.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
