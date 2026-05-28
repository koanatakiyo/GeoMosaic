#!/usr/bin/env python3
"""Per-event baseline vs enriched retrieval delta report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from geomosaic_hg.bpe import BPEConfig, retrieve
from geomosaic_hg.events import EVENTS, normalize_event
from geomosaic_hg.io import write_json
from geomosaic_hg.metrics import summarize_result
from geomosaic_hg.paths import REPORT_DIR, ensure_dir
from geomosaic_hg.smpi import SMPI


ACLED_COVERED_EVENTS = {"libya", "ukraine", "hongkong"}

EVENT_QUERIES = {
    "crimea": "Crimea territorial integrity referendum",
    "iraq": "Iraq invasion legitimacy sovereignty",
    "libya": "Libya NATO intervention civilian protection",
    "kosovo": "Kosovo independence recognition sovereignty",
    "scs": "South China Sea arbitration maritime claims",
    "jcpoa": "Iran nuclear agreement sanctions compliance",
    "ukraine": "Ukraine sovereignty territorial integrity",
    "hongkong": "Hong Kong national security law autonomy",
}

DELTA_FIELDS = [
    "candidate_count",
    "seed_count",
    "expanded_count",
    "objective_value",
    "viewpoint_coverage",
    "viewpoint_balance",
    "source_diversity",
    "layer_diversity",
    "modality_coverage",
    "provenance_completeness",
]


def csv_events(value: str) -> list[str]:
    events = [normalize_event(part.strip()) for part in value.split(",") if part.strip()]
    unknown = sorted(set(events) - set(EVENTS))
    if unknown:
        raise ValueError(f"unknown event(s): {', '.join(unknown)}")
    return events


def event_group(event_id: str) -> str:
    return "acled_covered" if event_id in ACLED_COVERED_EVENTS else "non_acled"


def query_for(event_id: str, query: str | None = None) -> str:
    return query or EVENT_QUERIES.get(event_id, f"{EVENTS[event_id].subject} geopolitical evidence")


def run_one(index: SMPI, event_id: str, query: str, args: argparse.Namespace) -> dict[str, Any]:
    config = BPEConfig(
        event_ids={event_id},
        cutoff=args.cutoff,
        hyperedge_budget=args.budget,
        max_match_level=args.max_match_level,
        ann_k=args.ann_k,
        expansion_depth=args.expansion_depth,
    )
    return retrieve(index, query, config)


def delta_row(event_id: str, query: str, baseline: dict[str, Any], enriched: dict[str, Any], cutoff: str | None, budget: int) -> dict[str, Any]:
    baseline_summary = summarize_result(baseline, cutoff=cutoff, k=budget)
    enriched_summary = summarize_result(enriched, cutoff=cutoff, k=budget)
    row: dict[str, Any] = {"event_id": event_id, "group": event_group(event_id), "query": query}
    for field in DELTA_FIELDS:
        a = baseline_summary.get(field)
        b = enriched_summary.get(field)
        row[f"baseline_{field}"] = a
        row[f"enriched_{field}"] = b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            row[f"delta_{field}"] = round(b - a, 6)
        else:
            row[f"delta_{field}"] = None
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bench-dir", type=Path, default=Path("data/baseline_full_bench"))
    parser.add_argument("--enriched-bench-dir", type=Path, default=Path("data/enriched_full_bench"))
    parser.add_argument("--events", default="libya,ukraine,hongkong", help="Comma-separated event ids.")
    parser.add_argument("--query", default="", help="Optional shared query for all events. Empty uses event-specific defaults.")
    parser.add_argument("--cutoff", default="2026-01-01T00:00:00Z")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--max-match-level", default="L4")
    parser.add_argument("--ann-k", type=int, default=100)
    parser.add_argument("--expansion-depth", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=REPORT_DIR / "enrichment_delta.json")
    parser.add_argument("--output-csv", type=Path, default=REPORT_DIR / "enrichment_delta.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_json.parent)
    events = csv_events(args.events)
    baseline_index = SMPI.from_dir(args.baseline_bench_dir)
    enriched_index = SMPI.from_dir(args.enriched_bench_dir)

    rows = []
    results: dict[str, Any] = {}
    for event_id in events:
        query = query_for(event_id, args.query or None)
        baseline = run_one(baseline_index, event_id, query, args)
        enriched = run_one(enriched_index, event_id, query, args)
        rows.append(delta_row(event_id, query, baseline, enriched, args.cutoff, args.budget))
        results[event_id] = {"query": query, "baseline": baseline, "enriched": enriched}

    write_json(args.output_json, {"rows": rows, "results": results})
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
