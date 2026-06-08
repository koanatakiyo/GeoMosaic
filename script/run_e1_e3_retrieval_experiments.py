#!/usr/bin/env python3
"""Run batch E1/E3 retrieval method comparisons and summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.paths import REPORT_DIR
from geomosaic_hg.retrieval_experiments import (
    parse_event_ids,
    run_retrieval_experiments,
    select_configs,
    summarize_comparison_rows,
    write_experiment_outputs,
)
from geomosaic_hg.smpi import SMPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=Path("data/enriched_full_bench"))
    parser.add_argument("--events", default="", help="Comma-separated event ids; empty means all events.")
    parser.add_argument("--configs", default="", help="Comma-separated config ids; empty means default E1/E3 grid.")
    parser.add_argument("--query", default="", help="Optional shared query. Empty uses event-specific defaults.")
    parser.add_argument("--cutoff", default="2026-01-01T00:00:00Z")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--ann-k", type=int, default=100)
    parser.add_argument("--expansion-depth", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=REPORT_DIR / "e1_e3_method_comparison.json")
    parser.add_argument("--output-csv", type=Path, default=REPORT_DIR / "e1_e3_method_comparison.csv")
    parser.add_argument("--summary-json", type=Path, default=REPORT_DIR / "e1_e3_summary_by_config.json")
    parser.add_argument("--summary-csv", type=Path, default=REPORT_DIR / "e1_e3_summary_by_config.csv")
    parser.add_argument("--list-configs", action="store_true", help="Print config ids and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = select_configs(args.configs or None)
    if args.list_configs:
        for config in configs:
            print(f"{config.config_id}\t{config.description}")
        return

    event_ids = parse_event_ids(args.events or None)
    index = SMPI.from_dir(args.bench_dir)
    rows = run_retrieval_experiments(
        index,
        event_ids=event_ids,
        configs=configs,
        query_override=args.query or None,
        cutoff=args.cutoff,
        budget=args.budget,
        ann_k=args.ann_k,
        expansion_depth=args.expansion_depth,
    )
    summary_rows = summarize_comparison_rows(rows)
    metadata = {
        "bench_dir": str(args.bench_dir),
        "events": event_ids,
        "configs": [config.config_id for config in configs],
        "cutoff": args.cutoff,
        "budget": args.budget,
        "ann_k": args.ann_k,
        "expansion_depth": args.expansion_depth,
        "rows": len(rows),
        "summary_rows": len(summary_rows),
    }
    write_experiment_outputs(
        rows=rows,
        summary_rows=summary_rows,
        output_json=args.output_json,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        summary_csv=args.summary_csv,
        metadata=metadata,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
