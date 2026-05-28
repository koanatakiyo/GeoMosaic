#!/usr/bin/env python3
"""Compare BPE against metadata and diversity baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from geomosaic_hg.baselines import run_baselines, summarize_methods
from geomosaic_hg.bpe import BPEConfig, retrieve
from geomosaic_hg.events import EVENTS, normalize_event
from geomosaic_hg.io import write_json
from geomosaic_hg.metrics import summarize_result
from geomosaic_hg.paths import BENCH_DIR, REPORT_DIR, ensure_dir
from geomosaic_hg.smpi import SMPI


def csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {v.strip() for v in value.split(",") if v.strip()}


def csv_events(value: str | None) -> set[str]:
    events = {normalize_event(v.strip()) for v in (value or "").split(",") if v.strip()}
    unknown = sorted(events - set(EVENTS))
    if unknown:
        raise ValueError(f"unknown event(s): {', '.join(unknown)}")
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--events", default="", help="Comma-separated event ids; empty means all.")
    parser.add_argument("--query", default="Ukraine sovereignty territorial integrity")
    parser.add_argument("--source-layers", default="")
    parser.add_argument("--modalities", default="")
    parser.add_argument("--roles", default="")
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--max-match-level", default="L4")
    parser.add_argument("--ann-k", type=int, default=100)
    parser.add_argument("--expansion-depth", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=REPORT_DIR / "method_comparison.json")
    parser.add_argument("--output-csv", type=Path, default=REPORT_DIR / "method_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(REPORT_DIR)
    index = SMPI.from_dir(args.bench_dir)
    config = BPEConfig(
        event_ids=csv_events(args.events),
        source_layers=csv_set(args.source_layers),
        modalities=csv_set(args.modalities),
        cutoff=args.cutoff,
        hyperedge_budget=args.budget,
        max_match_level=args.max_match_level,
        evidence_roles=csv_set(args.roles),
        ann_k=args.ann_k,
        expansion_depth=args.expansion_depth,
    )
    bpe_result = retrieve(index, args.query, config)
    results = {"GeoMosaic-HG BPE": bpe_result, **run_baselines(index, args.query, config)}
    rows = [{"method": "GeoMosaic-HG BPE", **summarize_result(bpe_result, cutoff=args.cutoff, k=args.budget)}]
    rows.extend(summarize_methods({k: v for k, v in results.items() if k != "GeoMosaic-HG BPE"}, cutoff=args.cutoff, k=args.budget))

    write_json(args.output_json, {"query": args.query, "config": config.__dict__, "rows": rows, "results": results})
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
