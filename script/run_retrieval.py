#!/usr/bin/env python3
"""Run GeoMosaic-HG BPE retrieval from project-local JSONL tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.bpe import BPEConfig, retrieve
from geomosaic_hg.events import EVENTS, normalize_event
from geomosaic_hg.io import write_json
from geomosaic_hg.metrics import summarize_result
from geomosaic_hg.paths import BENCH_DIR
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
    parser.add_argument("--query", default="Crimea territorial integrity referendum")
    parser.add_argument("--source-layers", default="", help="Comma-separated source layers; empty means all.")
    parser.add_argument("--modalities", default="", help="Comma-separated modalities; empty means all.")
    parser.add_argument("--roles", default="", help="Comma-separated evidence roles.")
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--max-match-level", default="L4")
    parser.add_argument("--ann-k", type=int, default=100)
    parser.add_argument("--expansion-depth", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    result = retrieve(index, args.query, config)
    payload = {
        "summary": summarize_result(result, cutoff=args.cutoff, k=args.budget),
        "index_summary": index.index_summary(),
        "result": result,
    }
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print("\nselected_hyperedges:")
    for h in result["selected_hyperedges"]:
        print(f"- {h['hyperedge_id']} {h['event_id']} {h.get('_rel', h.get('confidence')):.4f} {h['claim_id']}")


if __name__ == "__main__":
    main()
