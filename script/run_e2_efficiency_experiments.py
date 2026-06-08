#!/usr/bin/env python3
"""Run E2 independent timing, pruning, and scaling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.efficiency_experiments import (
    METHOD_RUNNERS,
    available_canonical_events,
    csv_set,
    index_summary_row,
    parse_bench_dirs,
    run_efficiency_for_index,
    summarize_efficiency_rows,
    time_index_load,
    write_e2_outputs,
)
from geomosaic_hg.events import EVENTS, normalize_event
from geomosaic_hg.paths import REPORT_DIR
from geomosaic_hg.retrieval_experiments import METHOD_ORDER


def parse_events(value: str | None) -> list[str] | None:
    if not value:
        return None
    event_ids = [normalize_event(part.strip()) for part in value.split(",") if part.strip()]
    unknown = sorted(set(event_ids) - set(EVENTS))
    if unknown:
        raise ValueError(f"unknown event(s): {', '.join(unknown)}")
    return event_ids


def parse_methods(value: str | None) -> list[str]:
    if not value:
        return METHOD_ORDER
    aliases = {name.lower(): name for name in METHOD_RUNNERS}
    aliases.update(
        {
            "bpe": "GeoMosaic-HG BPE",
            "expanded_mmr": "SMPI-Expanded + MMR",
            "smpi_expanded_mmr": "SMPI-Expanded + MMR",
            "naive": "NaiveRAG",
            "metadata": "Metadata++",
            "mmr": "Metadata++ + MMR",
            "random": "Random-SM",
        }
    )
    methods = []
    for part in value.split(","):
        raw = part.strip()
        if not raw:
            continue
        method = aliases.get(raw.lower(), raw)
        if method not in METHOD_RUNNERS:
            raise ValueError(f"unknown method: {raw}")
        methods.append(method)
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dirs", default="tier1=data/enriched_full_bench", help="Comma-separated paths or label=path specs.")
    parser.add_argument("--scope", choices=["per_event", "global"], default="per_event")
    parser.add_argument("--events", default="", help="Comma-separated event ids for per_event scope. Empty means canonical events available in each bench.")
    parser.add_argument("--methods", default="", help="Comma-separated methods. Empty means all E1/E3 methods.")
    parser.add_argument("--query", default="", help="Shared query. Empty means event-specific defaults or a global scaling query.")
    parser.add_argument("--cutoff", default="2026-01-01T00:00:00Z")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--ann-k", type=int, default=100)
    parser.add_argument("--expansion-depth", type=int, default=1)
    parser.add_argument("--source-layers", default="", help="Comma-separated source-layer filter for constrained pruning runs.")
    parser.add_argument("--modalities", default="", help="Comma-separated modality filter for constrained pruning runs.")
    parser.add_argument("--roles", default="", help="Comma-separated evidence-role filter for constrained pruning runs.")
    parser.add_argument("--max-match-level", default="L4")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=REPORT_DIR / "e2_efficiency_results.json")
    parser.add_argument("--output-csv", type=Path, default=REPORT_DIR / "e2_efficiency_results.csv")
    parser.add_argument("--summary-json", type=Path, default=REPORT_DIR / "e2_efficiency_summary_by_method.json")
    parser.add_argument("--summary-csv", type=Path, default=REPORT_DIR / "e2_efficiency_summary_by_method.csv")
    parser.add_argument("--index-json", type=Path, default=REPORT_DIR / "e2_index_summary.json")
    parser.add_argument("--index-csv", type=Path, default=REPORT_DIR / "e2_index_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_bench_dirs(args.bench_dirs)
    requested_events = parse_events(args.events or None)
    methods = parse_methods(args.methods or None)
    source_layers = csv_set(args.source_layers)
    modalities = csv_set(args.modalities)
    evidence_roles = csv_set(args.roles)

    rows = []
    index_rows = []
    for spec in specs:
        index, build_ms = time_index_load(spec.path)
        index_rows.append(index_summary_row(spec, index, build_ms))
        event_ids = [] if args.scope == "global" else available_canonical_events(index, requested_events)
        rows.extend(
            run_efficiency_for_index(
                index,
                spec=spec,
                scope=args.scope,
                event_ids=event_ids,
                query=args.query or None,
                cutoff=args.cutoff,
                budget=args.budget,
                ann_k=args.ann_k,
                expansion_depth=args.expansion_depth,
                source_layers=source_layers,
                modalities=modalities,
                evidence_roles=evidence_roles,
                max_match_level=args.max_match_level,
                warmup=args.warmup,
                repeats=args.repeats,
                methods=methods,
            )
        )

    summary_rows = summarize_efficiency_rows(rows)
    metadata = {
        "bench_dirs": [{"label": spec.label, "path": str(spec.path)} for spec in specs],
        "scope": args.scope,
        "events": requested_events or "available_canonical_events",
        "methods": methods,
        "cutoff": args.cutoff,
        "budget": args.budget,
        "ann_k": args.ann_k,
        "expansion_depth": args.expansion_depth,
        "source_layers": sorted(source_layers),
        "modalities": sorted(modalities),
        "evidence_roles": sorted(evidence_roles),
        "max_match_level": args.max_match_level,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rows": len(rows),
        "summary_rows": len(summary_rows),
        "index_rows": len(index_rows),
    }
    write_e2_outputs(
        rows=rows,
        summary_rows=summary_rows,
        index_rows=index_rows,
        output_json=args.output_json,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        summary_csv=args.summary_csv,
        index_json=args.index_json,
        index_csv=args.index_csv,
        metadata=metadata,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
