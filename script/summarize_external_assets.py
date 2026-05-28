#!/usr/bin/env python3
"""Summarize external EvidenceAsset coverage by event and adapter."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from geomosaic_hg.io import read_jsonl, write_json


def row_extra(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra", {})
    return extra if isinstance(extra, dict) else {}


def value(row: dict[str, Any], key: str, fallback: str = "unknown") -> str:
    raw = row.get(key)
    if raw is None or raw == "":
        return fallback
    return str(raw)


def extra_value(row: dict[str, Any], key: str, fallback: str = "unknown") -> str:
    extra = row_extra(row)
    raw = extra.get(key)
    if raw is None or raw == "":
        raw = row.get(key)
    if raw is None or raw == "":
        return fallback
    return str(raw)


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def summarize_external_assets(input_path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(input_path))
    by_event: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_record_type: Counter[str] = Counter()
    by_modality: Counter[str] = Counter()
    by_source_layer: Counter[str] = Counter()
    by_collection_channel: Counter[str] = Counter()
    by_active_policy: Counter[str] = Counter()
    coverage: Counter[tuple[str, str, str]] = Counter()
    adapter_policy: Counter[tuple[str, str, str, str]] = Counter()

    for row in rows:
        event_id = value(row, "event_id")
        asset_source = value(row, "asset_source")
        record_type = extra_value(row, "record_type")
        modality = value(row, "modality")
        source_layer = value(row, "source_layer")
        collection_channel = extra_value(row, "collection_channel")
        active_policy = extra_value(row, "active_policy")

        by_event[event_id] += 1
        by_source[asset_source] += 1
        by_record_type[record_type] += 1
        by_modality[modality] += 1
        by_source_layer[source_layer] += 1
        by_collection_channel[collection_channel] += 1
        by_active_policy[active_policy] += 1
        coverage[(event_id, asset_source, record_type)] += 1
        adapter_policy[(event_id, asset_source, record_type, active_policy)] += 1

    coverage_matrix = [
        {
            "event_id": event_id,
            "asset_source": asset_source,
            "record_type": record_type,
            "count": count,
        }
        for (event_id, asset_source, record_type), count in sorted(coverage.items())
    ]
    adapter_policy_matrix = [
        {
            "event_id": event_id,
            "asset_source": asset_source,
            "record_type": record_type,
            "active_policy": active_policy,
            "count": count,
        }
        for (event_id, asset_source, record_type, active_policy), count in sorted(adapter_policy.items())
    ]

    return {
        "input": input_path.as_posix(),
        "total_assets": len(rows),
        "by_event": sorted_counter(by_event),
        "by_source": sorted_counter(by_source),
        "by_record_type": sorted_counter(by_record_type),
        "by_modality": sorted_counter(by_modality),
        "by_source_layer": sorted_counter(by_source_layer),
        "by_collection_channel": sorted_counter(by_collection_channel),
        "by_active_policy": sorted_counter(by_active_policy),
        "coverage_matrix": coverage_matrix,
        "adapter_policy_matrix": adapter_policy_matrix,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/0_external/external_assets.jsonl"))
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to write the summary to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_external_assets(args.input)
    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
