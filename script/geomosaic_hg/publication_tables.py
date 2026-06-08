"""Prepare compact publication-facing tables from experiment artifacts."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json


TABLE_METRICS = [
    "viewpoint_coverage",
    "viewpoint_balance",
    "source_diversity",
    "layer_diversity",
    "modality_coverage",
    "provenance_completeness",
]

E2_LATENCY_METRICS = [
    "p50_latency_ms_mean",
    "p95_latency_ms_mean",
    "mean_latency_ms_mean",
    "candidate_count_mean",
    "expanded_count_mean",
]

E2_PRUNING_METRICS = [
    "p50_latency_ms_mean",
    "p95_latency_ms_mean",
    "candidate_reduction_rate_mean",
    "candidate_keep_rate_mean",
    "candidate_count_mean",
    "expanded_count_mean",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact_metric_row(row: dict[str, Any], extra_fields: list[str]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in extra_fields}
    for metric in TABLE_METRICS:
        out[metric] = row.get(metric, "")
    return out


def filter_e1_method_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        compact_metric_row(row, ["config_id", "method"])
        for row in rows
        if row.get("config_id") == "full"
    ]


def filter_e1_ablation_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        compact_metric_row(row, ["config_id", "config_description", "method"])
        for row in rows
        if row.get("method") == "GeoMosaic-HG BPE"
    ]


def _record_type(row: dict[str, Any]) -> str:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    return str(extra.get("record_type") or row.get("record_type") or "")


def coverage_matrix_rows(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in assets:
        counts[
            (
                str(row.get("event_id", "")),
                str(row.get("asset_source", "")),
                str(row.get("modality", "")),
                _record_type(row),
            )
        ] += 1
    return [
        {
            "event_id": event_id,
            "asset_source": asset_source,
            "modality": modality,
            "record_type": record_type,
            "count": count,
        }
        for (event_id, asset_source, modality, record_type), count in sorted(counts.items())
    ]


def bench_scale(label: str) -> int:
    if label == "tier1":
        return 1
    m = re.search(r"(\d+)x", label)
    return int(m.group(1)) if m else 9999


def sort_by_bench_scale(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (bench_scale(str(row.get("bench_label", ""))), str(row.get("method", ""))))


def compact_e2_table(rows: list[dict[str, Any]], metrics: list[str] | None = None) -> list[dict[str, Any]]:
    metrics = metrics or E2_LATENCY_METRICS
    out = []
    for row in rows:
        item = {"bench_label": row.get("bench_label", ""), "method": row.get("method", "")}
        for metric in metrics:
            item[metric] = row.get(metric, "")
        out.append(item)
    return sort_by_bench_scale(out)


def e4_pairwise_direction(e4_summary: dict[str, Any]) -> dict[str, Any]:
    raw = e4_summary.get("pairwise_direction")
    if not isinstance(raw, dict):
        raw = e4_summary.get("direction_consistency")
    if not isinstance(raw, dict):
        return {}
    comparable = raw.get("comparable_pairs") or raw.get("total_pairs") or ""
    same = raw.get("same_direction_pairs") or raw.get("consistent_pairs") or ""
    changed = raw.get("changed_direction_pairs", "")
    consistency = raw.get("direction_consistency", "")
    return {
        "same_direction_pairs": same,
        "comparable_pairs": comparable,
        "changed_direction_pairs": changed,
        "direction_consistency": consistency,
    }


def e4_main_table_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    out = []
    for row in rows:
        out.append(
            {
                "event": row.get("event", ""),
                "source_layer": row.get("source_layer", ""),
                "source_vp": row.get("source_vp", ""),
                "scored_vp": row.get("scored_vp", ""),
                "score_rate_5_scorers": row.get("score_5_full", ""),
                "score_rate_5_plus_partial_llama": row.get("score_5_plus_partial_llama", ""),
                "delta_partial_llama": row.get("delta_all_minus_5", ""),
            }
        )
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown_summary(
    path: Path,
    *,
    coverage_rows_count: int,
    e1_methods_count: int,
    e1_ablation_count: int,
    e2_scaling_count: int,
    e2_pruning_count: int,
    e4_rows_count: int,
    e4_summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pairwise = e4_pairwise_direction(e4_summary)
    lines = [
        "# GeoMosaic-HG Publication Table Inputs",
        "",
        f"- Dataset coverage rows: {coverage_rows_count}",
        f"- E1 method-comparison rows: {e1_methods_count}",
        f"- E1/E3 ablation rows: {e1_ablation_count}",
        f"- E2 scaling rows: {e2_scaling_count}",
        f"- E2 pruning rows: {e2_pruning_count}",
        f"- E4 direct-scoring rows: {e4_rows_count}",
        "",
        "## E4 Robustness Notes",
        "",
        f"- Five complete scorers: {', '.join(e4_summary.get('full_scorers', []))}",
        f"- Llama missing configurations: {e4_summary.get('llama_missing_config_count', '')}",
        f"- Mean absolute delta after adding partial Llama: {e4_summary.get('absolute_delta_all_minus_5', {}).get('mean', '')}",
        f"- Pairwise direction consistency: {pairwise.get('same_direction_pairs', '')}/{pairwise.get('comparable_pairs', '')} ({pairwise.get('direction_consistency', '')})",
        f"- Pairwise direction changes: {pairwise.get('changed_direction_pairs', '')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_publication_tables(
    *,
    bench_dir: Path,
    reports_dir: Path,
    direct_scoring_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = list(read_jsonl(bench_dir / "evidence_assets.jsonl"))
    coverage = coverage_matrix_rows(assets)
    write_csv_rows(output_dir / "table1_dataset_coverage_matrix.csv", coverage)

    e1_summary = read_csv_rows(reports_dir / "e1_e3_summary_by_config.csv")
    e1_methods = filter_e1_method_table(e1_summary)
    e1_ablation = filter_e1_ablation_table(e1_summary)
    write_csv_rows(output_dir / "table2a_method_comparison_full.csv", e1_methods)
    write_csv_rows(output_dir / "table2b_source_modality_ablation_bpe.csv", e1_ablation)

    e2_per_event = compact_e2_table(read_csv_rows(reports_dir / "e2_efficiency_summary_by_method.csv"), E2_LATENCY_METRICS)
    e2_scaling = compact_e2_table(read_csv_rows(reports_dir / "e2_scaling_summary_by_method.csv"), E2_LATENCY_METRICS)
    e2_pruning = compact_e2_table(read_csv_rows(reports_dir / "e2_scaling_official_pruning_summary_by_method.csv"), E2_PRUNING_METRICS)
    write_csv_rows(output_dir / "table3a_e2_per_event_latency.csv", e2_per_event)
    write_csv_rows(output_dir / "table3b_e2_global_scaling_latency.csv", e2_scaling)
    write_csv_rows(output_dir / "table3c_e2_official_pruning_scaling.csv", e2_pruning)

    e2_index = sort_by_bench_scale(read_csv_rows(reports_dir / "e2_scaling_index_summary.csv"))
    e2_pruning_index = sort_by_bench_scale(read_csv_rows(reports_dir / "e2_scaling_official_pruning_index_summary.csv"))
    write_csv_rows(output_dir / "table3d_e2_index_scaling.csv", e2_index)
    write_csv_rows(output_dir / "table3e_e2_official_pruning_index_scaling.csv", e2_pruning_index)

    e4_rows = e4_main_table_rows(direct_scoring_dir / "direct_scoring_sensitivity_by_config.csv")
    e4_summary = load_json(direct_scoring_dir / "direct_scoring_sensitivity_summary.json")
    write_csv_rows(output_dir / "table4_e4_direct_scoring_score_rates.csv", e4_rows)
    write_json(output_dir / "table4_e4_robustness_summary.json", e4_summary)

    summary = {
        "coverage_rows": len(coverage),
        "e1_method_rows": len(e1_methods),
        "e1_ablation_rows": len(e1_ablation),
        "e2_per_event_rows": len(e2_per_event),
        "e2_scaling_rows": len(e2_scaling),
        "e2_pruning_rows": len(e2_pruning),
        "e4_rows": len(e4_rows),
        "output_dir": output_dir.as_posix(),
    }
    write_json(output_dir / "publication_table_summary.json", summary)
    write_markdown_summary(
        output_dir / "publication_table_summary.md",
        coverage_rows_count=len(coverage),
        e1_methods_count=len(e1_methods),
        e1_ablation_count=len(e1_ablation),
        e2_scaling_count=len(e2_scaling),
        e2_pruning_count=len(e2_pruning),
        e4_rows_count=len(e4_rows),
        e4_summary=e4_summary,
    )
    return summary
