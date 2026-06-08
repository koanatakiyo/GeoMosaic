#!/usr/bin/env python3
"""Compare E4 direct-scoring aggregates with and without partial Llama rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_FULL_SCORERS = ("grok", "openai", "qwen", "doubao", "deepseek")
DEFAULT_ALL_SCORERS = DEFAULT_FULL_SCORERS + ("llama",)
DEFAULT_CONFIG_FIELDS = ("event", "source_layer", "source_vp", "scored_vp")


def csv_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "failed" or row.get("score") is None:
            continue
        key = (str(row.get("model")), str(row.get("task_id")), str(row.get("claim_id")))
        latest[key] = row
    return list(latest.values())


def config_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def aggregate_model_config_scores(
    rows: list[dict[str, Any]],
    config_fields: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, dict[str, float]]]:
    totals: dict[tuple[str, ...], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"score_sum": 0.0, "max_sum": 0.0, "claim_count": 0.0})
    )
    for row in rows:
        model = str(row.get("model", ""))
        key = config_key(row, config_fields)
        score = float(row.get("score") or 0)
        max_value = float(row.get("max") or 0)
        totals[key][model]["score_sum"] += score
        totals[key][model]["max_sum"] += max_value
        totals[key][model]["claim_count"] += 1
    out: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}
    for key, by_model in totals.items():
        out[key] = {}
        for model, stats in by_model.items():
            max_sum = stats["max_sum"]
            out[key][model] = {
                **stats,
                "score_rate": stats["score_sum"] / max_sum if max_sum else math.nan,
            }
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def scorer_group_value(
    by_model: dict[str, dict[str, float]],
    scorers: tuple[str, ...],
    metric: str,
) -> tuple[float, int, list[str]]:
    available = [model for model in scorers if model in by_model and not math.isnan(by_model[model][metric])]
    return mean([by_model[model][metric] for model in available]), len(available), available


def sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def pairwise_direction_summary(
    config_rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    metric_a: str,
    metric_b: str,
    eps: float,
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in config_rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)

    comparable = same = changed = ties = 0
    examples = []
    for _, rows in grouped.items():
        if len(rows) < 2:
            continue
        for left, right in combinations(rows, 2):
            delta_a = float(left[metric_a]) - float(right[metric_a])
            delta_b = float(left[metric_b]) - float(right[metric_b])
            sign_a = sign(delta_a, eps)
            sign_b = sign(delta_b, eps)
            if sign_a == 0 and sign_b == 0:
                ties += 1
                continue
            comparable += 1
            if sign_a == sign_b:
                same += 1
            else:
                changed += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "left": {key: left[key] for key in DEFAULT_CONFIG_FIELDS if key in left},
                            "right": {key: right[key] for key in DEFAULT_CONFIG_FIELDS if key in right},
                            "delta_5_full": round(delta_a, 6),
                            "delta_5_plus_partial_llama": round(delta_b, 6),
                        }
                    )
    return {
        "comparable_pairs": comparable,
        "same_direction_pairs": same,
        "changed_direction_pairs": changed,
        "both_tie_pairs": ties,
        "direction_consistency": same / comparable if comparable else math.nan,
        "changed_examples": examples,
    }


def missing_configs_for_scorers(
    config_summaries: list[dict[str, Any]],
    config_fields: tuple[str, ...],
    scorers: tuple[str, ...],
    availability_field: str,
) -> dict[str, list[dict[str, str]]]:
    missing: dict[str, list[dict[str, str]]] = {scorer: [] for scorer in scorers}
    for row in config_summaries:
        available = set(str(row[availability_field]).split("|")) if row[availability_field] else set()
        config = {field: str(row[field]) for field in config_fields}
        for scorer in scorers:
            if scorer not in available:
                missing[scorer].append(config)
    return {scorer: configs for scorer, configs in missing.items() if configs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/1_intermediate/direct_scoring/direct_scoring_results_six_scorers.jsonl")
    parser.add_argument("--output-json", default="data/1_intermediate/direct_scoring/direct_scoring_sensitivity_summary.json")
    parser.add_argument("--output-csv", default="data/1_intermediate/direct_scoring/direct_scoring_sensitivity_by_config.csv")
    parser.add_argument("--full-scorers", default=",".join(DEFAULT_FULL_SCORERS))
    parser.add_argument("--all-scorers", default=",".join(DEFAULT_ALL_SCORERS))
    parser.add_argument("--config-fields", default=",".join(DEFAULT_CONFIG_FIELDS))
    parser.add_argument("--metric", choices=["score_rate", "score_sum"], default="score_rate")
    parser.add_argument("--direction-eps", type=float, default=1e-9)
    args = parser.parse_args()

    full_scorers = csv_values(args.full_scorers)
    all_scorers = csv_values(args.all_scorers)
    config_fields = csv_values(args.config_fields)
    rows = read_jsonl(args.input)
    completed_rows = latest_completed_rows(rows)
    by_config = aggregate_model_config_scores(completed_rows, config_fields)

    config_summaries = []
    for key in sorted(by_config):
        by_model = by_config[key]
        value_5, count_5, available_5 = scorer_group_value(by_model, full_scorers, args.metric)
        value_all, count_all, available_all = scorer_group_value(by_model, all_scorers, args.metric)
        row = {field: key[idx] for idx, field in enumerate(config_fields)}
        row.update(
            {
                "score_5_full": value_5,
                "score_5_plus_partial_llama": value_all,
                "delta_all_minus_5": value_all - value_5 if not math.isnan(value_5) and not math.isnan(value_all) else math.nan,
                "n_scorers_5_full": count_5,
                "n_scorers_5_plus_partial_llama": count_all,
                "available_scorers_5_full": "|".join(available_5),
                "available_scorers_5_plus_partial_llama": "|".join(available_all),
            }
        )
        config_summaries.append(row)

    deltas = [abs(float(row["delta_all_minus_5"])) for row in config_summaries if not math.isnan(float(row["delta_all_minus_5"]))]
    pairwise = pairwise_direction_summary(
        config_summaries,
        group_fields=tuple(field for field in ("event", "scored_vp") if field in config_fields),
        metric_a="score_5_full",
        metric_b="score_5_plus_partial_llama",
        eps=args.direction_eps,
    )
    missing_full = missing_configs_for_scorers(
        config_summaries,
        config_fields,
        full_scorers,
        "available_scorers_5_full",
    )
    missing_all = missing_configs_for_scorers(
        config_summaries,
        config_fields,
        all_scorers,
        "available_scorers_5_plus_partial_llama",
    )
    summary = {
        "input": args.input,
        "metric": args.metric,
        "config_fields": list(config_fields),
        "full_scorers": list(full_scorers),
        "all_scorers": list(all_scorers),
        "completed_rows_after_dedupe": len(completed_rows),
        "config_count": len(config_summaries),
        "missing_configs_5_full_by_scorer": missing_full,
        "missing_configs_5_plus_partial_by_scorer": missing_all,
        "llama_missing_config_count": len(missing_all.get("llama", [])),
        "llama_missing_configs": missing_all.get("llama", []),
        "absolute_delta_all_minus_5": {
            "mean": mean(deltas),
            "max": max(deltas) if deltas else math.nan,
        },
        "pairwise_direction": pairwise,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(config_fields) + [
            "score_5_full",
            "score_5_plus_partial_llama",
            "delta_all_minus_5",
            "n_scorers_5_full",
            "n_scorers_5_plus_partial_llama",
            "available_scorers_5_full",
            "available_scorers_5_plus_partial_llama",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(config_summaries)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
