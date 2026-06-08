"""Batch runners and summaries for E1/E3 retrieval experiments."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .baselines import run_baselines
from .bpe import BPEConfig, retrieve
from .events import EVENTS, normalize_event
from .io import write_json
from .metrics import summarize_result
from .smpi import SMPI


ACLED_COVERED_EVENTS = frozenset({"hongkong", "libya", "ukraine"})

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

METHOD_ORDER = [
    "GeoMosaic-HG BPE",
    "SMPI-Expanded + MMR",
    "NaiveRAG",
    "Metadata++",
    "Metadata++ + MMR",
    "Random-SM",
]

METRIC_FIELDS = [
    "selected",
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
    "temporal_leakage_rate",
    "latency_ms",
]


@dataclass(frozen=True)
class RetrievalExperimentConfig:
    config_id: str
    description: str
    source_layers: frozenset[str] = field(default_factory=frozenset)
    modalities: frozenset[str] = field(default_factory=frozenset)
    evidence_roles: frozenset[str] = field(default_factory=frozenset)
    max_match_level: str = "L4"

    def to_bpe_config(
        self,
        *,
        event_id: str,
        cutoff: str | None,
        budget: int,
        ann_k: int,
        expansion_depth: int,
    ) -> BPEConfig:
        return BPEConfig(
            event_ids={event_id},
            source_layers=set(self.source_layers),
            modalities=set(self.modalities),
            cutoff=cutoff,
            hyperedge_budget=budget,
            max_match_level=self.max_match_level,
            evidence_roles=set(self.evidence_roles),
            ann_k=ann_k,
            expansion_depth=expansion_depth,
        )


def default_e1_e3_configs() -> list[RetrievalExperimentConfig]:
    return [
        RetrievalExperimentConfig("full", "All feasible source and modality layers."),
        RetrievalExperimentConfig("source_official", "Official-source primary layer only.", source_layers=frozenset({"official"})),
        RetrievalExperimentConfig("source_news", "News-source primary layer only.", source_layers=frozenset({"news"})),
        RetrievalExperimentConfig("modality_text", "Text-only assets.", modalities=frozenset({"text"})),
        RetrievalExperimentConfig(
            "modality_visual",
            "Redistributable and pointer visual/map assets.",
            modalities=frozenset({"image_full", "image_restricted_pointer", "map_pointer"}),
        ),
        RetrievalExperimentConfig(
            "modality_structured",
            "Structured event and structured document assets.",
            modalities=frozenset({"structured_event", "structured_document"}),
        ),
        RetrievalExperimentConfig("match_l3_or_better", "Require L3 or better source-asset match.", max_match_level="L3"),
        RetrievalExperimentConfig("match_l2_or_better", "Require L2 or better source-asset match.", max_match_level="L2"),
    ]


def format_set(values: Iterable[str]) -> str:
    return ",".join(sorted(values))


def event_group(event_id: str) -> str:
    return "acled_covered" if event_id in ACLED_COVERED_EVENTS else "non_acled"


def query_for_event(event_id: str, override: str | None = None) -> str:
    if override:
        return override
    return EVENT_QUERIES.get(event_id, f"{EVENTS[event_id].subject} geopolitical evidence")


def parse_event_ids(value: str | None) -> list[str]:
    if not value:
        return sorted(EVENTS)
    event_ids = [normalize_event(part.strip()) for part in value.split(",") if part.strip()]
    unknown = sorted(set(event_ids) - set(EVENTS))
    if unknown:
        raise ValueError(f"unknown event(s): {', '.join(unknown)}")
    return event_ids


def select_configs(config_ids: str | None = None) -> list[RetrievalExperimentConfig]:
    configs = default_e1_e3_configs()
    if not config_ids:
        return configs
    wanted = {part.strip() for part in config_ids.split(",") if part.strip()}
    by_id = {config.config_id: config for config in configs}
    unknown = sorted(wanted - set(by_id))
    if unknown:
        raise ValueError(f"unknown config(s): {', '.join(unknown)}")
    return [config for config in configs if config.config_id in wanted]


def _method_rows(
    *,
    event_id: str,
    config: RetrievalExperimentConfig,
    query: str,
    cutoff: str | None,
    budget: int,
    method_results: dict[str, dict[str, Any]],
    latencies_ms: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method not in method_results:
            continue
        summary = summarize_result(method_results[method], cutoff=cutoff, k=budget)
        rows.append(
            {
                "event_id": event_id,
                "event_group": event_group(event_id),
                "query": query,
                "config_id": config.config_id,
                "config_description": config.description,
                "source_layers": format_set(config.source_layers),
                "modalities": format_set(config.modalities),
                "evidence_roles": format_set(config.evidence_roles),
                "max_match_level": config.max_match_level,
                "method": method,
                **summary,
                "latency_ms": round(latencies_ms.get(method, 0.0), 3),
            }
        )
    return rows


def run_event_config(
    index: SMPI,
    *,
    event_id: str,
    config: RetrievalExperimentConfig,
    query: str,
    cutoff: str | None,
    budget: int,
    ann_k: int,
    expansion_depth: int,
) -> list[dict[str, Any]]:
    bpe_config = config.to_bpe_config(event_id=event_id, cutoff=cutoff, budget=budget, ann_k=ann_k, expansion_depth=expansion_depth)
    latencies_ms: dict[str, float] = {}

    start = time.perf_counter()
    bpe_result = retrieve(index, query, bpe_config)
    latencies_ms["GeoMosaic-HG BPE"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    baseline_results = run_baselines(index, query, bpe_config)
    baseline_total_ms = (time.perf_counter() - start) * 1000
    per_baseline_ms = baseline_total_ms / max(1, len(baseline_results))
    for method in baseline_results:
        latencies_ms[method] = per_baseline_ms

    method_results = {"GeoMosaic-HG BPE": bpe_result, **baseline_results}
    return _method_rows(
        event_id=event_id,
        config=config,
        query=query,
        cutoff=cutoff,
        budget=budget,
        method_results=method_results,
        latencies_ms=latencies_ms,
    )


def run_retrieval_experiments(
    index: SMPI,
    *,
    event_ids: list[str],
    configs: list[RetrievalExperimentConfig],
    query_override: str | None = None,
    cutoff: str | None = None,
    budget: int = 10,
    ann_k: int = 100,
    expansion_depth: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for event_id in event_ids:
            rows.extend(
                run_event_config(
                    index,
                    event_id=event_id,
                    config=config,
                    query=query_for_event(event_id, query_override),
                    cutoff=cutoff,
                    budget=budget,
                    ann_k=ann_k,
                    expansion_depth=expansion_depth,
                )
            )
    return rows


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def summarize_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["config_id"]), str(row["method"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (config_id, method), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "config_id": config_id,
            "method": method,
            "events_n": len({row["event_id"] for row in group}),
            "rows_n": len(group),
        }
        first = group[0]
        for field in ("config_description", "source_layers", "modalities", "evidence_roles", "max_match_level"):
            out[field] = first.get(field, "")
        for field in METRIC_FIELDS:
            vals = [float(row[field]) for row in group if isinstance(row.get(field), (int, float))]
            out[field] = _mean(vals)
        summary.append(out)
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_outputs(
    *,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_json: Path,
    output_csv: Path,
    summary_json: Path,
    summary_csv: Path,
    metadata: dict[str, Any],
) -> None:
    write_json(output_json, {"metadata": metadata, "rows": rows})
    write_rows_csv(output_csv, rows)
    write_json(summary_json, {"metadata": metadata, "rows": summary_rows})
    write_rows_csv(summary_csv, summary_rows)
