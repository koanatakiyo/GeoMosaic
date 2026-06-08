"""Independent timing and scaling helpers for E2 efficiency experiments."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .baselines import metadata_filter, metadata_mmr, naive_rag, random_sm, smpi_expanded_mmr
from .bpe import BPEConfig, retrieve
from .events import EVENTS
from .io import write_json
from .retrieval_experiments import METHOD_ORDER, query_for_event
from .smpi import SMPI


@dataclass(frozen=True)
class BenchSpec:
    label: str
    path: Path


METHOD_RUNNERS: dict[str, Callable[[SMPI, str, BPEConfig], dict[str, Any]]] = {
    "GeoMosaic-HG BPE": retrieve,
    "SMPI-Expanded + MMR": smpi_expanded_mmr,
    "NaiveRAG": naive_rag,
    "Metadata++": metadata_filter,
    "Metadata++ + MMR": metadata_mmr,
    "Random-SM": random_sm,
}

TIMING_FIELDS = [
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "candidate_reduction_rate",
    "candidate_keep_rate",
    "expanded_to_seed_ratio",
]


def parse_bench_dirs(value: str) -> list[BenchSpec]:
    specs: list[BenchSpec] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            label, path = item.split("=", 1)
            specs.append(BenchSpec(label=label.strip(), path=Path(path.strip())))
        else:
            path = Path(item)
            specs.append(BenchSpec(label=path.name, path=path))
    return specs


def csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def candidate_reduction_rate(candidate_count: int, scope_hyperedges: int) -> float:
    if scope_hyperedges <= 0:
        return 0.0
    return round(max(0.0, 1.0 - (candidate_count / scope_hyperedges)), 6)


def event_scope_size(index: SMPI, event_id: str) -> int:
    if event_id == "all":
        return len(index.hyperedges)
    return len(index.hyperedges_by_event.get(event_id, set()))


def table_file_bytes(bench_dir: Path) -> int:
    total = 0
    for name in ("source_records.jsonl", "evidence_assets.jsonl", "source_asset_links.jsonl", "claim_evidence_hyperedges.jsonl"):
        path = bench_dir / name
        if path.exists():
            total += path.stat().st_size
    return total


def time_index_load(bench_dir: Path) -> tuple[SMPI, float]:
    start = time.perf_counter()
    index = SMPI.from_dir(bench_dir)
    return index, (time.perf_counter() - start) * 1000


def index_summary_row(spec: BenchSpec, index: SMPI, build_ms: float) -> dict[str, Any]:
    summary = index.index_summary()
    return {
        "bench_label": spec.label,
        "bench_dir": str(spec.path),
        "index_build_ms": round(build_ms, 3),
        "table_file_bytes": table_file_bytes(spec.path),
        "source_records": summary["source_records"],
        "evidence_assets": summary["evidence_assets"],
        "source_asset_links": summary["source_asset_links"],
        "claim_evidence_hyperedges": summary["claim_evidence_hyperedges"],
    }


def time_method(
    index: SMPI,
    *,
    method: str,
    query: str,
    config: BPEConfig,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    runner = METHOD_RUNNERS[method]
    last_result: dict[str, Any] | None = None
    for _ in range(max(0, warmup)):
        last_result = runner(index, query, config)
    timings: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        last_result = runner(index, query, config)
        timings.append((time.perf_counter() - start) * 1000)
    if last_result is None:
        last_result = runner(index, query, config)
    return last_result, timings


def timing_row(
    *,
    bench_label: str,
    bench_dir: Path,
    scope: str,
    event_id: str,
    query: str,
    method: str,
    result: dict[str, Any],
    timings_ms: list[float],
    scope_hyperedges: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    candidate_count = int(result.get("candidate_count") or 0)
    seed_count = int(result.get("seed_count") or 0)
    expanded_count = int(result.get("expanded_count") or 0)
    keep_rate = round(candidate_count / scope_hyperedges, 6) if scope_hyperedges > 0 else 0.0
    expanded_ratio = round(expanded_count / seed_count, 6) if seed_count > 0 else 0.0
    return {
        "bench_label": bench_label,
        "bench_dir": str(bench_dir),
        "scope": scope,
        "event_id": event_id,
        "query": query,
        "method": method,
        "warmup": warmup,
        "repeats": repeats,
        "scope_hyperedges": scope_hyperedges,
        "candidate_count": candidate_count,
        "seed_count": seed_count,
        "expanded_count": expanded_count,
        "selected_count": len(result.get("selected_hyperedges", [])),
        "candidate_keep_rate": keep_rate,
        "candidate_reduction_rate": candidate_reduction_rate(candidate_count, scope_hyperedges),
        "expanded_to_seed_ratio": expanded_ratio,
        "p50_latency_ms": round(percentile(timings_ms, 50), 3),
        "p95_latency_ms": round(percentile(timings_ms, 95), 3),
        "mean_latency_ms": round(sum(timings_ms) / len(timings_ms), 3) if timings_ms else 0.0,
        "min_latency_ms": round(min(timings_ms), 3) if timings_ms else 0.0,
        "max_latency_ms": round(max(timings_ms), 3) if timings_ms else 0.0,
    }


def available_canonical_events(index: SMPI, requested: list[str] | None = None) -> list[str]:
    if requested:
        return requested
    return [event_id for event_id in sorted(EVENTS) if event_id in index.hyperedges_by_event]


def run_efficiency_for_index(
    index: SMPI,
    *,
    spec: BenchSpec,
    scope: str,
    event_ids: list[str],
    query: str | None,
    cutoff: str | None,
    budget: int,
    ann_k: int,
    expansion_depth: int,
    source_layers: set[str] | None,
    modalities: set[str] | None,
    evidence_roles: set[str] | None,
    max_match_level: str,
    warmup: int,
    repeats: int,
    methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    methods = methods or METHOD_ORDER
    rows: list[dict[str, Any]] = []
    if scope == "global":
        work_items = [("all", query or "geopolitical evidence sovereignty conflict diplomacy")]
    else:
        work_items = [(event_id, query_for_event(event_id, query)) for event_id in event_ids]

    for event_id, event_query in work_items:
        config = BPEConfig(
            event_ids=set() if event_id == "all" else {event_id},
            source_layers=set(source_layers or set()),
            modalities=set(modalities or set()),
            cutoff=cutoff,
            hyperedge_budget=budget,
            max_match_level=max_match_level,
            evidence_roles=set(evidence_roles or set()),
            ann_k=ann_k,
            expansion_depth=expansion_depth,
        )
        scope_hyperedges = event_scope_size(index, event_id)
        for method in methods:
            result, timings = time_method(index, method=method, query=event_query, config=config, warmup=warmup, repeats=repeats)
            rows.append(
                timing_row(
                    bench_label=spec.label,
                    bench_dir=spec.path,
                    scope=scope,
                    event_id=event_id,
                    query=event_query,
                    method=method,
                    result=result,
                    timings_ms=timings,
                    scope_hyperedges=scope_hyperedges,
                    warmup=warmup,
                    repeats=repeats,
                )
            )
    return rows


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def summarize_efficiency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["bench_label"]), str(row["method"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (bench_label, method), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "bench_label": bench_label,
            "method": method,
            "rows_n": len(group),
        }
        for field in TIMING_FIELDS:
            out[f"{field}_mean"] = _mean([float(row[field]) for row in group if isinstance(row.get(field), (int, float))])
        out["candidate_count_mean"] = _mean([float(row["candidate_count"]) for row in group if isinstance(row.get("candidate_count"), (int, float))])
        out["seed_count_mean"] = _mean([float(row["seed_count"]) for row in group if isinstance(row.get("seed_count"), (int, float))])
        out["expanded_count_mean"] = _mean([float(row["expanded_count"]) for row in group if isinstance(row.get("expanded_count"), (int, float))])
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


def write_e2_outputs(
    *,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    output_json: Path,
    output_csv: Path,
    summary_json: Path,
    summary_csv: Path,
    index_json: Path,
    index_csv: Path,
    metadata: dict[str, Any],
) -> None:
    write_json(output_json, {"metadata": metadata, "rows": rows})
    write_rows_csv(output_csv, rows)
    write_json(summary_json, {"metadata": metadata, "rows": summary_rows})
    write_rows_csv(summary_csv, summary_rows)
    write_json(index_json, {"metadata": metadata, "rows": index_rows})
    write_rows_csv(index_csv, index_rows)
