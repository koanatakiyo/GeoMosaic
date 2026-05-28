"""Retrieval and source-sensitivity metrics."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .events import normalize_viewpoint
from .io import read_jsonl


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    n = len(values)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def viewpoint_coverage(selected: list[dict[str, Any]], viewpoint_universe: set[str] | None = None) -> float:
    observed = {normalize_viewpoint(v) for h in selected for v in h.get("viewpoint_origin_set", [])}
    if viewpoint_universe is None:
        viewpoint_universe = {"us_anglo", "eu", "russia", "china", "un"}
    if not viewpoint_universe:
        return 0.0
    return len(observed & viewpoint_universe) / len(viewpoint_universe)


def viewpoint_balance(selected: list[dict[str, Any]], viewpoint_universe: set[str] | None = None) -> float:
    values = [normalize_viewpoint(v) for h in selected for v in h.get("viewpoint_origin_set", [])]
    if viewpoint_universe is None:
        viewpoint_universe = {"us_anglo", "eu", "russia", "china", "un"}
    values = [v for v in values if v in viewpoint_universe]
    if len(viewpoint_universe) <= 1 or not values:
        return 0.0
    return entropy(values) / math.log(len(viewpoint_universe))


def source_diversity(selected: list[dict[str, Any]]) -> int:
    return len({source_id for h in selected for source_id in h.get("source_record_set", [])})


def layer_diversity(selected: list[dict[str, Any]]) -> int:
    return len({layer for h in selected for layer in h.get("source_layer_set", [])})


def modality_coverage(selected: list[dict[str, Any]]) -> int:
    return len({mod for h in selected for mod in h.get("modality_set", [])})


def provenance_completeness(selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    complete = sum(1 for h in selected if h.get("provenance_trace"))
    return complete / len(selected)


def temporal_leakage_rate(selected: list[dict[str, Any]], cutoff: str | None) -> float:
    if not selected or not cutoff:
        return 0.0
    leaked = 0
    for h in selected:
        if h.get("time_span", {}).get("end", "") > cutoff:
            leaked += 1
    return leaked / len(selected)


def jaccard_distance(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 1.0
    return 1.0 - (len(sa & sb) / len(sa | sb))


def evidence_subgraph_jaccard(result_a: dict[str, Any], result_b: dict[str, Any]) -> float:
    ids_a = [h["hyperedge_id"] for h in result_a.get("selected_hyperedges", [])]
    ids_b = [h["hyperedge_id"] for h in result_b.get("selected_hyperedges", [])]
    return jaccard_distance(ids_a, ids_b)


def recall_at_k(selected: list[dict[str, Any]], relevant_claim_ids: set[str] | None = None, k: int = 10) -> float | None:
    if relevant_claim_ids is None:
        return None
    if not relevant_claim_ids:
        return 0.0
    top = selected[:k]
    found = {h.get("claim_id") for h in top if h.get("claim_id") in relevant_claim_ids}
    return len(found) / len(relevant_claim_ids)


def summarize_result(result: dict[str, Any], cutoff: str | None = None, k: int = 10) -> dict[str, Any]:
    selected = result.get("selected_hyperedges", [])[:k]
    return {
        "selected": len(selected),
        "candidate_count": result.get("candidate_count"),
        "seed_count": result.get("seed_count"),
        "expanded_count": result.get("expanded_count"),
        "objective_value": result.get("objective_value"),
        "viewpoint_coverage": round(viewpoint_coverage(selected), 6),
        "viewpoint_balance": round(viewpoint_balance(selected), 6),
        "source_diversity": source_diversity(selected),
        "layer_diversity": layer_diversity(selected),
        "modality_coverage": modality_coverage(selected),
        "provenance_completeness": round(provenance_completeness(selected), 6),
        "temporal_leakage_rate": round(temporal_leakage_rate(selected, cutoff), 6),
    }


def load_sas_long(csv_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["sas"] = float(row["sas"])
                row["raw_score"] = float(row["raw_score"])
                row["max_possible"] = float(row["max_possible"])
                rows.append(row)
    return rows


def dsas_from_claim_audit(audit_path: str | Path, scored_vp: str | None = None) -> float | None:
    score = 0.0
    max_score = 0.0
    wanted = normalize_viewpoint(scored_vp) if scored_vp else None
    for row in read_jsonl(audit_path):
        if wanted and normalize_viewpoint(str(row.get("scored_vp", ""))) != wanted:
            continue
        if row.get("score") is None:
            continue
        score += float(row.get("score", 0.0))
        max_score += float(row.get("max", 0.0))
    if max_score <= 0:
        return None
    return score / max_score
