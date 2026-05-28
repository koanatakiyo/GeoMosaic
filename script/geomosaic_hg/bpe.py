"""Balanced Provenance Expansion over a capped coverage objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .schema import match_level_value, max_match_level
from .smpi import RetrievalConstraints, SMPI


@dataclass
class BPEConfig:
    event_ids: set[str] = field(default_factory=set)
    source_layers: set[str] = field(default_factory=set)
    modalities: set[str] = field(default_factory=set)
    cutoff: str | None = None
    hyperedge_budget: int = 10
    max_match_level: str = "L4"
    evidence_roles: set[str] = field(default_factory=set)
    ann_k: int = 100
    expansion_depth: int = 1
    viewpoint_cap: int = 1
    source_cap: int = 1
    modality_cap: int = 1
    alpha: float = 1.0
    beta: float = 1.0
    lambda_rel: float = 0.25


class CoverageObjective:
    """Monotone submodular capped coverage plus non-negative modular relevance."""

    def __init__(
        self,
        viewpoint_cap: int = 1,
        source_cap: int = 1,
        modality_cap: int = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_rel: float = 0.25,
    ) -> None:
        self.viewpoint_cap = viewpoint_cap
        self.source_cap = source_cap
        self.modality_cap = modality_cap
        self.alpha = alpha
        self.beta = beta
        self.lambda_rel = lambda_rel

    @staticmethod
    def _capped_count(rows: Iterable[dict[str, Any]], field: str, cap: int) -> int:
        counts: dict[str, int] = {}
        for row in rows:
            for value in row.get(field, []) or []:
                counts[value] = counts.get(value, 0) + 1
        return sum(min(v, cap) for v in counts.values())

    def score(self, rows: list[dict[str, Any]]) -> float:
        v = self._capped_count(rows, "viewpoint_origin_set", self.viewpoint_cap)
        s = self._capped_count(rows, "source_layer_set", self.source_cap)
        m = self._capped_count(rows, "modality_set", self.modality_cap)
        rel = sum(max(0.0, float(row.get("_rel", row.get("relevance", row.get("confidence", 0.0))) or 0.0)) for row in rows)
        return v + self.alpha * s + self.beta * m + self.lambda_rel * rel

    def marginal_gain(self, selected: list[dict[str, Any]], candidate: dict[str, Any]) -> float:
        return self.score([*selected, candidate]) - self.score(selected)


def greedy_select(candidates: list[dict[str, Any]], budget: int, objective: CoverageObjective) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = {h["hyperedge_id"]: h for h in candidates}
    while remaining and len(selected) < budget:
        best = max(
            remaining.values(),
            key=lambda h: (objective.marginal_gain(selected, h), float(h.get("_rel", h.get("confidence", 0.0)) or 0.0), h["hyperedge_id"]),
        )
        gain = objective.marginal_gain(selected, best)
        if gain < 0:
            break
        selected.append(best)
        remaining.pop(best["hyperedge_id"], None)
    return selected


def induced_assets(index: SMPI, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    asset_ids = sorted({asset_id for h in selected for asset_id in h.get("evidence_asset_set", [])})
    return [index.evidence_assets[asset_id] for asset_id in asset_ids if asset_id in index.evidence_assets]


def induced_sources(index: SMPI, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ids = sorted({source_id for h in selected for source_id in h.get("source_record_set", [])})
    return [index.source_records[source_id] for source_id in source_ids if source_id in index.source_records]


def provenance_trace(selected: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for h in selected:
        for item in h.get("provenance_trace", []) or []:
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out


def retrieve(index: SMPI, query: str, config: BPEConfig | None = None) -> dict[str, Any]:
    config = config or BPEConfig()
    constraints = RetrievalConstraints(
        max_match_level=config.max_match_level,
        evidence_roles=frozenset(config.evidence_roles),
        require_provenance=True,
    )
    pruned = index.prune_candidates(config.source_layers, config.modalities, config.cutoff, constraints, config.event_ids)
    seeds = index.ann_seeds(query, pruned, config.ann_k)
    expanded = index.expand_seeds(seeds, pruned, config.expansion_depth)
    objective = CoverageObjective(
        viewpoint_cap=config.viewpoint_cap,
        source_cap=config.source_cap,
        modality_cap=config.modality_cap,
        alpha=config.alpha,
        beta=config.beta,
        lambda_rel=config.lambda_rel,
    )
    selected = greedy_select(expanded, config.hyperedge_budget, objective)
    return {
        "query": query,
        "config": {
            "source_layers": sorted(config.source_layers),
            "event_ids": sorted(config.event_ids),
            "modalities": sorted(config.modalities),
            "cutoff": config.cutoff,
            "hyperedge_budget": config.hyperedge_budget,
            "max_match_level": config.max_match_level,
            "evidence_roles": sorted(config.evidence_roles),
            "ann_k": config.ann_k,
            "expansion_depth": config.expansion_depth,
        },
        "candidate_count": len(pruned),
        "seed_count": len(seeds),
        "expanded_count": len(expanded),
        "objective_value": round(objective.score(selected), 6),
        "selected_hyperedges": selected,
        "induced_assets": induced_assets(index, selected),
        "induced_sources": induced_sources(index, selected),
        "provenance_trace": provenance_trace(selected),
        "max_selected_match_level": max_match_level([lvl for h in selected for lvl in h.get("match_level_multiset", [])]) if selected else None,
        "max_selected_match_level_value": match_level_value(max_match_level([lvl for h in selected for lvl in h.get("match_level_multiset", [])])) if selected else None,
    }
