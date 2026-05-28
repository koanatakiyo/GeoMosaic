"""Baseline retrieval methods from the execution plan."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from .bpe import BPEConfig, induced_assets, induced_sources, provenance_trace
from .metrics import summarize_result
from .smpi import RetrievalConstraints, SMPI
from .text import jaccard_similarity, metadata_atoms


def _base_candidates(index: SMPI, query: str, config: BPEConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    constraints = RetrievalConstraints(
        max_match_level=config.max_match_level,
        evidence_roles=frozenset(config.evidence_roles),
        require_provenance=True,
    )
    candidates = index.prune_candidates(config.source_layers, config.modalities, config.cutoff, constraints, config.event_ids)
    seeds = index.ann_seeds(query, candidates, max(config.ann_k, config.hyperedge_budget))
    return candidates, seeds


def _package(
    index: SMPI,
    query: str,
    selected: list[dict[str, Any]],
    candidate_count: int,
    seed_count: int,
    name: str,
) -> dict[str, Any]:
    return {
        "method": name,
        "query": query,
        "candidate_count": candidate_count,
        "seed_count": seed_count,
        "expanded_count": seed_count,
        "objective_value": None,
        "selected_hyperedges": selected,
        "induced_assets": induced_assets(index, selected),
        "induced_sources": induced_sources(index, selected),
        "provenance_trace": provenance_trace(selected),
    }


def naive_rag(index: SMPI, query: str, config: BPEConfig) -> dict[str, Any]:
    relaxed = replace(config, source_layers=set(), modalities=set(), evidence_roles=set(), max_match_level="L4")
    candidates, seeds = _base_candidates(index, query, relaxed)
    selected = seeds[: config.hyperedge_budget]
    return _package(index, query, selected, len(candidates), len(seeds), "NaiveRAG")


def metadata_filter(index: SMPI, query: str, config: BPEConfig) -> dict[str, Any]:
    candidates, seeds = _base_candidates(index, query, config)
    selected = seeds[: config.hyperedge_budget]
    return _package(index, query, selected, len(candidates), len(seeds), "Metadata++")


def metadata_mmr(index: SMPI, query: str, config: BPEConfig, lambda_rel: float = 0.65) -> dict[str, Any]:
    candidates, seeds = _base_candidates(index, query, config)
    remaining = {h["hyperedge_id"]: h for h in seeds}
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < config.hyperedge_budget:
        def mmr_score(h: dict[str, Any]) -> tuple[float, float, str]:
            rel = float(h.get("_rel", h.get("confidence", 0.0)) or 0.0)
            if not selected:
                diversity = 1.0
            else:
                atoms = metadata_atoms(h)
                diversity = min(1.0 - jaccard_similarity(atoms, metadata_atoms(s)) for s in selected)
            return (lambda_rel * rel + (1.0 - lambda_rel) * diversity, rel, h["hyperedge_id"])

        best = max(remaining.values(), key=mmr_score)
        selected.append(best)
        remaining.pop(best["hyperedge_id"], None)
    return _package(index, query, selected, len(candidates), len(seeds), "Metadata++ + MMR")


def random_sm(index: SMPI, query: str, config: BPEConfig, seed: int = 13) -> dict[str, Any]:
    candidates, seeds = _base_candidates(index, query, config)
    rng = random.Random(seed)
    layers = [tuple(h.get("source_layer_set", [])) for h in seeds]
    modalities = [tuple(h.get("modality_set", [])) for h in seeds]
    rng.shuffle(layers)
    rng.shuffle(modalities)
    shuffled: list[dict[str, Any]] = []
    for idx, h in enumerate(seeds):
        row = dict(h)
        row["source_layer_set"] = list(layers[idx]) if idx < len(layers) else row.get("source_layer_set", [])
        row["modality_set"] = list(modalities[idx]) if idx < len(modalities) else row.get("modality_set", [])
        shuffled.append(row)
    shuffled.sort(key=lambda h: (-(float(h.get("_rel", 0.0) or 0.0)), h["hyperedge_id"]))
    selected = shuffled[: config.hyperedge_budget]
    return _package(index, query, selected, len(candidates), len(seeds), "Random-SM")


def run_baselines(index: SMPI, query: str, config: BPEConfig) -> dict[str, dict[str, Any]]:
    return {
        "NaiveRAG": naive_rag(index, query, config),
        "Metadata++": metadata_filter(index, query, config),
        "Metadata++ + MMR": metadata_mmr(index, query, config),
        "Random-SM": random_sm(index, query, config),
    }


def summarize_methods(results: dict[str, dict[str, Any]], cutoff: str | None = None, k: int = 10) -> list[dict[str, Any]]:
    rows = []
    for method, result in results.items():
        row = {"method": method}
        row.update(summarize_result(result, cutoff=cutoff, k=k))
        rows.append(row)
    return rows
