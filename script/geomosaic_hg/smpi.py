"""Source-Modality Provenance Index (SMPI).

This is an offline, dependency-free implementation of the plan's SMPI idea:
temporal partitions, typed bitmap-like sets, lexical ANN-style seeds, and
claim-hyperedge incidence lists. It is intentionally small enough to run on
the copied project-local data without FAISS or GPU dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .build import load_tables
from .events import normalize_viewpoint
from .paths import BENCH_DIR
from .schema import MATCH_LEVELS, match_level_value, max_match_level
from .text import lexical_score


@dataclass(frozen=True)
class RetrievalConstraints:
    max_match_level: str = "L4"
    evidence_roles: frozenset[str] = frozenset()
    require_provenance: bool = True


class SMPI:
    def __init__(
        self,
        source_records: list[dict[str, Any]],
        evidence_assets: list[dict[str, Any]],
        source_asset_links: list[dict[str, Any]],
        hyperedges: list[dict[str, Any]],
    ) -> None:
        self.source_records = {r["source_id"]: r for r in source_records}
        self.evidence_assets = {a["asset_id"]: a for a in evidence_assets}
        self.source_asset_links = source_asset_links
        self.hyperedges = {h["hyperedge_id"]: h for h in hyperedges}

        self.links_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.links_by_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for link in source_asset_links:
            self.links_by_source[link["source_id"]].append(link)
            self.links_by_asset[link["asset_id"]].append(link)

        self.hyperedges_by_event: dict[str, set[str]] = defaultdict(set)
        self.hyperedges_by_source: dict[str, set[str]] = defaultdict(set)
        self.hyperedges_by_asset: dict[str, set[str]] = defaultdict(set)
        self.bitmap: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for hid, h in self.hyperedges.items():
            self.hyperedges_by_event[h["event_id"]].add(hid)
            for source_id in h.get("source_record_set", []):
                self.hyperedges_by_source[source_id].add(hid)
            for asset_id in h.get("evidence_asset_set", []):
                self.hyperedges_by_asset[asset_id].add(hid)
            for value in h.get("source_layer_set", []):
                self.bitmap["source_layer"][value].add(hid)
            for value in h.get("modality_set", []):
                self.bitmap["modality"][value].add(hid)
            for value in h.get("viewpoint_origin_set", []):
                self.bitmap["viewpoint_origin"][normalize_viewpoint(value)].add(hid)
            for value in h.get("match_level_multiset", []):
                self.bitmap["match_level"][value].add(hid)
            for value in h.get("evidence_role_multiset", []):
                self.bitmap["evidence_role"][value].add(hid)
            bucket = self.time_bucket(h.get("time_span", {}).get("end", ""))
            self.bitmap["time_bucket"][bucket].add(hid)

        self._doc_cache: dict[str, str] = {}

    @classmethod
    def from_dir(cls, bench_dir: str | Path = BENCH_DIR) -> "SMPI":
        tables = load_tables(Path(bench_dir))
        return cls(
            tables["source_records"],
            tables["evidence_assets"],
            tables["source_asset_links"],
            tables["claim_evidence_hyperedges"],
        )

    @staticmethod
    def time_bucket(timestamp: str) -> str:
        if len(timestamp) < 7:
            return "unknown"
        year = timestamp[:4]
        month = int(timestamp[5:7])
        quarter = ((month - 1) // 3) + 1
        return f"{year}Q{quarter}"

    def index_summary(self) -> dict[str, Any]:
        return {
            "source_records": len(self.source_records),
            "evidence_assets": len(self.evidence_assets),
            "source_asset_links": len(self.source_asset_links),
            "claim_evidence_hyperedges": len(self.hyperedges),
            "bitmap_dimensions": {dim: {k: len(v) for k, v in values.items()} for dim, values in self.bitmap.items()},
        }

    def hyperedge_text(self, h: dict[str, Any]) -> str:
        hid = h["hyperedge_id"]
        if hid in self._doc_cache:
            return self._doc_cache[hid]
        parts = [h.get("claim_text", ""), h.get("claim_id", ""), h.get("event_id", "")]
        for asset_id in h.get("evidence_asset_set", []):
            asset = self.evidence_assets.get(asset_id)
            if asset:
                parts.extend(
                    [
                        asset.get("caption_or_transcript", ""),
                        " ".join(asset.get("extracted_entities", []) or []),
                        " ".join(asset.get("extracted_claims", []) or []),
                    ]
                )
        for source_id in h.get("source_record_set", []):
            src = self.source_records.get(source_id)
            if src:
                parts.extend(
                    [
                        src.get("institution_or_outlet", ""),
                        src.get("document_type", ""),
                        src.get("extra", {}).get("text_preview", ""),
                    ]
                )
        doc = " ".join(str(p) for p in parts if p)
        self._doc_cache[hid] = doc
        return doc

    def relevance(self, query: str, h: dict[str, Any]) -> float:
        lexical = lexical_score(query, self.hyperedge_text(h)) if query else 0.0
        prior = float(h.get("relevance", h.get("confidence", 0.0)) or 0.0)
        if not query:
            return prior
        return min(1.0, 0.75 * lexical + 0.25 * prior)

    def _asset_temporal_ok(self, h: dict[str, Any], cutoff: str | None) -> bool:
        if not cutoff:
            return True
        for asset_id in h.get("evidence_asset_set", []):
            asset = self.evidence_assets.get(asset_id)
            if not asset:
                return False
            publish_time = asset.get("publish_time") or ""
            if publish_time > cutoff:
                return False
        return True

    def _asset_provenance_ok(self, h: dict[str, Any]) -> bool:
        if not h.get("provenance_trace"):
            return False
        for asset_id in h.get("evidence_asset_set", []):
            asset = self.evidence_assets.get(asset_id)
            if not asset or not asset.get("url_or_pointer") or not asset.get("perceptual_hash"):
                return False
        return True

    def feasible(
        self,
        h: dict[str, Any],
        source_layers: set[str] | None = None,
        modalities: set[str] | None = None,
        cutoff: str | None = None,
        constraints: RetrievalConstraints | None = None,
    ) -> bool:
        constraints = constraints or RetrievalConstraints()
        primary_layers = set(h.get("primary_source_layer_set") or h.get("source_layer_set", []))
        if source_layers and not primary_layers.intersection(source_layers):
            return False
        if modalities and not set(h.get("modality_set", [])).intersection(modalities):
            return False
        if cutoff and h.get("time_span", {}).get("end", "") > cutoff:
            return False
        if not self._asset_temporal_ok(h, cutoff):
            return False
        if constraints.require_provenance and not self._asset_provenance_ok(h):
            return False
        if max_match_level(h.get("match_level_multiset", [])) not in MATCH_LEVELS:
            return False
        if match_level_value(max_match_level(h.get("match_level_multiset", []))) > match_level_value(constraints.max_match_level):
            return False
        if constraints.evidence_roles:
            roles = set(h.get("evidence_role_multiset", []))
            if not roles.intersection(constraints.evidence_roles):
                return False
        return True

    def prune_candidates(
        self,
        source_layers: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        cutoff: str | None = None,
        constraints: RetrievalConstraints | None = None,
        event_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        layers = set(source_layers or [])
        mods = set(modalities or [])
        events = set(event_ids or [])
        out = []
        for h in self.hyperedges.values():
            if events and h.get("event_id") not in events:
                continue
            if self.feasible(h, layers or None, mods or None, cutoff, constraints):
                out.append(dict(h))
        return out

    def ann_seeds(self, query: str, candidates: list[dict[str, Any]], k: int = 50) -> list[dict[str, Any]]:
        scored = []
        for h in candidates:
            h = dict(h)
            h["_rel"] = self.relevance(query, h)
            scored.append(h)
        scored.sort(key=lambda row: (-row["_rel"], row["hyperedge_id"]))
        return scored[:k]

    def expand_seeds(self, seeds: list[dict[str, Any]], candidates: list[dict[str, Any]], depth: int = 1) -> list[dict[str, Any]]:
        if depth <= 0 or not seeds:
            return seeds
        candidate_ids = {h["hyperedge_id"] for h in candidates}
        selected = {h["hyperedge_id"] for h in seeds}
        frontier = set(selected)
        for _ in range(depth):
            nxt: set[str] = set()
            for hid in frontier:
                h = self.hyperedges.get(hid)
                if not h:
                    continue
                nxt.update(self.hyperedges_by_event.get(h["event_id"], set()))
                for source_id in h.get("source_record_set", []):
                    nxt.update(self.hyperedges_by_source.get(source_id, set()))
                for asset_id in h.get("evidence_asset_set", []):
                    nxt.update(self.hyperedges_by_asset.get(asset_id, set()))
            nxt &= candidate_ids
            selected |= nxt
            frontier = nxt
        by_id = {h["hyperedge_id"]: h for h in candidates}
        return [by_id[hid] for hid in sorted(selected) if hid in by_id]
