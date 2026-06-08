"""Human relevance audit sampling and summarization utilities."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .baselines import run_baselines
from .bpe import BPEConfig, retrieve
from .events import EVENTS
from .retrieval_experiments import EVENT_QUERIES, query_for_event
from .smpi import SMPI


AUDIT_METHODS = ["GeoMosaic-HG BPE", "Metadata++ + MMR", "Metadata++"]


def parse_csv_list(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def stable_audit_id(event_id: str, hyperedge_id: str) -> str:
    digest = hashlib.sha1(f"{event_id}:{hyperedge_id}".encode("utf-8")).hexdigest()[:12]
    return f"audit_{event_id}_{digest}"


def compact_text(value: Any, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def pipe_join(values: Iterable[Any]) -> str:
    return " | ".join(str(value) for value in values if value not in (None, ""))


def method_aliases(value: str | None) -> list[str]:
    aliases = {
        "bpe": "GeoMosaic-HG BPE",
        "mmr": "Metadata++ + MMR",
        "metadata": "Metadata++",
        "metadata++": "Metadata++",
    }
    out = []
    for item in parse_csv_list(value) or ["bpe", "mmr", "metadata"]:
        out.append(aliases.get(item.lower(), item))
    return out


def select_audit_ids(
    memberships: list[dict[str, Any]],
    *,
    event_ids: list[str],
    methods: list[str],
    max_pairs: int,
) -> set[str]:
    """Select audit ids while preserving top-rank coverage by event and method."""

    if max_pairs <= 0:
        return set()
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in memberships:
        by_event[str(row["event_id"])].append(row)

    per_event_limit = max(1, math.ceil(max_pairs / max(1, len(event_ids))))
    selected: list[str] = []
    seen: set[str] = set()

    for event_id in event_ids:
        event_rows = by_event.get(event_id, [])
        max_rank = max((int(row["rank"]) for row in event_rows), default=0)
        event_count = 0
        for rank in range(1, max_rank + 1):
            for method in methods:
                for row in sorted(event_rows, key=lambda item: str(item["audit_id"])):
                    if int(row["rank"]) != rank or row["method"] != method:
                        continue
                    audit_id = str(row["audit_id"])
                    if audit_id in seen:
                        continue
                    selected.append(audit_id)
                    seen.add(audit_id)
                    event_count += 1
                    if event_count >= per_event_limit or len(selected) >= max_pairs:
                        break
                if event_count >= per_event_limit or len(selected) >= max_pairs:
                    break
            if event_count >= per_event_limit or len(selected) >= max_pairs:
                break

    if len(selected) < max_pairs:
        for row in sorted(memberships, key=lambda item: (int(item["rank"]), str(item["event_id"]), methods.index(item["method"]) if item["method"] in methods else 99, str(item["audit_id"]))):
            audit_id = str(row["audit_id"])
            if audit_id in seen:
                continue
            selected.append(audit_id)
            seen.add(audit_id)
            if len(selected) >= max_pairs:
                break

    return set(selected)


def source_asset_preview(index: SMPI, h: dict[str, Any]) -> str:
    parts = []
    for source_id in h.get("source_record_set", []) or []:
        source = index.source_records.get(source_id, {})
        parts.append(
            "source="
            + pipe_join(
                [
                    source_id,
                    source.get("source_layer"),
                    source.get("institution_or_outlet"),
                    source.get("document_type"),
                ]
            )
        )
    for asset_id in h.get("evidence_asset_set", []) or []:
        asset = index.evidence_assets.get(asset_id, {})
        parts.append(
            "asset="
            + pipe_join(
                [
                    asset_id,
                    asset.get("modality"),
                    asset.get("asset_source"),
                    compact_text(asset.get("caption_or_transcript"), 180),
                ]
            )
        )
    return compact_text(" ; ".join(parts), 1200)


def audit_row_from_hyperedge(index: SMPI, query: str, h: dict[str, Any]) -> dict[str, Any]:
    event_id = str(h.get("event_id", ""))
    return {
        "audit_id": stable_audit_id(event_id, str(h.get("hyperedge_id", ""))),
        "human_relevance": "",
        "human_notes": "",
        "event_id": event_id,
        "query": query,
        "hyperedge_id": h.get("hyperedge_id", ""),
        "claim_id": h.get("claim_id", ""),
        "claim_text": compact_text(h.get("claim_text"), 1400),
        "source_vp": h.get("extra", {}).get("source_vp", ""),
        "scored_vp": h.get("extra", {}).get("scored_vp", ""),
        "relation": h.get("relation", ""),
        "primary_source_layer_set": pipe_join(h.get("primary_source_layer_set", []) or []),
        "source_layer_set": pipe_join(h.get("source_layer_set", []) or []),
        "modality_set": pipe_join(h.get("modality_set", []) or []),
        "viewpoint_origin_set": pipe_join(h.get("viewpoint_origin_set", []) or []),
        "source_record_set": pipe_join(h.get("source_record_set", []) or []),
        "evidence_asset_set": pipe_join(h.get("evidence_asset_set", []) or []),
        "match_level_multiset": pipe_join(h.get("match_level_multiset", []) or []),
        "evidence_role_multiset": pipe_join(h.get("evidence_role_multiset", []) or []),
        "provenance_trace": pipe_join(h.get("provenance_trace", []) or []),
        "evidence_preview": compact_text(index.hyperedge_text(h), 1400),
        "source_asset_preview": source_asset_preview(index, h),
        "ranked_by_methods": "",
        "method_ranks": "",
    }


def run_audit_sampling(
    index: SMPI,
    *,
    event_ids: list[str],
    methods: list[str],
    top_k: int,
    max_pairs: int,
    cutoff: str | None,
    budget: int,
    ann_k: int,
    expansion_depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audit_rows_by_id: dict[str, dict[str, Any]] = {}
    memberships: list[dict[str, Any]] = []

    for event_id in event_ids:
        query = query_for_event(event_id)
        config = BPEConfig(
            event_ids={event_id},
            cutoff=cutoff,
            hyperedge_budget=budget,
            ann_k=ann_k,
            expansion_depth=expansion_depth,
        )
        results = {"GeoMosaic-HG BPE": retrieve(index, query, config)}
        results.update(run_baselines(index, query, config))

        for method in methods:
            for rank, h in enumerate(results[method].get("selected_hyperedges", [])[:top_k], start=1):
                row = audit_row_from_hyperedge(index, query, h)
                audit_rows_by_id.setdefault(row["audit_id"], row)
                memberships.append(
                    {
                        "audit_id": row["audit_id"],
                        "event_id": event_id,
                        "query": query,
                        "method": method,
                        "rank": rank,
                        "hyperedge_id": h.get("hyperedge_id", ""),
                    }
                )

    selected_ids = select_audit_ids(memberships, event_ids=event_ids, methods=methods, max_pairs=max_pairs)
    selected_memberships = [row for row in memberships if row["audit_id"] in selected_ids]
    method_ranks: dict[str, list[str]] = defaultdict(list)
    method_names: dict[str, set[str]] = defaultdict(set)
    for row in selected_memberships:
        method_ranks[str(row["audit_id"])].append(f"{row['method']}@{row['rank']}")
        method_names[str(row["audit_id"])].add(str(row["method"]))

    audit_rows = []
    for audit_id in sorted(selected_ids, key=lambda aid: (audit_rows_by_id[aid]["event_id"], aid)):
        row = dict(audit_rows_by_id[audit_id])
        row["ranked_by_methods"] = pipe_join(sorted(method_names[audit_id]))
        row["method_ranks"] = pipe_join(sorted(method_ranks[audit_id]))
        audit_rows.append(row)

    metadata = {
        "event_ids": event_ids,
        "methods": methods,
        "top_k": top_k,
        "max_pairs": max_pairs,
        "audit_rows": len(audit_rows),
        "method_pair_rows": len(selected_memberships),
        "cutoff": cutoff,
        "budget": budget,
        "ann_k": ann_k,
        "expansion_depth": expansion_depth,
        "instructions": "Fill human_relevance with 2=directly evidentiary, 1=contextual/useful, 0=irrelevant or misleading.",
    }
    return audit_rows, selected_memberships, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_audit_html(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>GeoMosaic Relevance Audit</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:6px;vertical-align:top;font-size:12px;} th{background:#f4f4f4;position:sticky;top:0;} .claim{max-width:420px;} .preview{max-width:520px;} code{white-space:nowrap;}</style>",
        "</head><body>",
        "<h1>GeoMosaic-HG Relevance Audit</h1>",
        "<p>Label each row in the CSV column <code>human_relevance</code>: <b>2</b>=directly evidentiary, <b>1</b>=contextual/useful, <b>0</b>=irrelevant, misleading, or broken.</p>",
        f"<p>Rows: {metadata.get('audit_rows')} unique audit pairs; method memberships: {metadata.get('method_pair_rows')}.</p>",
        "<table><thead><tr><th>audit_id</th><th>label</th><th>event/query</th><th>method ranks</th><th>claim</th><th>evidence preview</th><th>source/assets</th></tr></thead><tbody>",
    ]
    for row in rows:
        body.append(
            "<tr>"
            f"<td><code>{html.escape(str(row['audit_id']))}</code></td>"
            "<td>0 / 1 / 2</td>"
            f"<td><b>{html.escape(str(row['event_id']))}</b><br>{html.escape(str(row['query']))}</td>"
            f"<td>{html.escape(str(row['method_ranks']))}</td>"
            f"<td class='claim'>{html.escape(str(row['claim_text']))}</td>"
            f"<td class='preview'>{html.escape(str(row['evidence_preview']))}</td>"
            f"<td class='preview'>{html.escape(str(row['source_asset_preview']))}</td>"
            "</tr>"
        )
    body.extend(["</tbody></table>", "</body></html>"])
    path.write_text("\n".join(body), encoding="utf-8")


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_relevance(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        score = int(float(str(value).strip()))
    except ValueError:
        return None
    return score if score in {0, 1, 2} else None


def dcg(labels: list[int], k: int) -> float:
    return sum(float(rel) / math.log2(rank + 1) for rank, rel in enumerate(labels[:k], start=1))


def ndcg_at_k(labels: list[int], k: int) -> float:
    ideal = sorted(labels[:k], reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(labels, k) / ideal_dcg


def summarize_relevance(
    labels: dict[str, dict[str, Any]],
    memberships: list[dict[str, Any]],
    *,
    k: int,
    relevance_threshold: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in memberships:
        if int(row["rank"]) <= k:
            grouped[(str(row["event_id"]), str(row["method"]))].append(row)

    rows = []
    for (event_id, method), group in sorted(grouped.items()):
        top = sorted(group, key=lambda row: int(row["rank"]))[:k]
        rels = [parse_relevance(labels.get(str(row["audit_id"]), {}).get("human_relevance")) for row in top]
        labeled = [rel for rel in rels if rel is not None]
        complete = len(labeled) == k
        rows.append(
            {
                "event_id": event_id,
                "method": method,
                "k": k,
                "expected_at_k": k,
                "labeled_at_k": len(labeled),
                "complete": complete,
                "p_at_k": (sum(1 for rel in labeled if rel >= relevance_threshold) / k) if complete else None,
                "ndcg_at_k": ndcg_at_k([int(rel) for rel in labeled], k) if complete else None,
                "mean_relevance": (sum(labeled) / len(labeled)) if labeled else None,
            }
        )
    return rows


def aggregate_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    out = []
    for method, group in sorted(grouped.items()):
        complete = [row for row in group if row["complete"]]
        out.append(
            {
                "event_id": "ALL",
                "method": method,
                "k": group[0]["k"] if group else None,
                "expected_at_k": sum(int(row["expected_at_k"]) for row in group),
                "labeled_at_k": sum(int(row["labeled_at_k"]) for row in group),
                "complete": len(complete) == len(group),
                "p_at_k": mean([row["p_at_k"] for row in complete]),
                "ndcg_at_k": mean([row["ndcg_at_k"] for row in complete]),
                "mean_relevance": mean([row["mean_relevance"] for row in group if row["mean_relevance"] is not None]),
            }
        )
    return out


def mean(values: list[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def write_summary_markdown(path: Path, rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Relevance Audit Summary",
        "",
        "Scores use `human_relevance`: 2=directly evidentiary, 1=contextual/useful, 0=irrelevant.",
        "",
        "## Aggregate By Method",
        "",
        "| Method | labeled@k | P@k | nDCG@k | Mean relevance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| {method} | {labeled}/{expected} | {p} | {ndcg} | {mean_rel} |".format(
                method=row["method"],
                labeled=row["labeled_at_k"],
                expected=row["expected_at_k"],
                p=format_metric(row["p_at_k"]),
                ndcg=format_metric(row["ndcg_at_k"]),
                mean_rel=format_metric(row["mean_relevance"]),
            )
        )
    lines.extend(["", "## Per Event", "", "| Event | Method | labeled@k | P@k | nDCG@k | Mean relevance |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            "| {event} | {method} | {labeled}/{expected} | {p} | {ndcg} | {mean_rel} |".format(
                event=row["event_id"],
                method=row["method"],
                labeled=row["labeled_at_k"],
                expected=row["expected_at_k"],
                p=format_metric(row["p_at_k"]),
                ndcg=format_metric(row["ndcg_at_k"]),
                mean_rel=format_metric(row["mean_relevance"]),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"
