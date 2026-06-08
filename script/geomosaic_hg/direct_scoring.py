"""E4 GeoGround-style direct scoring over document/evidence bundles.

This module intentionally scores document bundles, not claim-conditioned top-k
passages. Stage C passage judgments remain diagnostic triage only.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import read_jsonl, stable_hash, write_json, write_jsonl
from .metadata_extraction import append_jsonl, strip_json_fence


DirectScoringCaller = Callable[[str, str], str | dict[str, Any]]

DIRECT_SCORING_PROTOCOL = "geoground_direct_scoring_v0"
DEFAULT_LANGUAGE_PRIORITY = (
    "en",
    "zh-Hant-HK",
    "zh-Hans-CN",
    "zh",
    "uk",
    "sq",
    "ru",
    "fr",
    "es",
    "ar",
)


def csv_values(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    else:
        values = [str(item).strip() for item in value]
    values = [item for item in values if item]
    return set(values) if values else None


def language_rank(language: str, priority: Iterable[str] = DEFAULT_LANGUAGE_PRIORITY) -> tuple[int, str]:
    order = {value: idx for idx, value in enumerate(priority)}
    return (order.get(language, len(order)), language)


def parse_claim_id(claim_id: str) -> dict[str, str]:
    parts = str(claim_id).split(":")
    if len(parts) >= 5:
        return {
            "event_id": parts[0],
            "source_layer": parts[1],
            "source_vp": parts[2],
            "scored_vp": parts[3],
            "short_claim_id": parts[4],
        }
    return {
        "event_id": parts[0] if parts else "",
        "source_layer": "",
        "source_vp": "",
        "scored_vp": "",
        "short_claim_id": parts[-1] if parts else "",
    }


def claim_text_quality(text: str) -> int:
    value = str(text or "").strip()
    if not value:
        return -100
    score = min(len(value), 500)
    lowered = value.lower()
    for marker in ("parse_fail", " evaluated for ", "no mention", "does not mention"):
        if marker in lowered:
            score -= 120
    if re.search(r"\bclaim [A-Z]\d+\b", value):
        score -= 120
    return score


def load_direct_scoring_claims(
    claims_path: str | Path,
    source_layers: Iterable[str] | None = ("official",),
) -> list[dict[str, Any]]:
    wanted_layers = csv_values(source_layers)
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(claims_path):
        parsed = parse_claim_id(str(row.get("claim_id", "")))
        if wanted_layers and parsed["source_layer"] not in wanted_layers:
            continue
        event_id = str(row.get("event_id") or parsed["event_id"])
        scored_vp = parsed["scored_vp"]
        short_claim_id = parsed["short_claim_id"]
        if not event_id or not scored_vp or not short_claim_id:
            continue
        key = (event_id, scored_vp, short_claim_id)
        claim_text = str(row.get("claim_text", "")).strip()
        candidate = {
            "event_id": event_id,
            "scored_vp": scored_vp,
            "claim_id": short_claim_id,
            "claim_text": claim_text,
            "max": int(row.get("max", 2) or 2),
            "source_claim_id": row.get("claim_id", ""),
            "claim_text_source": "claim_evidence_hyperedges.claim_text",
        }
        if key not in candidates or claim_text_quality(claim_text) > claim_text_quality(str(candidates[key].get("claim_text", ""))):
            candidates[key] = candidate
    return sorted(candidates.values(), key=lambda row: (row["event_id"], row["scored_vp"], row["claim_id"]))


def load_score_maxima(score_dirs: Iterable[str | Path]) -> dict[tuple[str, str, str], int]:
    maxima: dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    for score_dir in score_dirs:
        root = Path(score_dir)
        if not root.exists():
            continue
        for path in sorted(root.glob("*_claim_audit.jsonl")):
            for row in read_jsonl(path):
                event_id = str(row.get("event", "")).lower()
                scored_vp = str(row.get("scored_vp", ""))
                claim_id = str(row.get("claim_id", ""))
                try:
                    max_value = int(row.get("max", 2) or 2)
                except (TypeError, ValueError):
                    max_value = 2
                if event_id and scored_vp and claim_id:
                    maxima[(event_id, scored_vp, claim_id)][max_value] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in maxima.items()}


def apply_score_maxima(claims: list[dict[str, Any]], maxima: dict[tuple[str, str, str], int]) -> list[dict[str, Any]]:
    out = []
    for claim in claims:
        row = dict(claim)
        row["max"] = maxima.get((row["event_id"], row["scored_vp"], row["claim_id"]), row.get("max", 2))
        out.append(row)
    return out


def manifest_by_source_filename(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("source_filename", "")): row for row in read_jsonl(manifest_path) if row.get("source_filename")}


def load_representative_documents(
    parsed_dir: str | Path,
    manifest_path: str | Path,
    language_priority: Iterable[str] = DEFAULT_LANGUAGE_PRIORITY,
) -> list[dict[str, Any]]:
    manifest = manifest_by_source_filename(manifest_path)
    candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(parsed_dir) / "official_doc_text.jsonl"):
        meta = manifest.get(str(row.get("source_filename", "")), {})
        event_id = str(row.get("event_id") or meta.get("event_id", ""))
        viewpoint = str(meta.get("viewpoint_origin") or "unknown")
        document_id = str(row.get("document_id") or meta.get("document_id", ""))
        if not event_id or not document_id:
            continue
        doc = {
            "document_id": document_id,
            "event_id": event_id,
            "viewpoint_origin": viewpoint,
            "evidence_scope": meta.get("evidence_scope", ""),
            "language": str(row.get("language") or meta.get("language", "")),
            "source_filename": row.get("source_filename", ""),
            "source_url": meta.get("source_url") or (row.get("extra") or {}).get("source_url", ""),
            "char_count": int(row.get("char_count") or len(str(row.get("text", "")))),
            "page_count": row.get("page_count"),
            "text": str(row.get("text", "")),
        }
        candidates[(event_id, viewpoint, document_id)].append(doc)

    selected = []
    for docs in candidates.values():
        docs.sort(key=lambda row: language_rank(str(row.get("language", "")), language_priority))
        selected.append(docs[0])
    return sorted(selected, key=lambda row: (row["event_id"], row["viewpoint_origin"], row["document_id"]))


def build_document_bundles(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        groups[(str(doc["event_id"]), str(doc["viewpoint_origin"]))].append(doc)
    bundles = []
    for (event_id, source_vp), docs in sorted(groups.items()):
        docs = sorted(docs, key=lambda row: (str(row.get("evidence_scope", "")), str(row.get("document_id", ""))))
        bundle_text = "\n\n".join(
            [
                f"[DOCUMENT {idx + 1}: {doc['document_id']} | language={doc['language']} | scope={doc.get('evidence_scope','')}]\n{doc['text']}"
                for idx, doc in enumerate(docs)
            ]
        )
        bundles.append(
            {
                "event_id": event_id,
                "source_vp": source_vp,
                "documents": [
                    {key: doc.get(key) for key in ("document_id", "language", "source_filename", "source_url", "evidence_scope", "char_count", "page_count")}
                    for doc in docs
                ],
                "bundle_text": bundle_text,
                "bundle_char_count": len(bundle_text),
            }
        )
    return bundles


def group_claims_by_event_scored_vp(claims: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[(str(claim["event_id"]), str(claim["scored_vp"]))].append(claim)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["claim_id"]))
    return grouped


def split_text_chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = end
    return chunks


def task_for_bundle(bundle: dict[str, Any], scored_vp: str, claims: list[dict[str, Any]], max_chars: int) -> dict[str, Any]:
    task_id = f"direct_score_{bundle['event_id']}_{bundle['source_vp']}_{scored_vp}_{stable_hash(bundle['bundle_text'] + scored_vp)}"
    strategy = "full_text" if len(bundle["bundle_text"]) <= max_chars else "document_map_reduce"
    chunks = split_text_chunks(bundle["bundle_text"], max_chars) if strategy != "full_text" else []
    return {
        "task_id": task_id,
        "task_type": "direct_scoring",
        "protocol": DIRECT_SCORING_PROTOCOL,
        "event_id": bundle["event_id"],
        "source_layer": "official",
        "source_vp": bundle["source_vp"],
        "scored_vp": scored_vp,
        "run_id": 1,
        "strategy": strategy,
        "documents": bundle["documents"],
        "bundle_char_count": bundle["bundle_char_count"],
        "bundle_text_sha256": stable_hash(bundle["bundle_text"]),
        "bundle_text": bundle["bundle_text"] if strategy == "full_text" else "",
        "chunks": chunks,
        "claims": [
            {key: claim.get(key) for key in ("claim_id", "claim_text", "max", "source_claim_id", "claim_text_source")}
            for claim in claims
        ],
        "stage_b_inputs_used": False,
        "stage_c_inputs_used": False,
        "stage_c_policy": "diagnostic_only_not_ground_truth",
    }


def build_direct_scoring_tasks(
    parsed_dir: str | Path,
    manifest_path: str | Path,
    claims_path: str | Path,
    output_dir: str | Path,
    source_layers: Iterable[str] | None = ("official",),
    score_dirs: Iterable[str | Path] = ("data/3_direct_scores/combined_official",),
    max_bundle_chars: int = 40000,
    events: Iterable[str] | None = None,
) -> dict[str, Any]:
    wanted_events = csv_values(events)
    docs = load_representative_documents(parsed_dir, manifest_path)
    if wanted_events:
        docs = [doc for doc in docs if doc["event_id"] in wanted_events]
    claims = load_direct_scoring_claims(claims_path, source_layers=source_layers)
    if wanted_events:
        claims = [claim for claim in claims if claim["event_id"] in wanted_events]
    claims = apply_score_maxima(claims, load_score_maxima(score_dirs))
    claim_groups = group_claims_by_event_scored_vp(claims)

    bundles = build_document_bundles(docs)
    tasks = []
    for bundle in bundles:
        scored_vps = sorted(scored_vp for event_id, scored_vp in claim_groups if event_id == bundle["event_id"])
        for scored_vp in scored_vps:
            tasks.append(task_for_bundle(bundle, scored_vp, claim_groups[(bundle["event_id"], scored_vp)], max_chars=max_bundle_chars))

    output = Path(output_dir)
    tasks_path = output / "direct_scoring_tasks.jsonl"
    summary_path = output / "direct_scoring_plan_summary.json"
    write_jsonl(tasks_path, tasks)
    summary = {
        "tasks_path": tasks_path.as_posix(),
        "parsed_dir": Path(parsed_dir).as_posix(),
        "manifest_path": Path(manifest_path).as_posix(),
        "claims_path": Path(claims_path).as_posix(),
        "protocol": DIRECT_SCORING_PROTOCOL,
        "documents_representative": len(docs),
        "bundles": len(bundles),
        "claims": len(claims),
        "tasks": len(tasks),
        "tasks_by_event": dict(sorted(Counter(task["event_id"] for task in tasks).items())),
        "tasks_by_strategy": dict(sorted(Counter(task["strategy"] for task in tasks).items())),
        "claim_text_warning": "Default claims are sourced from claim_evidence_hyperedges.claim_text; replace --claims with canonical GeoGround claim statements when available.",
    }
    write_json(summary_path, summary)
    return summary


def direct_scoring_prompt(task: dict[str, Any], chunk_text: str | None = None) -> str:
    document_text = chunk_text if chunk_text is not None else str(task.get("bundle_text", ""))
    claims_text = "\n".join(
        f"- {claim['claim_id']} (max={claim.get('max', 2)}): {claim.get('claim_text', '')}"
        for claim in task.get("claims", [])
    )
    return "\n".join(
        [
            "GeoGround-style direct scoring task.",
            "Score whether the supplied official document bundle expresses each scored-viewpoint claim.",
            "Use only the supplied document text. Do not use outside knowledge.",
            "Return JSON only with this schema:",
            '{"scores":[{"claim_id":"A1","score":0,"max":2,"justification":"short evidence-based reason"}]}',
            "Scoring: 0=absent/opposes/not expressed, 1=partial/implicit/presence-only, 2=explicit/strong. Use null only if the text is unreadable.",
            f"event_id: {task.get('event_id')}",
            f"source_vp: {task.get('source_vp')}",
            f"scored_vp: {task.get('scored_vp')}",
            "Claims:",
            claims_text,
            "Official document bundle:",
            document_text,
        ]
    )


def normalize_score(value: Any, max_value: int) -> int | None:
    if value is None:
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(max_value, score))


def parse_json_object_response(response: str) -> dict[str, Any]:
    value = strip_json_fence(str(response)).strip()
    if value.endswith("```"):
        value = value[:-3].strip()
    decoder = json.JSONDecoder(strict=False)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed, _ = decoder.raw_decode(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            trimmed = value[start : end + 1]
            try:
                return json.loads(trimmed)
            except json.JSONDecodeError:
                parsed, _ = decoder.raw_decode(trimmed)
                if isinstance(parsed, dict):
                    return parsed
                raise
        raise


def normalize_direct_scoring_response(response: str | dict[str, Any], expected_claims: Iterable[str]) -> dict[str, dict[str, Any]]:
    if isinstance(response, dict):
        data = response
    else:
        data = parse_json_object_response(str(response))
    raw_scores = data.get("scores", []) if isinstance(data, dict) else []
    by_claim: dict[str, dict[str, Any]] = {}
    for row in raw_scores:
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id", "")).strip()
        if not claim_id:
            continue
        try:
            max_value = int(row.get("max", 2) or 2)
        except (TypeError, ValueError):
            max_value = 2
        max_value = max(1, max_value)
        by_claim[claim_id] = {
            "claim_id": claim_id,
            "score": normalize_score(row.get("score"), max_value),
            "max": max_value,
            "justification": str(row.get("justification", "")),
        }
    for claim_id in expected_claims:
        by_claim.setdefault(str(claim_id), {"claim_id": str(claim_id), "score": None, "max": 2, "justification": "MISSING_SCORE"})
    return by_claim


def aggregate_chunk_scores(chunks: list[dict[str, dict[str, Any]]], claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for claim in claims:
        claim_id = str(claim["claim_id"])
        max_value = int(claim.get("max", 2) or 2)
        candidates = [chunk.get(claim_id) for chunk in chunks if chunk.get(claim_id)]
        scored = [row for row in candidates if row and row.get("score") is not None]
        if not scored:
            out[claim_id] = {"claim_id": claim_id, "score": None, "max": max_value, "justification": "MISSING_SCORE"}
            continue
        best = max(scored, key=lambda row: (int(row.get("score") or 0), len(str(row.get("justification", "")))))
        out[claim_id] = {
            "claim_id": claim_id,
            "score": int(best.get("score") or 0),
            "max": max_value,
            "justification": str(best.get("justification", "")),
        }
    return out


def score_task_with_model(task: dict[str, Any], model_id: str, model_caller: DirectScoringCaller) -> tuple[dict[str, dict[str, Any]], int]:
    expected = [str(claim["claim_id"]) for claim in task.get("claims", [])]
    if task.get("strategy") == "document_map_reduce":
        chunk_scores = []
        calls = 0
        for chunk in task.get("chunks", []):
            chunk_scores.append(normalize_direct_scoring_response(model_caller(direct_scoring_prompt(task, chunk_text=chunk), model_id), expected))
            calls += 1
        return aggregate_chunk_scores(chunk_scores, task.get("claims", [])), calls
    return normalize_direct_scoring_response(model_caller(direct_scoring_prompt(task), model_id), expected), 1


def result_rows_for_task(task: dict[str, Any], model_name: str, model_id: str, scores: dict[str, dict[str, Any]], llm_calls: int) -> list[dict[str, Any]]:
    claim_meta = {str(claim["claim_id"]): claim for claim in task.get("claims", [])}
    rows = []
    for claim_id, claim in sorted(claim_meta.items()):
        score = scores.get(claim_id, {"score": None, "max": claim.get("max", 2), "justification": "MISSING_SCORE"})
        rows.append(
            {
                "event": task.get("event_id", ""),
                "model": model_name,
                "model_id": model_id,
                "source_vp": task.get("source_vp", ""),
                "scored_vp": task.get("scored_vp", ""),
                "run_id": task.get("run_id", 1),
                "claim_id": claim_id,
                "score": score.get("score"),
                "max": int(claim.get("max", score.get("max", 2)) or 2),
                "justification": score.get("justification", ""),
                "task_id": task.get("task_id", ""),
                "protocol": DIRECT_SCORING_PROTOCOL,
                "strategy": task.get("strategy", ""),
                "source_layer": task.get("source_layer", "official"),
                "document_ids": [doc.get("document_id") for doc in task.get("documents", [])],
                "document_languages": [doc.get("language") for doc in task.get("documents", [])],
                "llm_calls": llm_calls,
                "stage_b_inputs_used": False,
                "stage_c_inputs_used": False,
            }
        )
    return rows


def existing_completed_task_models(
    output_path: str | Path,
    expected_claim_counts: dict[str, int] | None = None,
) -> set[tuple[str, str]]:
    if expected_claim_counts:
        scored_claims: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in read_jsonl(output_path):
            task_id = row.get("task_id")
            model = row.get("model")
            claim_id = row.get("claim_id")
            if task_id and model and claim_id and row.get("score") is not None:
                scored_claims[(str(task_id), str(model))].add(str(claim_id))
        return {
            key
            for key, claim_ids in scored_claims.items()
            if len(claim_ids) >= expected_claim_counts.get(key[0], 0)
        }

    completed = set()
    for row in read_jsonl(output_path):
        task_id = row.get("task_id")
        model = row.get("model")
        if task_id and model and row.get("score") is not None:
            completed.add((str(task_id), str(model)))
    return completed


def compact_direct_scoring_results(output_path: str | Path, backup_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(output_path)
    rows = list(read_jsonl(path))
    last_completed_row_for_claim = {
        (str(row.get("task_id")), str(row.get("model")), str(row.get("claim_id"))): idx
        for idx, row in enumerate(rows)
        if row.get("task_id")
        and row.get("model")
        and row.get("claim_id")
        and row.get("status") != "failed"
        and row.get("score") is not None
    }
    resolved = {
        (str(row.get("task_id")), str(row.get("model")))
        for row in rows
        if row.get("task_id") and row.get("model") and row.get("status") != "failed" and row.get("score") is not None
    }
    compacted = []
    removed_failed = 0
    removed_duplicate_completed = 0
    removed_resolved_missing_scores = 0
    for idx, row in enumerate(rows):
        key = (str(row.get("task_id")), str(row.get("model")))
        if row.get("status") == "failed" and key in resolved:
            removed_failed += 1
            continue
        completed_claim_key = (str(row.get("task_id")), str(row.get("model")), str(row.get("claim_id")))
        if (
            row.get("status") != "failed"
            and row.get("score") is None
            and completed_claim_key in last_completed_row_for_claim
        ):
            removed_resolved_missing_scores += 1
            continue
        if (
            row.get("status") != "failed"
            and row.get("score") is not None
            and completed_claim_key in last_completed_row_for_claim
            and idx != last_completed_row_for_claim[completed_claim_key]
        ):
            removed_duplicate_completed += 1
            continue
        compacted.append(row)

    if backup_path is not None and path.exists():
        Path(backup_path).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    write_jsonl(path, compacted)
    return {
        "output_path": path.as_posix(),
        "rows_before": len(rows),
        "rows_after": len(compacted),
        "removed_failed_rows": removed_failed,
        "removed_duplicate_completed_rows": removed_duplicate_completed,
        "removed_resolved_missing_score_rows": removed_resolved_missing_scores,
    }


def reset_direct_scoring_model_rows(
    output_path: str | Path,
    models: set[str],
    backup_path: str | Path,
) -> dict[str, Any]:
    path = Path(output_path)
    rows = list(read_jsonl(path))
    wanted = {str(model) for model in models}
    removed = [row for row in rows if str(row.get("model", "")) in wanted]
    kept = [row for row in rows if str(row.get("model", "")) not in wanted]
    write_jsonl(backup_path, removed)
    write_jsonl(path, kept)
    return {
        "output_path": path.as_posix(),
        "backup_path": Path(backup_path).as_posix(),
        "models": sorted(wanted),
        "rows_before": len(rows),
        "removed_rows": len(removed),
        "kept_rows": len(kept),
    }


def execute_direct_scoring(
    tasks_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    model_name: str,
    model_id: str,
    model_caller: DirectScoringCaller,
    resume: bool = True,
    limit: int | None = None,
    task_ids: Iterable[str] | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    wanted = csv_values(task_ids)
    tasks = [row for row in read_jsonl(tasks_path) if not wanted or str(row.get("task_id", "")) in wanted]
    selected = tasks[:limit] if limit is not None else tasks
    expected_claim_counts = {str(task.get("task_id", "")): len(task.get("claims", [])) for task in tasks}
    existing = existing_completed_task_models(output_path, expected_claim_counts=expected_claim_counts) if resume else set()

    attempted = completed = failed = skipped_existing = llm_calls = 0
    for task in selected:
        key = (str(task.get("task_id", "")), model_name)
        if key in existing:
            skipped_existing += 1
            continue
        attempted += 1
        try:
            scores, calls = score_task_with_model(task, model_id=model_id, model_caller=model_caller)
            for row in result_rows_for_task(task, model_name, model_id, scores, llm_calls=calls):
                append_jsonl(output_path, row)
            completed += 1
            llm_calls += calls
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception as exc:  # noqa: BLE001 - batch runner records failures and continues.
            failed += 1
            append_jsonl(
                output_path,
                {
                    "task_id": task.get("task_id", ""),
                    "event": task.get("event_id", ""),
                    "model": model_name,
                    "model_id": model_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "protocol": DIRECT_SCORING_PROTOCOL,
                },
            )

    summary = {
        "tasks_path": Path(tasks_path).as_posix(),
        "output_path": Path(output_path).as_posix(),
        "protocol": DIRECT_SCORING_PROTOCOL,
        "model": model_name,
        "model_id": model_id,
        "tasks_total": len(tasks),
        "tasks_selected": len(selected),
        "attempted_tasks": attempted,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "skipped_existing": skipped_existing,
        "llm_calls_made": llm_calls,
        "stage_b_inputs_used": False,
        "stage_c_inputs_used": False,
    }
    write_json(summary_path, summary)
    return summary
