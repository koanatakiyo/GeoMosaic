"""Stage C load-bearing claim-to-passage grounding helpers."""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import read_jsonl, stable_hash, write_json, write_jsonl
from .metadata_extraction import append_jsonl, normalize_string_list, strip_json_fence


GroundingCaller = Callable[[str, str], dict[str, Any] | str]

GROUNDING_LABELS = {"support", "contradict", "context", "insufficient"}
GROUNDING_LABEL_ALIASES = {
    "supports": "support",
    "supported": "support",
    "supporting": "support",
    "contradiction": "contradict",
    "contradictions": "contradict",
    "contradicts": "contradict",
    "conflict": "contradict",
    "conflicting": "contradict",
    "contextual": "context",
    "background": "context",
    "related": "context",
    "neutral": "context",
    "no_evidence": "insufficient",
    "not_enough_evidence": "insufficient",
    "unknown": "insufficient",
    "none": "insufficient",
}

GROUNDING_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": sorted(GROUNDING_LABELS)},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "cited_passage_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label", "confidence", "rationale", "cited_passage_ids"],
}

STAGE_C_POLICY = "exclude_stage_b_outputs"
RETRIEVAL_METHOD = "lexical_overlap_v0"
DIAGNOSTIC_LABEL_OPTIONS = [
    "valid_evidence",
    "passage_quality_issue",
    "retrieval_failure",
    "language_retrieval_failure",
    "ambiguous_claim",
    "model_error",
    "skip",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def claim_source_layer(claim_id: str) -> str:
    parts = claim_id.split(":")
    return parts[1] if len(parts) > 1 else ""


def csv_values(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    else:
        values = [str(item).strip() for item in value]
    values = [item for item in values if item]
    return set(values) if values else None


def load_unique_claims(
    claims_path: str | Path,
    events: Iterable[str] | None = None,
    source_layers: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    wanted_events = csv_values(events)
    wanted_layers = csv_values(source_layers)
    claims_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(claims_path):
        claim_id = str(row.get("claim_id", ""))
        event_id = str(row.get("event_id", ""))
        if not claim_id or not event_id:
            continue
        if wanted_events and event_id not in wanted_events:
            continue
        layer = claim_source_layer(claim_id)
        if wanted_layers and layer not in wanted_layers:
            continue
        claims_by_id.setdefault(
            claim_id,
            {
                "claim_id": claim_id,
                "claim_text": str(row.get("claim_text", "")),
                "event_id": event_id,
                "claim_source_layer": layer,
            },
        )
    return sorted(claims_by_id.values(), key=lambda row: (row["event_id"], row["claim_source_layer"], row["claim_id"]))


def load_passages_by_event(parsed_dir: str | Path, languages: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    wanted_languages = csv_values(languages)
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(parsed_dir) / "passages.jsonl"):
        language = str(row.get("language", ""))
        if wanted_languages and language not in wanted_languages:
            continue
        event_id = str(row.get("event_id", ""))
        if event_id:
            by_event[event_id].append(row)
    for rows in by_event.values():
        rows.sort(key=lambda row: (str(row.get("document_id", "")), str(row.get("language", "")), int(row.get("passage_index") or 0)))
    return dict(by_event)


def lexical_overlap_score(claim_text: str, passage_text: str) -> float:
    claim_counts = Counter(tokenize(claim_text))
    passage_counts = Counter(tokenize(passage_text))
    if not claim_counts or not passage_counts:
        return 0.0
    overlap = sum(min(count, passage_counts[token]) for token, count in claim_counts.items())
    denom = math.sqrt(sum(claim_counts.values()) * sum(passage_counts.values()))
    return round(overlap / denom, 8) if denom else 0.0


def compact_passage(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "passage_id": str(row.get("passage_id", "")),
        "document_id": str(row.get("document_id", "")),
        "event_id": str(row.get("event_id", "")),
        "language": str(row.get("language", "")),
        "source_filename": str(row.get("source_filename", "")),
        "page_start": int(row.get("page_start") or 1),
        "page_end": int(row.get("page_end") or row.get("page_start") or 1),
        "char_start": int(row.get("char_start") or 0),
        "char_end": int(row.get("char_end") or 0),
        "score": score,
        "text": str(row.get("text", "")),
    }


def rank_passages_for_claim(claim: dict[str, Any], passages: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = [
        (lexical_overlap_score(str(claim.get("claim_text", "")), str(row.get("text", ""))), idx, row)
        for idx, row in enumerate(passages)
    ]
    scored.sort(key=lambda item: (-item[0], str(item[2].get("language", "")) != "en", item[1]))
    return [compact_passage(row, score) for score, _idx, row in scored[: max(1, top_k)]]


def task_for_claim(
    claim: dict[str, Any],
    passages: list[dict[str, Any]],
    top_k: int,
    audit_sample: bool = False,
) -> dict[str, Any]:
    task_id = f"claim_ground_{claim['event_id']}_{stable_hash(str(claim['claim_id']))}"
    return {
        "task_id": task_id,
        "task_type": "claim_grounding",
        "event_id": claim["event_id"],
        "claim_id": claim["claim_id"],
        "claim_text": claim["claim_text"],
        "claim_source_layer": claim.get("claim_source_layer", claim_source_layer(str(claim["claim_id"]))),
        "top_k": top_k,
        "retrieval_method": RETRIEVAL_METHOD,
        "passages": rank_passages_for_claim(claim, passages, top_k=top_k),
        "load_bearing": True,
        "stage_b_inputs_used": False,
        "stage_b_policy": STAGE_C_POLICY,
        "audit_sample": audit_sample,
    }


def plan_claim_grounding_tasks(
    claims_path: str | Path,
    parsed_dir: str | Path,
    output_dir: str | Path,
    events: Iterable[str] | None = None,
    source_layers: Iterable[str] | None = ("official",),
    languages: Iterable[str] | None = None,
    top_k: int = 3,
    limit: int | None = None,
    audit_sample_rate: float = 0.0,
    seed: int = 13,
) -> dict[str, Any]:
    claims = load_unique_claims(claims_path, events=events, source_layers=source_layers)
    passages_by_event = load_passages_by_event(parsed_dir, languages=languages)
    rng = random.Random(seed)

    tasks: list[dict[str, Any]] = []
    missing_passage_events: Counter[str] = Counter()
    for claim in claims:
        event_passages = passages_by_event.get(str(claim["event_id"]), [])
        if not event_passages:
            missing_passage_events[str(claim["event_id"])] += 1
            continue
        audit_sample = audit_sample_rate > 0 and rng.random() < audit_sample_rate
        tasks.append(task_for_claim(claim, event_passages, top_k=top_k, audit_sample=audit_sample))
        if limit is not None and len(tasks) >= limit:
            break

    output = Path(output_dir)
    tasks_path = output / "claim_grounding_tasks.jsonl"
    summary_path = output / "claim_grounding_plan_summary.json"
    write_jsonl(tasks_path, tasks)
    summary = {
        "claims_path": Path(claims_path).as_posix(),
        "parsed_dir": Path(parsed_dir).as_posix(),
        "tasks_path": tasks_path.as_posix(),
        "source_layers": sorted(csv_values(source_layers) or []),
        "languages": sorted(csv_values(languages) or []),
        "top_k": top_k,
        "retrieval_method": RETRIEVAL_METHOD,
        "stage_b_policy": STAGE_C_POLICY,
        "claims_considered": len(claims),
        "tasks": len(tasks),
        "audit_sample_rate": audit_sample_rate,
        "audit_sample_tasks": sum(1 for task in tasks if task.get("audit_sample")),
        "missing_passage_events": dict(sorted(missing_passage_events.items())),
        "load_bearing": True,
    }
    write_json(summary_path, summary)
    return summary


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    label = GROUNDING_LABEL_ALIASES.get(label, label)
    return label if label in GROUNDING_LABELS else "insufficient"


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, round(confidence, 6)))


def normalize_grounding_response(response: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(response, str):
        response = json.loads(strip_json_fence(response))
    return {
        "label": normalize_label(response.get("label")),
        "confidence": normalize_confidence(response.get("confidence")),
        "rationale": str(response.get("rationale", "")).strip(),
        "cited_passage_ids": normalize_string_list(response.get("cited_passage_ids")),
    }


def normalize_grounding_with_model_retries(
    prompt: str,
    model_id: str,
    model_caller: GroundingCaller,
    parse_retries: int = 2,
) -> tuple[dict[str, Any], int]:
    last_exc: json.JSONDecodeError | None = None
    current_prompt = prompt
    attempts = max(0, parse_retries) + 1
    for attempt in range(attempts):
        response = model_caller(current_prompt, model_id)
        try:
            return normalize_grounding_response(response), attempt + 1
        except json.JSONDecodeError as exc:
            last_exc = exc
            current_prompt = "\n".join(
                [
                    prompt,
                    "",
                    "Your previous response was not valid JSON.",
                    "Retry with compact valid JSON only using this schema:",
                    json.dumps(GROUNDING_RESPONSE_SCHEMA, ensure_ascii=False),
                ]
            )
    assert last_exc is not None
    raise last_exc


def grounding_prompt(task: dict[str, Any]) -> str:
    lines = [
        "Stage C load-bearing claim grounding task.",
        "Judge whether the provided official-document passages support, contradict, or only contextualize the claim.",
        "Use only the passages below. Do not use outside knowledge. If the passages are insufficient, choose insufficient.",
        "Stage B metadata summaries are intentionally excluded from this task.",
        "",
        "Return JSON only with fields:",
        "- label: one of support, contradict, context, insufficient.",
        "- confidence: number between 0 and 1.",
        "- rationale: brief explanation grounded in the cited passages.",
        "- cited_passage_ids: passage IDs used for the judgment.",
        "",
        f"event_id: {task.get('event_id')}",
        f"claim_id: {task.get('claim_id')}",
        f"claim: {task.get('claim_text')}",
        "",
        "Passages:",
    ]
    for passage in task.get("passages", []):
        lines.extend(
            [
                "",
                (
                    f"[passage_id={passage.get('passage_id')} document_id={passage.get('document_id')} "
                    f"language={passage.get('language')} pages={passage.get('page_start')}-{passage.get('page_end')} "
                    f"score={passage.get('score')}]"
                ),
                str(passage.get("text", "")),
            ]
        )
    return "\n".join(lines)


def existing_completed_judgment_keys(output_path: str | Path) -> set[tuple[str, str]]:
    return {
        (str(row.get("task_id", "")), str(row.get("model_name", "")))
        for row in read_jsonl(output_path)
        if row.get("status") == "completed" and row.get("task_id") and row.get("model_name")
    }


def completed_judgment_row(
    task: dict[str, Any],
    model_name: str,
    model_id: str,
    judgment: dict[str, Any],
    llm_calls: int,
) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id", ""),
        "task_type": "claim_grounding_judgment",
        "event_id": task.get("event_id", ""),
        "claim_id": task.get("claim_id", ""),
        "claim_source_layer": task.get("claim_source_layer", ""),
        "model_name": model_name,
        "model_id": model_id,
        "label": judgment["label"],
        "confidence": judgment["confidence"],
        "rationale": judgment["rationale"],
        "cited_passage_ids": judgment["cited_passage_ids"],
        "llm_calls": llm_calls,
        "load_bearing": True,
        "stage_b_inputs_used": False,
        "stage_b_policy": STAGE_C_POLICY,
        "status": "completed",
    }


def failed_judgment_row(task: dict[str, Any], model_name: str, model_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id", ""),
        "task_type": "claim_grounding_judgment",
        "event_id": task.get("event_id", ""),
        "claim_id": task.get("claim_id", ""),
        "model_name": model_name,
        "model_id": model_id,
        "load_bearing": True,
        "stage_b_inputs_used": False,
        "stage_b_policy": STAGE_C_POLICY,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def adjudicate_task_judgments(
    task: dict[str, Any],
    judgments: list[dict[str, Any]],
    expected_models: list[str],
) -> dict[str, Any]:
    completed = {str(row.get("model_name", "")): row for row in judgments if row.get("status") == "completed"}
    labels_by_model = {model: str(completed[model].get("label", "")) for model in sorted(completed) if model}
    missing_models = [model for model in expected_models if model not in completed]

    status = "incomplete"
    needs_human_audit = False
    human_audit_reason = ""
    canonical_label = ""
    confidence_mean = None
    if not missing_models and labels_by_model:
        labels = list(labels_by_model.values())
        if len(set(labels)) == 1:
            status = "agreement"
            canonical_label = labels[0]
            needs_human_audit = bool(task.get("audit_sample"))
            human_audit_reason = "random_agreement_sample" if needs_human_audit else ""
        else:
            status = "disagreement"
            needs_human_audit = True
            human_audit_reason = "model_disagreement"
        confidences = [float(row.get("confidence") or 0.0) for row in completed.values()]
        confidence_mean = round(sum(confidences) / len(confidences), 6) if confidences else None

    return {
        "task_id": task.get("task_id", ""),
        "task_type": "claim_grounding_adjudication",
        "event_id": task.get("event_id", ""),
        "claim_id": task.get("claim_id", ""),
        "claim_text": task.get("claim_text", ""),
        "claim_source_layer": task.get("claim_source_layer", ""),
        "labels_by_model": labels_by_model,
        "canonical_label": canonical_label,
        "confidence_mean": confidence_mean,
        "status": status,
        "missing_models": missing_models,
        "needs_human_audit": needs_human_audit,
        "human_audit_reason": human_audit_reason,
        "load_bearing": True,
        "stage_b_inputs_used": False,
        "stage_b_policy": STAGE_C_POLICY,
        "passage_ids": [str(p.get("passage_id", "")) for p in task.get("passages", [])],
    }


def build_adjudications(tasks: list[dict[str, Any]], judgment_rows: list[dict[str, Any]], expected_models: list[str]) -> list[dict[str, Any]]:
    judgments_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgment_rows:
        if row.get("task_id"):
            judgments_by_task[str(row["task_id"])].append(row)
    return [adjudicate_task_judgments(task, judgments_by_task.get(str(task.get("task_id", "")), []), expected_models) for task in tasks]


def task_passage_ids(task: dict[str, Any]) -> set[str]:
    return {str(passage.get("passage_id", "")) for passage in task.get("passages", []) if passage.get("passage_id")}


def model_judgment_for_audit(task: dict[str, Any], judgment: dict[str, Any] | None) -> dict[str, Any]:
    if not judgment:
        return {
            "label": "",
            "confidence": None,
            "rationale": "",
            "cited_passage_ids": [],
            "invalid_cited_passage_ids": [],
            "status": "missing",
        }
    allowed = task_passage_ids(task)
    cited = normalize_string_list(judgment.get("cited_passage_ids"))
    invalid = [passage_id for passage_id in cited if passage_id not in allowed]
    return {
        "label": str(judgment.get("label", "")),
        "confidence": judgment.get("confidence"),
        "rationale": str(judgment.get("rationale", "")),
        "cited_passage_ids": cited,
        "invalid_cited_passage_ids": invalid,
        "status": str(judgment.get("status", "")),
    }


def clean_human_evidence_text(text: str) -> str:
    value = str(text or "").replace("\f", "\n")
    cleaned_lines: list[str] = []
    for line in value.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", line.strip())
        if not line:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"\d+/\d+", line):
            continue
        if re.fullmatch(r"\*?\d{6,}\*?", line):
            continue
        if re.fullmatch(r"\d{2}-\d{5}(?: \([A-Z]\))?", line):
            continue
        cleaned_lines.append(line)
    value = "\n".join(cleaned_lines).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"\n+", " ", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\bStat es\b", "States", value)
    stripped = value.lstrip()
    if re.match(r"^(and|or|to|of|in|on|the|that|with)\b", stripped, flags=re.IGNORECASE):
        match = re.search(r"(?m)\b\d+\.\s+", stripped[:600])
        if match:
            stripped = stripped[match.start() :]
    return stripped.strip()


def passage_for_audit(passage: dict[str, Any]) -> dict[str, Any]:
    row = dict(passage)
    row["human_evidence_text"] = clean_human_evidence_text(str(passage.get("text", "")))
    return row


def audit_queue_row(
    task: dict[str, Any],
    adjudication: dict[str, Any],
    judgments_by_model: dict[str, dict[str, Any]],
    audit_flags: list[str],
) -> dict[str, Any]:
    return {
        "audit_mode": "diagnostic_triage",
        "task_id": task.get("task_id", ""),
        "event_id": task.get("event_id", ""),
        "claim_id": task.get("claim_id", ""),
        "claim_text": task.get("claim_text", ""),
        "claim_source_layer": task.get("claim_source_layer", ""),
        "adjudication_status": adjudication.get("status", ""),
        "canonical_label": adjudication.get("canonical_label", ""),
        "labels_by_model": adjudication.get("labels_by_model", {}),
        "audit_flags": audit_flags,
        "model_judgments": {
            model: model_judgment_for_audit(task, row)
            for model, row in sorted(judgments_by_model.items())
        },
        "passages": [passage_for_audit(passage) for passage in task.get("passages", [])],
        "confidence_use_policy": {
            "gemini": "display_only_do_not_sort_or_weight",
            "claude": "may_help_triage_but_not_ground_truth",
        },
        "diagnostic_label_options": DIAGNOSTIC_LABEL_OPTIONS,
        "audit_instruction": (
            "Diagnostic triage only, not ground truth. Judge whether the shown passages are usable for audit, "
            "whether retrieval failed, whether passage quality is the issue, or whether the model judgment is wrong. "
            "Ignore cited_passage_ids that are not present in passages. Do not use model confidence as a ground-truth label."
        ),
    }


def export_claim_grounding_audit(
    tasks_path: str | Path,
    judgments_path: str | Path,
    adjudication_path: str | Path,
    output_dir: str | Path,
    hk_insufficient_sample_size: int = 10,
    seed: int = 13,
) -> dict[str, Any]:
    tasks = {str(row.get("task_id", "")): row for row in read_jsonl(tasks_path) if row.get("task_id")}
    adjudications = [row for row in read_jsonl(adjudication_path) if row.get("task_id")]
    judgments: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(judgments_path):
        if row.get("status") == "completed" and row.get("task_id") and row.get("model_name"):
            judgments[str(row["task_id"])][str(row["model_name"])] = row

    hk_insufficient = [
        row
        for row in adjudications
        if row.get("event_id") == "hongkong"
        and row.get("status") == "agreement"
        and row.get("canonical_label") == "insufficient"
    ]
    rng = random.Random(seed)
    hk_sample_ids = {
        str(row.get("task_id", ""))
        for row in rng.sample(hk_insufficient, min(max(0, hk_insufficient_sample_size), len(hk_insufficient)))
    }

    queue: list[dict[str, Any]] = []
    invalid_citation_tasks = 0
    invalid_citations_by_model: Counter[str] = Counter()
    for adjudication in adjudications:
        task_id = str(adjudication.get("task_id", ""))
        task = tasks.get(task_id)
        if not task:
            continue
        model_rows = judgments.get(task_id, {})
        flags: list[str] = []
        if adjudication.get("status") == "disagreement":
            flags.append("model_disagreement")
        invalid_models: list[str] = []
        for model, row in sorted(model_rows.items()):
            invalid = model_judgment_for_audit(task, row)["invalid_cited_passage_ids"]
            if invalid:
                invalid_models.append(model)
                invalid_citations_by_model[model] += len(invalid)
                flags.append(f"invalid_citation:{model}")
        if invalid_models:
            invalid_citation_tasks += 1
        if task_id in hk_sample_ids:
            flags.append("hongkong_insufficient_agreement_sample")
        if flags:
            queue.append(audit_queue_row(task, adjudication, model_rows, audit_flags=flags))

    output = Path(output_dir)
    queue_path = output / "human_audit_queue.jsonl"
    summary_path = output / "human_audit_summary.json"
    write_jsonl(queue_path, queue)
    summary = {
        "tasks_path": Path(tasks_path).as_posix(),
        "judgments_path": Path(judgments_path).as_posix(),
        "adjudication_path": Path(adjudication_path).as_posix(),
        "audit_queue_path": queue_path.as_posix(),
        "total_tasks": len(tasks),
        "adjudication_rows": len(adjudications),
        "disagreement_tasks": sum(1 for row in adjudications if row.get("status") == "disagreement"),
        "invalid_citation_tasks": invalid_citation_tasks,
        "invalid_citations_by_model": dict(sorted(invalid_citations_by_model.items())),
        "hongkong_insufficient_agreement_total": len(hk_insufficient),
        "hongkong_insufficient_agreement_sample": len(hk_sample_ids),
        "audit_rows": len(queue),
        "audit_mode": "diagnostic_triage",
        "diagnostic_label_options": DIAGNOSTIC_LABEL_OPTIONS,
        "confidence_use_policy": {
            "gemini": "display_only_do_not_sort_or_weight",
            "claude": "may_help_triage_but_not_ground_truth",
        },
        "notes": [
            "Stage C audit is diagnostic triage, not ground-truth label construction.",
            "Gemini confidence is not used for ranking, weighting, or sampling.",
            "Invalid cited_passage_ids are audit flags; auditors should judge only from task passages.",
            "Hong Kong insufficient agreements are sampled to check lexical retrieval quality.",
        ],
    }
    write_json(summary_path, summary)
    return summary


def execute_claim_grounding(
    tasks_path: str | Path,
    output_path: str | Path,
    adjudication_path: str | Path,
    summary_path: str | Path,
    model_callers: dict[str, GroundingCaller],
    model_ids: dict[str, str],
    resume: bool = True,
    limit: int | None = None,
    task_ids: Iterable[str] | None = None,
    stop_on_error: bool = False,
    sleep_seconds: float = 0.0,
    parse_retries: int = 2,
) -> dict[str, Any]:
    selected_task_ids = csv_values(task_ids)
    tasks = [row for row in read_jsonl(tasks_path) if not selected_task_ids or str(row.get("task_id", "")) in selected_task_ids]
    selected_tasks = tasks[:limit] if limit is not None else tasks
    existing = existing_completed_judgment_keys(output_path) if resume else set()
    expected_models = list(model_callers)

    attempted = 0
    completed = 0
    failed = 0
    skipped_existing = 0
    llm_calls_made = 0
    completed_by_model: Counter[str] = Counter()

    for task in selected_tasks:
        for model_name, caller in model_callers.items():
            key = (str(task.get("task_id", "")), model_name)
            if key in existing:
                skipped_existing += 1
                continue
            attempted += 1
            model_id = model_ids[model_name]
            try:
                judgment, calls = normalize_grounding_with_model_retries(
                    grounding_prompt(task),
                    model_id,
                    caller,
                    parse_retries=parse_retries,
                )
                llm_calls_made += calls
                append_jsonl(output_path, completed_judgment_row(task, model_name, model_id, judgment, llm_calls=calls))
                completed += 1
                completed_by_model[model_name] += 1
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except Exception as exc:  # noqa: BLE001 - batch runner records row-level provider failures.
                append_jsonl(output_path, failed_judgment_row(task, model_name, model_id, exc))
                failed += 1
                if stop_on_error:
                    raise

    judgment_rows = list(read_jsonl(output_path))
    adjudications = build_adjudications(selected_tasks, judgment_rows, expected_models=expected_models)
    write_jsonl(adjudication_path, adjudications)
    status_counts = Counter(row.get("status", "") for row in adjudications)
    audit_count = sum(1 for row in adjudications if row.get("needs_human_audit"))
    summary = {
        "tasks_path": Path(tasks_path).as_posix(),
        "output_path": Path(output_path).as_posix(),
        "adjudication_path": Path(adjudication_path).as_posix(),
        "tasks_total": len(tasks),
        "tasks_selected": len(selected_tasks),
        "partial_adjudication": len(selected_tasks) < len(tasks),
        "adjudication_scope": "selected_tasks" if len(selected_tasks) < len(tasks) else "all_tasks",
        "models": expected_models,
        "model_ids": model_ids,
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "llm_calls_made": llm_calls_made,
        "completed_by_model": dict(sorted(completed_by_model.items())),
        "adjudication_status_counts": dict(sorted(status_counts.items())),
        "human_audit_needed": audit_count,
        "load_bearing": True,
        "stage_b_policy": STAGE_C_POLICY,
    }
    if summary["partial_adjudication"]:
        summary["partial_adjudication_note"] = (
            "adjudication_path was written for the selected task subset only; "
            "run without --limit/--task-id before using it as a full analysis table."
        )
    write_json(summary_path, summary)
    return summary
