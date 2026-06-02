"""Execute Stage B non-load-bearing official-document metadata extraction."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .io import read_jsonl, write_json, write_jsonl
from .metadata_extraction_plan import NON_LOAD_BEARING_FIELDS, doc_key


MetadataCaller = Callable[[str, str], dict[str, Any] | str]


METADATA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "actors": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "array", "items": {"type": "string"}},
        "language_note": {"type": "string"},
        "section_outline": {"type": "array", "items": {"type": "string"}},
        "candidate_passage_hints": {"type": "array", "items": {"type": "string"}},
    },
    "required": NON_LOAD_BEARING_FIELDS,
}


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            first_payload = lines[0][3:].strip()
            if first_payload.lower().startswith("json"):
                first_payload = first_payload[4:].lstrip()
            if first_payload.endswith("```"):
                first_payload = first_payload[:-3].strip()
            lines = ([first_payload] if first_payload else []) + lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def normalize_metadata_response(response: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(response, str):
        response = json.loads(strip_json_fence(response))
    return {
        "summary": str(response.get("summary", "")).strip(),
        "actors": normalize_string_list(response.get("actors")),
        "dates": normalize_string_list(response.get("dates")),
        "language_note": str(response.get("language_note", "")).strip(),
        "section_outline": normalize_string_list(response.get("section_outline")),
        "candidate_passage_hints": normalize_string_list(response.get("candidate_passage_hints")),
    }


def normalize_with_model_retries(
    prompt: str,
    model_id: str,
    model_caller: MetadataCaller,
    parse_retries: int = 2,
) -> tuple[dict[str, Any], int]:
    last_exc: json.JSONDecodeError | None = None
    current_prompt = prompt
    attempts = max(0, parse_retries) + 1
    for attempt in range(attempts):
        response = model_caller(current_prompt, model_id)
        try:
            return normalize_metadata_response(response), attempt + 1
        except json.JSONDecodeError as exc:
            last_exc = exc
            current_prompt = "\n".join(
                [
                    prompt,
                    "",
                    "Your previous response was not valid JSON.",
                    "Retry with compact valid JSON only. Keep arrays short and close every string.",
                ]
            )
    assert last_exc is not None
    raise last_exc


def load_docs_and_passages(parsed_dir: str | Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str, str], list[dict[str, Any]]]]:
    parsed = Path(parsed_dir)
    docs = {doc_key(row): row for row in read_jsonl(parsed / "official_doc_text.jsonl")}
    passages: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in read_jsonl(parsed / "passages.jsonl"):
        passages.setdefault(doc_key(row), []).append(row)
    for rows in passages.values():
        rows.sort(key=lambda row: int(row.get("passage_index") or 0))
    return docs, passages


def existing_completed_task_ids(output_path: str | Path) -> set[str]:
    return {
        str(row.get("task_id", ""))
        for row in read_jsonl(output_path)
        if row.get("status") == "completed" and row.get("task_id")
    }


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, size)
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def base_prompt(task: dict[str, Any]) -> list[str]:
    lines = [
        "Extract Stage B non-load-bearing metadata from an official geopolitical document.",
        "Return JSON only. Do not infer claim support, contradiction, relevance, or ground-truth labels.",
        "Never produce support/contradict/context/insufficient labels.",
        "These outputs are forbidden from entering Stage C claim-grounding decisions.",
        "",
        f"event_id: {task.get('event_id')}",
        f"document_id: {task.get('document_id')}",
        f"language: {task.get('language')}",
        f"source_filename: {task.get('source_filename')}",
        "",
        "Required fields:",
        "- summary: concise factual summary.",
        "- actors: named states, institutions, organizations, or officials.",
        "- dates: explicit dates found in the text.",
        "- language_note: one sentence about language/script and translation caveats.",
        "- section_outline: visible section headings or logical sections.",
        "- candidate_passage_hints: short non-binding hints for later human/audit retrieval.",
    ]
    prompt_version = str(task.get("prompt_version", ""))
    if "compact" in prompt_version:
        lines.extend(
            [
                "",
                "Length constraints:",
                "- summary <= 120 words.",
                "- actors <= 12 items.",
                "- dates <= 12 items.",
                "- section_outline <= 12 items.",
                "- candidate_passage_hints <= 12 items.",
                "- each array item should be a short phrase, not a paragraph.",
            ]
        )
    return lines


def full_text_prompt(task: dict[str, Any], doc: dict[str, Any]) -> str:
    lines = base_prompt(task)
    lines.extend(["", "Official document text:", str(doc.get("text", ""))])
    return "\n".join(lines)


def passage_batch_prompt(task: dict[str, Any], batch: list[dict[str, Any]], batch_index: int, batch_total: int) -> str:
    lines = base_prompt(task)
    lines.extend(["", f"Input is passage batch {batch_index + 1}/{batch_total}. Extract metadata for this batch only."])
    for passage in batch:
        lines.append("")
        lines.append(
            f"[passage_id={passage.get('passage_id')} pages={passage.get('page_start')}-{passage.get('page_end')}]"
        )
        lines.append(str(passage.get("text", "")))
    return "\n".join(lines)


def reduce_prompt(task: dict[str, Any], batch_rows: list[dict[str, Any]]) -> str:
    lines = base_prompt(task)
    lines.extend(
        [
            "",
            "Reduce these batch-level metadata extractions into one document-level metadata record.",
            "Deduplicate actors, dates, section labels, and passage hints. Keep the result concise.",
            "",
            json.dumps([row["metadata"] for row in batch_rows], ensure_ascii=False, indent=2),
        ]
    )
    return "\n".join(lines)


def completed_row(task: dict[str, Any], metadata: dict[str, Any], batch_count: int, llm_calls: int) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id", ""),
        "task_type": task.get("task_type", "official_doc_metadata_extraction"),
        "event_id": task.get("event_id", ""),
        "document_id": task.get("document_id", ""),
        "language": task.get("language", ""),
        "source_filename": task.get("source_filename", ""),
        "model_id": task.get("model_id", ""),
        "prompt_version": task.get("prompt_version", ""),
        "schema_version": task.get("schema_version", ""),
        "input_strategy": task.get("input_strategy", ""),
        "input_text_sha256": task.get("input_text_sha256", ""),
        "metadata": metadata,
        "fields": NON_LOAD_BEARING_FIELDS,
        "load_bearing": False,
        "stage_c_excluded": True,
        "stage_c_policy": "never_use_stage_b_outputs",
        "batch_count": batch_count,
        "llm_calls": llm_calls,
        "status": "completed",
    }


def failed_row(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id", ""),
        "task_type": task.get("task_type", "official_doc_metadata_extraction"),
        "event_id": task.get("event_id", ""),
        "document_id": task.get("document_id", ""),
        "language": task.get("language", ""),
        "source_filename": task.get("source_filename", ""),
        "model_id": task.get("model_id", ""),
        "input_strategy": task.get("input_strategy", ""),
        "load_bearing": False,
        "stage_c_excluded": True,
        "stage_c_policy": "never_use_stage_b_outputs",
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def execute_task(
    task: dict[str, Any],
    docs: dict[tuple[str, str, str], dict[str, Any]],
    passages: dict[tuple[str, str, str], list[dict[str, Any]]],
    model_caller: MetadataCaller,
    batch_output_path: str | Path,
    sleep_seconds: float = 0.0,
    parse_retries: int = 2,
) -> tuple[dict[str, Any], int]:
    key = (str(task.get("document_id", "")), str(task.get("language", "")), str(task.get("source_filename", "")))
    doc = docs.get(key)
    if doc is None:
        raise ValueError(
            "No parsed official document found for "
            f"document_id={key[0]!r}, language={key[1]!r}, source_filename={key[2]!r}"
        )
    model_id = str(task.get("model_id") or "gemini-2.5-flash")
    if task.get("input_strategy") == "passage_batch_map_reduce":
        passage_rows = passages.get(key, [])
        batches = chunked(passage_rows, int(task.get("batch_passages") or 20))
        batch_outputs: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches):
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            metadata, calls = normalize_with_model_retries(
                passage_batch_prompt(task, batch, batch_index, len(batches)),
                model_id,
                model_caller,
                parse_retries=parse_retries,
            )
            batch_row = {
                "task_id": task.get("task_id", ""),
                "batch_id": f"{task.get('task_id', '')}_batch_{batch_index:04d}",
                "batch_index": batch_index,
                "batch_total": len(batches),
                "passage_ids": [row.get("passage_id", "") for row in batch],
                "metadata": metadata,
                "llm_calls": calls,
                "load_bearing": False,
                "stage_c_excluded": True,
                "status": "completed",
            }
            append_jsonl(batch_output_path, batch_row)
            batch_outputs.append(batch_row)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        reduced, reduce_calls = normalize_with_model_retries(
            reduce_prompt(task, batch_outputs),
            model_id,
            model_caller,
            parse_retries=parse_retries,
        )
        llm_calls = sum(int(row.get("llm_calls", 1)) for row in batch_outputs) + reduce_calls
        return completed_row(task, reduced, batch_count=len(batch_outputs), llm_calls=llm_calls), llm_calls

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    metadata, calls = normalize_with_model_retries(
        full_text_prompt(task, doc),
        model_id,
        model_caller,
        parse_retries=parse_retries,
    )
    return completed_row(task, metadata, batch_count=0, llm_calls=calls), calls


def compact_jsonl_by_id(path: str | Path, key_name: str) -> int:
    rows = list(read_jsonl(path))
    if not rows:
        return 0
    latest_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_name, ""))
        if key:
            existing = latest_by_key.get(key)
            if existing and existing.get("status") == "completed" and row.get("status") != "completed":
                continue
            latest_by_key[key] = row
    write_jsonl(path, latest_by_key.values())
    return len(latest_by_key)


def dedupe_extend(values: list[str], new_values: list[str], seen: set[str], max_items: int) -> None:
    for value in new_values:
        normalized = value.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            values.append(normalized)
            seen.add(key)
        if len(values) >= max_items:
            break


def deterministic_reduce_metadata(batch_rows: list[dict[str, Any]], max_items: int = 30) -> dict[str, Any]:
    summaries: list[str] = []
    actors: list[str] = []
    dates: list[str] = []
    language_notes: list[str] = []
    section_outline: list[str] = []
    hints: list[str] = []
    seen_actors: set[str] = set()
    seen_dates: set[str] = set()
    seen_notes: set[str] = set()
    seen_sections: set[str] = set()
    seen_hints: set[str] = set()

    for row in sorted(batch_rows, key=lambda item: int(item.get("batch_index") or 0)):
        metadata = normalize_metadata_response(row.get("metadata") or {})
        if metadata["summary"]:
            summaries.append(metadata["summary"])
        dedupe_extend(actors, metadata["actors"], seen_actors, max_items)
        dedupe_extend(dates, metadata["dates"], seen_dates, max_items)
        dedupe_extend(language_notes, [metadata["language_note"]], seen_notes, 3)
        dedupe_extend(section_outline, metadata["section_outline"], seen_sections, max_items)
        dedupe_extend(hints, metadata["candidate_passage_hints"], seen_hints, max_items)

    summary = " ".join(summaries[:3]).strip()
    if len(summaries) > 3:
        summary = f"{summary} Deterministic merge of {len(summaries)} batch-level summaries."
    return {
        "summary": summary,
        "actors": actors,
        "dates": dates,
        "language_note": " | ".join(language_notes),
        "section_outline": section_outline,
        "candidate_passage_hints": hints,
    }


def finalize_failed_map_reduce_from_batches(
    tasks_path: str | Path,
    output_path: str | Path,
    batch_output_path: str | Path,
    summary_path: str | Path,
    max_items: int = 30,
) -> dict[str, Any]:
    tasks_by_id = {str(row.get("task_id", "")): row for row in read_jsonl(tasks_path)}
    latest_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(output_path):
        task_id = str(row.get("task_id", ""))
        if task_id:
            latest_rows[task_id] = row

    batch_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(batch_output_path):
        task_id = str(row.get("task_id", ""))
        if task_id:
            batch_rows_by_task.setdefault(task_id, []).append(row)

    finalized = 0
    skipped: list[dict[str, Any]] = []
    for task_id, row in list(latest_rows.items()):
        task = tasks_by_id.get(task_id)
        if row.get("status") != "failed" or not task or task.get("input_strategy") != "passage_batch_map_reduce":
            continue
        batch_rows = batch_rows_by_task.get(task_id, [])
        if not batch_rows:
            skipped.append({"task_id": task_id, "reason": "no_batch_rows"})
            continue
        expected_total = int(batch_rows[0].get("batch_total") or 0)
        if expected_total and len(batch_rows) < expected_total:
            skipped.append({"task_id": task_id, "reason": f"incomplete_batches:{len(batch_rows)}/{expected_total}"})
            continue
        metadata = deterministic_reduce_metadata(batch_rows, max_items=max_items)
        completed = completed_row(
            task,
            metadata,
            batch_count=len(batch_rows),
            llm_calls=sum(int(item.get("llm_calls") or 1) for item in batch_rows),
        )
        completed["reduce_strategy"] = "deterministic_batch_merge_fallback"
        completed["status"] = "completed"
        latest_rows[task_id] = completed
        finalized += 1

    write_jsonl(output_path, latest_rows.values())
    summary = {
        "tasks_path": Path(tasks_path).as_posix(),
        "output_path": Path(output_path).as_posix(),
        "batch_output_path": Path(batch_output_path).as_posix(),
        "summary_path": Path(summary_path).as_posix(),
        "finalized": finalized,
        "skipped": skipped,
        "output_rows": len(latest_rows),
        "remaining_failed": sum(1 for row in latest_rows.values() if row.get("status") == "failed"),
        "reduce_strategy": "deterministic_batch_merge_fallback",
    }
    write_json(summary_path, summary)
    return summary


def execute_metadata_extraction(
    tasks_path: str | Path,
    parsed_dir: str | Path,
    output_path: str | Path,
    batch_output_path: str | Path,
    summary_path: str | Path,
    model_caller: MetadataCaller,
    resume: bool = True,
    limit: int | None = None,
    stop_on_error: bool = False,
    sleep_seconds: float = 0.0,
    parse_retries: int = 2,
    compact_output: bool = True,
) -> dict[str, Any]:
    tasks = list(read_jsonl(tasks_path))
    docs, passages = load_docs_and_passages(parsed_dir)
    completed_existing = existing_completed_task_ids(output_path) if resume else set()
    skipped_existing = 0
    completed = 0
    failed = 0
    llm_calls_made = 0
    attempted = 0
    status_by_event: Counter[str] = Counter()

    for task in tasks:
        if limit is not None and attempted >= limit:
            break
        task_id = str(task.get("task_id", ""))
        if task_id in completed_existing:
            skipped_existing += 1
            continue
        attempted += 1
        try:
            row, calls = execute_task(
                task,
                docs,
                passages,
                model_caller,
                batch_output_path,
                sleep_seconds=sleep_seconds,
                parse_retries=parse_retries,
            )
            append_jsonl(output_path, row)
            completed += 1
            llm_calls_made += calls
            status_by_event[str(task.get("event_id", ""))] += 1
        except Exception as exc:  # noqa: BLE001 - execution should record task-level failures.
            failed += 1
            append_jsonl(output_path, failed_row(task, exc))
            if stop_on_error:
                raise

    output_rows_after_compaction = compact_jsonl_by_id(output_path, "task_id") if compact_output else None
    batch_rows_after_compaction = compact_jsonl_by_id(batch_output_path, "batch_id") if compact_output else None
    summary = {
        "tasks_path": Path(tasks_path).as_posix(),
        "parsed_dir": Path(parsed_dir).as_posix(),
        "output_path": Path(output_path).as_posix(),
        "batch_output_path": Path(batch_output_path).as_posix(),
        "tasks_total": len(tasks),
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "llm_calls_made": llm_calls_made,
        "parse_retries": parse_retries,
        "compact_output": compact_output,
        "output_rows_after_compaction": output_rows_after_compaction,
        "batch_rows_after_compaction": batch_rows_after_compaction,
        "load_bearing": False,
        "stage_c_policy": "never_use_stage_b_outputs",
        "completed_by_event": dict(sorted(status_by_event.items())),
    }
    write_json(summary_path, summary)
    return summary
