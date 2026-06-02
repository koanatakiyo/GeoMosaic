"""Dry-run planning for non-load-bearing official-document metadata extraction."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, stable_hash, write_json, write_jsonl


NON_LOAD_BEARING_FIELDS = [
    "summary",
    "actors",
    "dates",
    "language_note",
    "section_outline",
    "candidate_passage_hints",
]
POINTER_RECORD_TYPES = {"news_pointer", "image_restricted_pointer", "article_pointer", "visual_pointer"}
POINTER_MODALITIES = {"image_restricted_pointer", "map_pointer"}


def doc_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("document_id", "")), str(row.get("language", "")), str(row.get("source_filename", "")))


def is_pointer_asset(row: dict[str, Any]) -> bool:
    extra = row.get("extra") or {}
    record_type = str(extra.get("record_type", ""))
    modality = str(row.get("modality", ""))
    if record_type in POINTER_RECORD_TYPES:
        return True
    if modality in POINTER_MODALITIES:
        return True
    return str(row.get("asset_source", "")) == "GDELT_DOC" and record_type.endswith("pointer")


def pointer_skip_type(row: dict[str, Any]) -> str:
    extra = row.get("extra") or {}
    record_type = str(extra.get("record_type", ""))
    modality = str(row.get("modality", ""))
    if record_type in POINTER_RECORD_TYPES:
        return record_type
    if modality in POINTER_MODALITIES:
        return modality
    return record_type or modality or "unknown"


def extraction_task_for_doc(
    row: dict[str, Any],
    passage_count: int,
    model_id: str,
    max_full_text_chars: int,
    batch_passages: int,
    prompt_version: str,
    schema_version: str,
) -> dict[str, Any]:
    document_id = str(row.get("document_id", ""))
    language = str(row.get("language", ""))
    event_id = str(row.get("event_id", ""))
    source_filename = str(row.get("source_filename", ""))
    char_count = int(row.get("char_count") or 0)
    input_strategy = "document_full_text" if char_count <= max_full_text_chars else "passage_batch_map_reduce"
    planned_batches = 1 if input_strategy == "document_full_text" else max(1, math.ceil(max(1, passage_count) / max(1, batch_passages)))
    return {
        "task_id": f"metadata_extract_{event_id}_{document_id}_{language}_{stable_hash(source_filename)}",
        "task_type": "official_doc_metadata_extraction",
        "input_kind": "parsed_official_document",
        "event_id": event_id,
        "document_id": document_id,
        "language": language,
        "source_filename": source_filename,
        "local_path": row.get("local_path", ""),
        "parse_quality": row.get("parse_quality", ""),
        "char_count": char_count,
        "page_count": row.get("page_count"),
        "passage_count": passage_count,
        "input_strategy": input_strategy,
        "planned_batches": planned_batches,
        "max_full_text_chars": max_full_text_chars,
        "batch_passages": batch_passages,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "input_text_sha256": (row.get("extra") or {}).get("text_sha256", ""),
        "fields_planned": NON_LOAD_BEARING_FIELDS,
        "load_bearing": False,
        "dry_run": True,
        "status": "planned",
        "pointer_extraction_policy": "skip_pointers",
    }


def plan_metadata_extraction(
    parsed_dir: str | Path,
    external_assets_path: str | Path | None,
    output_dir: str | Path,
    dry_run: bool = True,
    model_id: str = "gemini-2.5-flash",
    max_full_text_chars: int = 60000,
    batch_passages: int = 20,
    prompt_version: str = "official_doc_metadata_v0",
    schema_version: str = "official_doc_metadata_v0",
) -> dict[str, Any]:
    if not dry_run:
        raise ValueError("This planner only supports dry_run=True; no LLM calls are made.")

    parsed = Path(parsed_dir)
    output = Path(output_dir)
    docs = list(read_jsonl(parsed / "official_doc_text.jsonl"))
    passages = list(read_jsonl(parsed / "passages.jsonl"))
    passage_counts = Counter(doc_key(row) for row in passages)

    tasks: list[dict[str, Any]] = []
    documents_skipped_by_quality: Counter[str] = Counter()
    for row in docs:
        quality = str(row.get("parse_quality", ""))
        if quality != "ok" or not row.get("text"):
            documents_skipped_by_quality[quality or "unknown"] += 1
            continue
        tasks.append(
            extraction_task_for_doc(
                row,
                passage_count=passage_counts.get(doc_key(row), 0),
                model_id=model_id,
                max_full_text_chars=max_full_text_chars,
                batch_passages=batch_passages,
                prompt_version=prompt_version,
                schema_version=schema_version,
            )
        )

    pointer_assets_by_record_type: Counter[str] = Counter()
    pointer_assets_by_source_temporal_coverage: Counter[str] = Counter()
    external_assets_total = 0
    if external_assets_path:
        for row in read_jsonl(external_assets_path):
            external_assets_total += 1
            if is_pointer_asset(row):
                extra = row.get("extra") or {}
                pointer_assets_by_record_type[pointer_skip_type(row)] += 1
                pointer_assets_by_source_temporal_coverage[str(extra.get("source_temporal_coverage") or "unknown")] += 1

    tasks_path = output / "metadata_extraction_tasks.jsonl"
    summary_path = output / "metadata_extraction_dry_run_summary.json"
    write_jsonl(tasks_path, tasks)
    summary = {
        "dry_run": True,
        "llm_calls_made": 0,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "parsed_dir": parsed.as_posix(),
        "external_assets_path": Path(external_assets_path).as_posix() if external_assets_path else "",
        "output_dir": output.as_posix(),
        "tasks_path": tasks_path.as_posix(),
        "documents_total": len(docs),
        "documents_planned": len(tasks),
        "documents_skipped_by_quality": dict(sorted(documents_skipped_by_quality.items())),
        "tasks_total": len(tasks),
        "tasks_by_event": dict(sorted(Counter(task["event_id"] for task in tasks).items())),
        "tasks_by_language": dict(sorted(Counter(task["language"] for task in tasks).items())),
        "tasks_by_input_strategy": dict(sorted(Counter(task["input_strategy"] for task in tasks).items())),
        "planned_batches_total": sum(int(task["planned_batches"]) for task in tasks),
        "fields_planned": NON_LOAD_BEARING_FIELDS,
        "load_bearing": False,
        "pointer_extraction_policy": "skip_pointers",
        "external_assets_total": external_assets_total,
        "pointer_assets_skipped": sum(pointer_assets_by_record_type.values()),
        "pointer_assets_by_record_type": dict(sorted(pointer_assets_by_record_type.items())),
        "pointer_assets_by_source_temporal_coverage": dict(sorted(pointer_assets_by_source_temporal_coverage.items())),
    }
    write_json(summary_path, summary)
    return summary


def add_plan_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/0_external/official_doc_parsed"))
    parser.add_argument("--external-assets", type=Path, default=Path("data/0_external/external_assets.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/1_intermediate/metadata_extraction"))
    parser.add_argument("--model-id", default="gemini-2.5-flash")
    parser.add_argument("--max-full-text-chars", type=int, default=60000)
    parser.add_argument("--batch-passages", type=int, default=20)
    parser.add_argument("--prompt-version", default="official_doc_metadata_v0")
    parser.add_argument("--schema-version", default="official_doc_metadata_v0")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Plan tasks without making LLM calls. This is always true for this script.")
    return parser
