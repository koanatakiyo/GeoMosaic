#!/usr/bin/env python3
"""Sample Gemini metadata extraction on short docs and long-doc passage batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from geomosaic_hg.io import read_jsonl, write_jsonl


DEFAULT_PROJECT = "project-c4865349-a336-4706-8ca"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODELS = ("gemini-2.5-flash", "gemini-2.5-pro")


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "event_relation": {"type": "string"},
        "actors": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "array", "items": {"type": "string"}},
        "language_note": {"type": "string"},
        "section_outline": {"type": "array", "items": {"type": "string"}},
        "candidate_passage_hints": {"type": "array", "items": {"type": "string"}},
        "document_position": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "summary",
        "event_relation",
        "actors",
        "dates",
        "language_note",
        "section_outline",
        "candidate_passage_hints",
        "document_position",
        "confidence",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/0_external/official_doc_parsed"))
    parser.add_argument("--task-plan", type=Path, default=Path("data/1_intermediate/metadata_extraction/metadata_extraction_tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/1_intermediate/metadata_extraction/sample_gemini_metadata_outputs.jsonl"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--api-version", default="v1")
    parser.add_argument("--short-docs", type=int, default=2)
    parser.add_argument("--long-docs", type=int, default=1)
    parser.add_argument("--long-passages", type=int, default=3)
    parser.add_argument("--max-short-chars", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def load_rows(parsed_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs = list(read_jsonl(parsed_dir / "official_doc_text.jsonl"))
    passages = list(read_jsonl(parsed_dir / "passages.jsonl"))
    return docs, passages


def choose_samples(tasks: list[dict[str, Any]], docs_by_key: dict[tuple[str, str, str], dict[str, Any]], short_docs: int, long_docs: int) -> list[dict[str, Any]]:
    full_text_tasks = [
        task
        for task in tasks
        if task.get("input_strategy") == "document_full_text"
        and 0 < int(task.get("char_count") or 0) <= 4000
        and (task.get("document_id"), task.get("language"), task.get("source_filename")) in docs_by_key
    ]
    long_tasks = [
        task
        for task in tasks
        if task.get("input_strategy") == "passage_batch_map_reduce"
        and (task.get("document_id"), task.get("language"), task.get("source_filename")) in docs_by_key
    ]
    selected = sorted(full_text_tasks, key=lambda row: int(row.get("char_count") or 0))[:short_docs]
    selected.extend(sorted(long_tasks, key=lambda row: int(row.get("char_count") or 0), reverse=True)[:long_docs])
    return selected


def prompt_for_sample(task: dict[str, Any], doc: dict[str, Any], sample_text: str, sample_kind: str) -> str:
    return "\n".join(
        [
            "Extract non-load-bearing metadata from this official geopolitical document sample.",
            "Return JSON only. Do not infer claim support/contradiction labels.",
            "",
            f"event_id: {task.get('event_id')}",
            f"document_id: {task.get('document_id')}",
            f"language: {task.get('language')}",
            f"source_filename: {task.get('source_filename')}",
            f"sample_kind: {sample_kind}",
            "",
            "Fields:",
            "- summary: concise 2-4 sentence factual summary of this sample.",
            "- event_relation: how this document/sample relates to the event.",
            "- actors: named states, institutions, organizations, or officials.",
            "- dates: explicit dates found in the text.",
            "- language_note: one sentence about language/script and any translation caveat.",
            "- section_outline: bullet-like section labels visible in the sample.",
            "- candidate_passage_hints: 2-5 short hints useful for later retrieval.",
            "- document_position: one of supports, opposes, neutral, procedural, mixed, unknown.",
            "- confidence: number between 0 and 1.",
            "",
            "Text sample:",
            sample_text,
        ]
    )


def sample_text_for_task(
    task: dict[str, Any],
    doc: dict[str, Any],
    passages_by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
    long_passages: int,
    max_short_chars: int,
) -> tuple[str, str]:
    key = (task["document_id"], task["language"], task["source_filename"])
    if task["input_strategy"] == "document_full_text":
        return str(doc.get("text", ""))[:max_short_chars], "document_full_text"
    passages = passages_by_key.get(key, [])[:long_passages]
    parts = []
    for passage in passages:
        parts.append(
            f"[passage_id={passage.get('passage_id')} pages={passage.get('page_start')}-{passage.get('page_end')}]\n{passage.get('text', '')}"
        )
    return "\n\n".join(parts), f"long_doc_first_{len(passages)}_passages"


def call_model(client: genai.Client, model: str, prompt: str, temperature: float) -> tuple[str, dict[str, Any] | None]:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )
    text = response.text or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return text, parsed


def main() -> None:
    args = parse_args()
    docs, passages = load_rows(args.parsed_dir)
    tasks = list(read_jsonl(args.task_plan))
    docs_by_key = {(row["document_id"], row["language"], row["source_filename"]): row for row in docs}
    passages_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for passage in passages:
        key = (passage["document_id"], passage["language"], passage["source_filename"])
        passages_by_key.setdefault(key, []).append(passage)
    for rows in passages_by_key.values():
        rows.sort(key=lambda row: int(row.get("passage_index") or 0))

    selected = choose_samples(tasks, docs_by_key, short_docs=args.short_docs, long_docs=args.long_docs)
    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
        http_options=types.HttpOptions(api_version=args.api_version),
    )

    output_rows: list[dict[str, Any]] = []
    for task in selected:
        key = (task["document_id"], task["language"], task["source_filename"])
        doc = docs_by_key[key]
        sample_text, sample_kind = sample_text_for_task(task, doc, passages_by_key, args.long_passages, args.max_short_chars)
        prompt = prompt_for_sample(task, doc, sample_text, sample_kind)
        for model in args.models:
            raw_text, parsed = call_model(client, model, prompt, temperature=args.temperature)
            output_rows.append(
                {
                    "model": model,
                    "event_id": task["event_id"],
                    "document_id": task["document_id"],
                    "language": task["language"],
                    "source_filename": task["source_filename"],
                    "sample_kind": sample_kind,
                    "input_strategy": task["input_strategy"],
                    "input_chars": len(sample_text),
                    "parsed_ok": parsed is not None,
                    "parsed": parsed,
                    "raw_text": raw_text,
                }
            )

    count = write_jsonl(args.output, output_rows)
    print(json.dumps({"output": args.output.as_posix(), "rows": count, "models": args.models}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
