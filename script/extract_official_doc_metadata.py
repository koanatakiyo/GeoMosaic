#!/usr/bin/env python3
"""Run Stage B non-load-bearing official-document metadata extraction."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from geomosaic_hg.metadata_extraction import METADATA_RESPONSE_SCHEMA, execute_metadata_extraction


DEFAULT_LOCATION = "us-central1"


class VertexGeminiMetadataCaller:
    """Small retrying wrapper around Vertex AI Gemini structured output."""

    def __init__(
        self,
        project: str,
        location: str,
        api_version: str,
        temperature: float,
        max_output_tokens: int,
        retries: int,
        retry_backoff_seconds: float,
    ) -> None:
        from google import genai
        from google.genai import types

        self.types = types
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version=api_version),
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retries = max(0, retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def __call__(self, prompt: str, model_id: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=METADATA_RESPONSE_SCHEMA,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                return response.text or ""
            except Exception as exc:  # noqa: BLE001 - retry wrapper should preserve provider exceptions.
                last_exc = exc
                if attempt >= self.retries:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        assert last_exc is not None
        raise last_exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/1_intermediate/metadata_extraction/metadata_extraction_tasks.jsonl"),
        help="Stage B task plan produced by plan_metadata_extraction.py.",
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("data/0_external/official_doc_parsed"),
        help="Directory containing official_doc_text.jsonl and passages.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/1_intermediate/metadata_extraction/official_doc_metadata_enrichment.jsonl"),
        help="Document-level Stage B metadata sidecar output.",
    )
    parser.add_argument(
        "--batch-output",
        type=Path,
        default=Path("data/1_intermediate/metadata_extraction/official_doc_metadata_batch_outputs.jsonl"),
        help="Intermediate batch metadata output for long documents.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/1_intermediate/metadata_extraction/official_doc_metadata_enrichment_summary.json"),
        help="Execution summary JSON path.",
    )
    parser.add_argument("--model-id", default="gemini-2.5-flash")
    parser.add_argument(
        "--prompt-version",
        default="",
        help="Optional prompt-version override. Use official_doc_metadata_v1_compact only with --overwrite or a separate output.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT") or "",
        help="Google Cloud project for Vertex AI. Defaults to GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_QUOTA_PROJECT.",
    )
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--api-version", default="v1")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=15.0)
    parser.add_argument("--parse-retries", type=int, default=2, help="Retry model calls that return malformed JSON.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between provider calls.")
    parser.add_argument("--task-id", action="append", default=[], help="Run only this task_id. May be passed multiple times.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N not-yet-completed tasks.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output, batch output, and summary before running. Implies --no-resume.",
    )
    return parser.parse_args()


def rewrite_task_fields(tasks_path: Path, model_id: str, prompt_version: str = "", task_ids: list[str] | None = None) -> Path | None:
    """Create a temporary task file only when requested task fields differ."""

    wanted = set(task_ids or [])
    if not model_id and not prompt_version and not wanted:
        return None
    rows: list[dict[str, Any]] = []
    changed = bool(wanted)
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if wanted and row.get("task_id") not in wanted:
            continue
        if row.get("model_id") != model_id:
            row["model_id"] = model_id
            changed = True
        if prompt_version and row.get("prompt_version") != prompt_version:
            row["prompt_version"] = prompt_version
            changed = True
        rows.append(row)
    if wanted and len(rows) != len(wanted):
        found = {str(row.get("task_id", "")) for row in rows}
        missing = sorted(wanted - found)
        raise SystemExit(f"Task id(s) not found: {', '.join(missing)}")
    if not changed:
        return None
    suffix_parts = [value.replace("/", "_") for value in (model_id, prompt_version, "selected" if wanted else "") if value]
    temp_path = tasks_path.with_suffix(f".{'.'.join(suffix_parts)}.tmp.jsonl")
    temp_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return temp_path


def main() -> None:
    args = parse_args()
    if not args.project:
        raise SystemExit("Missing --project or GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_QUOTA_PROJECT.")
    if args.overwrite:
        args.resume = False
        for path in (args.output, args.batch_output, args.summary):
            if path.exists():
                path.unlink()

    tasks_path = rewrite_task_fields(args.tasks, args.model_id, prompt_version=args.prompt_version, task_ids=args.task_id) or args.tasks
    try:
        caller = VertexGeminiMetadataCaller(
            project=args.project,
            location=args.location,
            api_version=args.api_version,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        summary = execute_metadata_extraction(
            tasks_path=tasks_path,
            parsed_dir=args.parsed_dir,
            output_path=args.output,
            batch_output_path=args.batch_output,
            summary_path=args.summary,
            model_caller=caller,
            resume=args.resume,
            limit=args.limit,
            stop_on_error=args.stop_on_error,
            sleep_seconds=args.sleep_seconds,
            parse_retries=args.parse_retries,
        )
    finally:
        if tasks_path != args.tasks and tasks_path.exists():
            tasks_path.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
