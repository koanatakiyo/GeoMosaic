#!/usr/bin/env python3
"""Capture raw E4 scorer responses for failed direct-scoring tasks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.direct_scoring import (  # noqa: E402
    aggregate_chunk_scores,
    direct_scoring_prompt,
    normalize_direct_scoring_response,
)
from geomosaic_hg.io import read_jsonl, write_jsonl  # noqa: E402
from llm_manager import LLMManager  # noqa: E402
from run_direct_scoring import DEFAULT_MODEL_IDS, build_model_caller, provider_available, provider_unavailable_message  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/1_intermediate/direct_scoring/direct_scoring_tasks.jsonl")
    parser.add_argument("--output", default="data/1_intermediate/direct_scoring/direct_scoring_raw_captures.jsonl")
    parser.add_argument("--model", required=True, choices=sorted(DEFAULT_MODEL_IDS))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=40000)
    parser.add_argument("--sleep-seconds", type=float, default=4.0)
    parser.add_argument("--parse-after-capture", action="store_true")
    args = parser.parse_args()

    task_ids = set(args.task_id)
    tasks = [row for row in read_jsonl(args.tasks) if row.get("task_id") in task_ids]
    missing = sorted(task_ids - {str(row.get("task_id")) for row in tasks})
    if missing:
        raise SystemExit(f"Missing task ids in {args.tasks}: {missing}")

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]
    manager = LLMManager(models_to_init=[args.model])
    if not provider_available(manager, args.model):
        raise SystemExit(provider_unavailable_message(args.model))
    caller = build_model_caller(manager, args.model, max_output_tokens=args.max_output_tokens)

    rows = []
    for task in tasks:
        prompts = []
        if task.get("strategy") == "document_map_reduce":
            prompts = [
                (idx, direct_scoring_prompt(task, chunk_text=chunk))
                for idx, chunk in enumerate(task.get("chunks", []))
            ]
        else:
            prompts = [(None, direct_scoring_prompt(task))]

        chunk_scores = []
        for chunk_index, prompt in prompts:
            captured = {
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": args.model,
                "model_id": model_id,
                "task_id": task.get("task_id"),
                "event": task.get("event_id"),
                "source_vp": task.get("source_vp"),
                "scored_vp": task.get("scored_vp"),
                "strategy": task.get("strategy"),
                "chunk_index": chunk_index,
                "chunk_count": len(prompts),
                "prompt_sha256": __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
            }
            try:
                raw = caller(prompt, model_id)
                captured["status"] = "captured"
                captured["raw_response"] = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                captured["raw_response_prefix"] = str(captured["raw_response"])[:1000]
                if args.parse_after_capture:
                    expected = [str(claim.get("claim_id")) for claim in task.get("claims", [])]
                    parsed = normalize_direct_scoring_response(captured["raw_response"], expected)
                    captured["parse_status"] = "parsed"
                    captured["parsed_scores"] = parsed
                    chunk_scores.append(parsed)
            except Exception as exc:
                captured["status"] = "failed"
                captured["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(captured)
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        if args.parse_after_capture and chunk_scores and task.get("strategy") == "document_map_reduce":
            rows.append(
                {
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "model": args.model,
                    "model_id": model_id,
                    "task_id": task.get("task_id"),
                    "event": task.get("event_id"),
                    "source_vp": task.get("source_vp"),
                    "scored_vp": task.get("scored_vp"),
                    "strategy": task.get("strategy"),
                    "status": "aggregate_probe",
                    "chunk_count": len(prompts),
                    "parsed_chunk_count": len(chunk_scores),
                    "aggregated_scores": aggregate_chunk_scores(chunk_scores, task.get("claims", [])),
                }
            )

    write_jsonl(args.output, rows)
    print(json.dumps({"output": args.output, "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
