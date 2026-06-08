#!/usr/bin/env python3
"""Retry failed E4 direct-scoring rows and compact resolved failures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.direct_scoring import compact_direct_scoring_results, execute_direct_scoring  # noqa: E402
from geomosaic_hg.io import read_jsonl  # noqa: E402
from llm_manager import LLMManager  # noqa: E402
from run_direct_scoring import DEFAULT_MODEL_IDS, build_model_caller, provider_available, provider_unavailable_message  # noqa: E402


def csv_arg(value: str | None) -> set[str] | None:
    if value is None:
        return None
    values = {item.strip() for item in value.split(",") if item.strip()}
    return values or None


def parse_model_overrides(values: list[str] | None) -> dict[str, str]:
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--model-id expects MODEL=ID, got: {value}")
        model, model_id = value.split("=", 1)
        overrides[model.strip()] = model_id.strip()
    return overrides


def collect_failed_task_ids(output_path: str | Path, models: set[str] | None = None) -> dict[str, list[str]]:
    failed: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(output_path):
        if row.get("status") != "failed":
            continue
        model = str(row.get("model", ""))
        task_id = str(row.get("task_id", ""))
        if not model or not task_id:
            continue
        if models and model not in models:
            continue
        failed[model].add(task_id)
    return {model: sorted(task_ids) for model, task_ids in sorted(failed.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/1_intermediate/direct_scoring/direct_scoring_tasks.jsonl")
    parser.add_argument("--output", default="data/1_intermediate/direct_scoring/direct_scoring_results_six_scorers.jsonl")
    parser.add_argument("--summary-dir", default="data/1_intermediate/direct_scoring/run_summaries")
    parser.add_argument("--models", default=None, help="Comma-separated subset of failed models to retry.")
    parser.add_argument("--model-id", action="append", default=None, help="Override model id as MODEL=ID. Repeatable.")
    parser.add_argument("--sleep-seconds", type=float, default=4.0)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-compact", action="store_true")
    args = parser.parse_args()

    model_filter = csv_arg(args.models)
    failed = collect_failed_task_ids(args.output, models=model_filter)
    overrides = parse_model_overrides(args.model_id)
    model_ids = dict(DEFAULT_MODEL_IDS)
    model_ids.update(overrides)
    summary = {
        "output": args.output,
        "failed_models": {model: len(task_ids) for model, task_ids in failed.items()},
        "model_ids": {model: model_ids.get(model) for model in failed},
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if args.dry_run or not failed:
        return

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    for model, task_ids in failed.items():
        if model not in model_ids:
            raise SystemExit(f"No model id configured for failed model: {model}")
        manager = LLMManager(models_to_init=[model])
        if not provider_available(manager, model):
            raise SystemExit(provider_unavailable_message(model))
        execute_direct_scoring(
            tasks_path=args.tasks,
            output_path=args.output,
            summary_path=summary_dir / f"direct_scoring_retry_summary_{model}.json",
            model_name=model,
            model_id=model_ids[model],
            model_caller=build_model_caller(manager, model, max_output_tokens=args.max_output_tokens),
            resume=True,
            task_ids=task_ids,
            sleep_seconds=args.sleep_seconds,
        )

    if not args.no_compact:
        backup = f"{args.output}.pre_compact.{time.strftime('%Y%m%d_%H%M%S')}.bak"
        compact_summary = compact_direct_scoring_results(args.output, backup_path=backup)
        print(json.dumps({"compact": compact_summary, "backup": backup}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
