#!/usr/bin/env python3
"""Run E4 direct scoring for one model over planned document-bundle tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.direct_scoring import execute_direct_scoring  # noqa: E402
from llm_manager import LLMManager  # noqa: E402


DEFAULT_MODEL_IDS = {
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-5-mini",
    "grok": "grok-4-1-fast",
    "llama": "meta-llama/llama-4-scout",
    "doubao": "doubao-seed",
    "deepseek": "deepseek-chat",
    "qwen": "qwen3.5-122b",
}


def build_model_caller(manager: LLMManager, provider: str):
    async def call_async(prompt: str, model_id: str) -> str:
        if provider == "gemini":
            return await manager.call_gemini(prompt, model=model_id) or ""
        if provider == "openai":
            manager.config.setdefault("openai", {})["model"] = {"text": model_id}
            return await manager.call_openai(prompt, model=model_id) or ""
        if provider == "grok":
            manager.grok_model_name = model_id
            return await manager.call_grok(prompt, temperature=0) or ""
        if provider == "llama":
            return await manager.call_llama_chat(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8192,
                model_name=model_id,
            ) or ""
        if provider == "doubao":
            return await manager.call_doubao(prompt, temperature=0, max_tokens=8192, model_name=model_id) or ""
        if provider == "deepseek":
            manager.deepseek_model_name = model_id
            return await manager.call_deepseek(prompt) or ""
        if provider == "qwen":
            manager.qwen_model_name = model_id
            return await manager.call_qwen(prompt) or ""
        raise ValueError(f"Unsupported provider: {provider}")

    def call(prompt: str, model_id: str) -> str | dict[str, Any]:
        return asyncio.run(call_async(prompt, model_id))

    return call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/1_intermediate/direct_scoring/direct_scoring_tasks.jsonl")
    parser.add_argument("--output", default="data/1_intermediate/direct_scoring/direct_scoring_results.jsonl")
    parser.add_argument("--summary", default="data/1_intermediate/direct_scoring/direct_scoring_run_summary.json")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(DEFAULT_MODEL_IDS),
        help="Provider key from llm.yaml to use for scoring.",
    )
    parser.add_argument("--model-id", default=None, help="Provider-specific model id override.")
    parser.add_argument("--task-id", action="append", dest="task_ids", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]
    manager = LLMManager(models_to_init=[args.model])
    summary = execute_direct_scoring(
        tasks_path=args.tasks,
        output_path=args.output,
        summary_path=args.summary,
        model_name=args.model,
        model_id=model_id,
        model_caller=build_model_caller(manager, args.model),
        resume=not args.no_resume,
        limit=args.limit,
        task_ids=args.task_ids,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
