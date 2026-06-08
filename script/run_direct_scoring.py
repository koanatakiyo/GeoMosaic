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
    "openai": "gpt-5-mini",
    "grok": "grok-4-1-fast",
    "llama": "meta-llama/llama-4-scout",
    "doubao": "doubao-seed-2-0-pro-260215",
    "deepseek": "deepseek-chat",
    "qwen": "qwen3.5-122b-a10b",
}


def provider_available(manager: LLMManager, provider: str) -> bool:
    attr = f"{provider}_available"
    return bool(getattr(manager, attr, False))


def provider_unavailable_message(provider: str) -> str:
    if provider in {"openai", "deepseek", "qwen", "doubao", "llama"}:
        return f"{provider} provider is unavailable. Check API key/config and install the openai package if this is an OpenAI-compatible provider."
    return f"{provider} provider is unavailable. Check API key/config and required client package."


def build_model_caller(manager: LLMManager, provider: str, max_output_tokens: int = 16000):
    async def call_async(prompt: str, model_id: str) -> str:
        if provider == "openai":
            manager.config.setdefault("openai", {})["model"] = {"text": model_id}
            manager.config.setdefault("openai", {})["max_tokens"] = max_output_tokens
            return await manager.call_openai(prompt, model=model_id) or ""
        if provider == "grok":
            manager.grok_model_name = model_id
            return await manager.call_grok(prompt, temperature=0) or ""
        if provider == "llama":
            return await manager.call_llama_chat(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_output_tokens,
                model_name=model_id,
                extra_body={},
            ) or ""
        if provider == "doubao":
            return await manager.call_doubao(prompt, temperature=0, max_tokens=max_output_tokens, model_name=model_id) or ""
        if provider == "deepseek":
            manager.deepseek_model_name = model_id
            return await manager.call_deepseek(prompt) or ""
        if provider == "qwen":
            manager.qwen_model_name = model_id
            return await manager.call_qwen(prompt) or ""
        raise ValueError(f"Unsupported provider: {provider}")

    loop = asyncio.new_event_loop()

    def call(prompt: str, model_id: str) -> str | dict[str, Any]:
        return loop.run_until_complete(call_async(prompt, model_id))

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
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    args = parser.parse_args()

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]
    manager = LLMManager(models_to_init=[args.model])
    if not provider_available(manager, args.model):
        raise SystemExit(provider_unavailable_message(args.model))
    summary = execute_direct_scoring(
        tasks_path=args.tasks,
        output_path=args.output,
        summary_path=args.summary,
        model_name=args.model,
        model_id=model_id,
        model_caller=build_model_caller(manager, args.model, max_output_tokens=args.max_output_tokens),
        resume=not args.no_resume,
        limit=args.limit,
        task_ids=args.task_ids,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
