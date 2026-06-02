#!/usr/bin/env python3
"""Run Stage C Gemini/Claude claim-grounding judgments over cached tasks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import request

from geomosaic_hg.claim_grounding import GROUNDING_RESPONSE_SCHEMA, execute_claim_grounding


DEFAULT_LOCATION = "us-central1"
DEFAULT_CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_LLM_CONFIG = Path(os.getenv("GEOMOSAIC_LLM_CONFIG", "APIs/llm.yaml"))


def load_llm_config(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read --llm-config. Install yaml or set ANTHROPIC_API_KEY.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def load_anthropic_api_key(env_name: str = "ANTHROPIC_API_KEY", config_path: str | Path | None = DEFAULT_LLM_CONFIG) -> str:
    env_value = os.getenv(env_name)
    if env_value and env_value.strip():
        return env_value.strip()
    config = load_llm_config(config_path)
    anthropic = config.get("anthropic") if isinstance(config.get("anthropic"), dict) else {}
    api_key = anthropic.get("api_key") or anthropic.get("key")
    return str(api_key).strip() if api_key else ""


class VertexGeminiGroundingCaller:
    """Retrying Vertex AI Gemini wrapper with JSON schema output."""

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
                        response_schema=GROUNDING_RESPONSE_SCHEMA,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                return response.text or ""
            except Exception as exc:  # noqa: BLE001 - provider retry wrapper.
                last_exc = exc
                if attempt >= self.retries:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        assert last_exc is not None
        raise last_exc


class AnthropicClaudeGroundingCaller:
    """Small urllib-based Anthropic Messages API wrapper."""

    def __init__(
        self,
        api_key: str,
        temperature: float,
        max_output_tokens: int,
        retries: int,
        retry_backoff_seconds: float,
        timeout: int,
    ) -> None:
        self.api_key = api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retries = max(0, retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.timeout = timeout

    def __call__(self, prompt: str, model_id: str) -> str:
        payload = {
            "model": model_id,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "tool_choice": {"type": "tool", "name": "ground_claim"},
            "tools": [
                {
                    "name": "ground_claim",
                    "description": "Return a structured claim-grounding judgment for the supplied passages.",
                    "input_schema": GROUNDING_RESPONSE_SCHEMA,
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }
        body = json.dumps(payload).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=body,
                    headers={
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                        "x-api-key": self.api_key,
                    },
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                for block in data.get("content", []):
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and block.get("name") == "ground_claim"
                        and isinstance(block.get("input"), dict)
                    ):
                        return json.dumps(block["input"], ensure_ascii=False)
                return "\n".join(
                    block.get("text", "")
                    for block in data.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            except Exception as exc:  # noqa: BLE001 - provider retry wrapper.
                last_exc = exc
                if attempt >= self.retries:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        assert last_exc is not None
        raise last_exc


def csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/claim_grounding_tasks.jsonl"),
        help="Stage C task plan produced by plan_claim_grounding.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/claim_grounding_judgments.jsonl"),
        help="Per-model judgment JSONL output.",
    )
    parser.add_argument(
        "--adjudication-output",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/claim_grounding_adjudication.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/claim_grounding_run_summary.json"),
    )
    parser.add_argument("--models", default="gemini,claude", help="Comma-separated subset: gemini,claude.")
    parser.add_argument("--gemini-model", default="gemini-2.5-pro")
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT") or "",
        help="Google Cloud project for Vertex AI. Defaults to GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_QUOTA_PROJECT.",
    )
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--api-version", default="v1")
    parser.add_argument("--anthropic-api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG,
        help="Optional local LLM YAML config. Env var takes precedence; default: APIs/llm.yaml.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=15.0)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected tasks, across all requested models.")
    parser.add_argument("--task-id", action="append", default=[], help="Run only this task_id. May be passed multiple times.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="Delete outputs before running. Implies --no-resume.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite:
        args.resume = False
        for path in (args.output, args.adjudication_output, args.summary):
            if path.exists():
                path.unlink()

    requested = csv_arg(args.models)
    callers: dict[str, Any] = {}
    model_ids: dict[str, str] = {}
    if "gemini" in requested:
        if not args.project:
            raise SystemExit("Missing --project or GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_QUOTA_PROJECT for Gemini.")
        callers["gemini"] = VertexGeminiGroundingCaller(
            project=args.project,
            location=args.location,
            api_version=args.api_version,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        model_ids["gemini"] = args.gemini_model
    if "claude" in requested:
        api_key = load_anthropic_api_key(args.anthropic_api_key_env, args.llm_config)
        if not api_key:
            raise SystemExit(f"Missing {args.anthropic_api_key_env} or anthropic.api_key in {args.llm_config} for Claude.")
        callers["claude"] = AnthropicClaudeGroundingCaller(
            api_key=api_key,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            timeout=args.timeout,
        )
        model_ids["claude"] = args.claude_model
    unknown = sorted(set(requested) - {"gemini", "claude"})
    if unknown:
        raise SystemExit(f"Unknown model name(s): {', '.join(unknown)}")
    if not callers:
        raise SystemExit("No models requested.")

    summary = execute_claim_grounding(
        tasks_path=args.tasks,
        output_path=args.output,
        adjudication_path=args.adjudication_output,
        summary_path=args.summary,
        model_callers=callers,
        model_ids=model_ids,
        resume=args.resume,
        limit=args.limit,
        task_ids=args.task_id,
        stop_on_error=args.stop_on_error,
        sleep_seconds=args.sleep_seconds,
        parse_retries=args.parse_retries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
