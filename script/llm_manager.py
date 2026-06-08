#!/usr/bin/env python3
"""
LLM Manager - Standalone LLM interface for knowledge corpus building

This module provides a clean interface for calling LLM APIs without
dependencies on the model comparison testing infrastructure.
"""

import os
import asyncio
import yaml
import time
import logging
import re
import base64
import mimetypes
import json
from pathlib import Path
from typing import Optional, List, Any, Dict
from collections import deque

LOGGER = logging.getLogger(__name__)

_RETRY_DELAY_PATTERNS = (
    re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
    re.compile(r"retrydelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE),
)

VERTEX_PROJECT = "project-c4865349-a336-4706-8ca"
VERTEX_LOCATION = "us-central1"
MODELS = {
    "gemini_flash": "gemini-2.5-flash",
    "gemini_pro": "gemini-2.5-pro",
}


class LLMManager:
    """Standalone LLM manager for knowledge corpus building"""

    def __init__(self, models_to_init: list = None, fallback_api_keys: Optional[Dict[str, List[str]]] = None):
        """Initialize the LLM manager

        Args:
            models_to_init: List of models to initialize ['openai', 'gemini'].
                           If None, initializes all available models.
        """
        self.project_dir = Path(__file__).resolve().parent.parent
        config_override = os.getenv("GEOMOSAIC_LLM_CONFIG")
        config_candidates = [
            Path(config_override).expanduser() if config_override else None,
            self.project_dir / "APIs" / "llm.yaml",
            self.project_dir / "llm.yaml",
            Path(__file__).resolve().parent / "llm.yaml",
            Path(__file__).resolve().parent / "config" / "llm.yaml",
        ]
        self.config_path = next((p for p in config_candidates if p and p.exists()), self.project_dir / "APIs" / "llm.yaml")

        # Load config
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: LLM config not found at {self.config_path}")
            self.config = {}

        # Apply fallback API key overrides per provider when supplied
        if fallback_api_keys:
            for provider, keys in fallback_api_keys.items():
                norm_keys: List[str] = []
                if isinstance(keys, str):
                    norm_keys = [k.strip() for k in keys.split(",") if k.strip()]
                elif isinstance(keys, (list, tuple, set)):
                    norm_keys = [str(k).strip() for k in keys if str(k).strip()]
                if not norm_keys:
                    continue
                self.config.setdefault(provider, {})
                self.config[provider]["fallback_api_keys"] = norm_keys

        # Rate-limit settings
        anthropic_cfg = self.config.get("anthropic", {})
        self.anthropic_call_delay = float(anthropic_cfg.get("rate_limit_delay_sec", 20.0))
        self._anthropic_last_call = 0.0

        # Initialize only requested models
        if models_to_init is None:
            models_to_init = ['openai', 'gemini', 'grok', 'deepseek', 'anthropic', 'qwen', 'doubao', 'ollama', 'vllm']

        # Initialize API clients based on request
        if 'openai' in models_to_init:
            self._init_openai()
        else:
            self.openai_available = False

        if 'gemini' in models_to_init:
            self._init_gemini()
        else:
            self.gemini_available = False

        if 'grok' in models_to_init:
            self._init_grok()
        else:
            self.grok_available = False

        if 'deepseek' in models_to_init:
            self._init_deepseek()
        else:
            self.deepseek_available = False

        if 'anthropic' in models_to_init:
            self._init_anthropic()
        else:
            self.anthropic_available = False

        if 'qwen' in models_to_init:
            self._init_qwen()
        else:
            self.qwen_available = False

        if 'doubao' in models_to_init:
            self._init_doubao()
        else:
            self.doubao_available = False

        if 'llama' in models_to_init:
            self._init_llama()
        else:
            self.llama_available = False

        if 'ollama' in models_to_init:
            self._init_ollama()
        else:
            self.ollama_available = False

        if 'vllm' in models_to_init:
            self._init_vllm()
        else:
            self.vllm_available = False

        # Rate limiting for Gemini API.
        # Defaults are conservative; override via config:
        # gemini:
        #   rate_limit:
        #     rpm_limit: 900
        #     rpd_limit: 9500   # set to 0/null/"unlimited" to disable daily limiting
        gemini_rl = (self.config.get("gemini", {}) or {}).get("rate_limit") or {}

        def _parse_limit(value, default: int) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                if value <= 0:
                    return None
                if value == float("inf"):
                    return None
                return int(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if not normalized:
                    return default
                if normalized in {"none", "null", "inf", "infinite", "unlimited", "no_limit", "no limit"}:
                    return None
                try:
                    parsed = float(normalized)
                except ValueError:
                    return default
                if parsed <= 0 or parsed == float("inf"):
                    return None
                return int(parsed)
            return default

        self.gemini_rpm_limit = _parse_limit(gemini_rl.get("rpm_limit", 900), 900)
        self.gemini_rpd_limit = _parse_limit(gemini_rl.get("rpd_limit", 9500), 9500)

        request_times_maxlen = 1000
        if self.gemini_rpm_limit is not None:
            request_times_maxlen = max(request_times_maxlen, int(self.gemini_rpm_limit) * 2)
        self.gemini_request_times = deque(maxlen=request_times_maxlen)
        self.gemini_daily_requests = 0
        self.gemini_daily_reset_time = time.time() + 86400  # Reset after 24 hours

    async def _apply_rate_limit_delay(self, last_call_attr: str, delay_attr: str) -> None:
        """Ensure at least configured delay between successive calls."""
        delay = max(0.0, getattr(self, delay_attr, 0.0))
        if delay <= 0:
            return

        last_call = getattr(self, last_call_attr, 0.0)
        now = time.time()
        elapsed = now - last_call
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)

    def _mark_rate_limit_timestamp(self, last_call_attr: str) -> None:
        setattr(self, last_call_attr, time.time())

    def _extract_retry_delay_seconds(self, exc: Exception) -> Optional[float]:
        msg = str(exc)
        for pattern in _RETRY_DELAY_PATTERNS:
            match = pattern.search(msg)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None

    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai

            # Get API key from config or environment
            openai_config = self.config.get("openai", {})
            api_key = (
                openai_config.get("api_key") or
                os.getenv("OPENAI_API_KEY")
            )

            if api_key:
                self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                model_cfg = openai_config.get("model", {})
                if isinstance(model_cfg, dict):
                    self.openai_default_model = model_cfg.get("text", "gpt-4o")
                else:
                    self.openai_default_model = model_cfg or "gpt-4o"
                self.openai_available = True
                print("   OpenAI client initialized")
            else:
                self.openai_available = False
                print("   OpenAI API key not found")

        except ImportError:
            self.openai_available = False
            print("   OpenAI library not installed")

    @staticmethod
    def _config_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _resolve_gemini_model(self, model_name: Optional[str] = None) -> str:
        selected = model_name or self.gemini_model_name
        return self.gemini_models.get(selected, selected)

    def _init_gemini(self):
        """Initialize Gemini client with Vertex AI or fallback API keys."""
        try:
            from google import genai
            from google.genai import types
            import logging

            # Suppress verbose Gemini SDK logging
            logging.getLogger("google.genai").setLevel(logging.WARNING)
            logging.getLogger("google").setLevel(logging.WARNING)

            # Store types for later use
            self.genai_types = types
            gemini_config = self.config.get("gemini", {}) or {}
            configured_models = gemini_config.get("models") or {}
            self.gemini_models = dict(MODELS)
            if isinstance(configured_models, dict):
                self.gemini_models.update(configured_models)

            model_config = gemini_config.get("model", "gemini_flash")
            self.gemini_model_name = self._resolve_gemini_model(model_config)
            self.gemini_vertexai = self._config_bool(
                gemini_config.get("vertexai", gemini_config.get("use_vertexai")),
            )

            # Track current API key index. Vertex AI mode does not use API keys.
            self.gemini_api_keys = []
            self.gemini_current_key_index = 0

            if self.gemini_vertexai:
                project = (
                    gemini_config.get("project")
                    or gemini_config.get("vertex_project")
                    or os.getenv("GOOGLE_CLOUD_PROJECT")
                    or os.getenv("GCLOUD_PROJECT")
                    or VERTEX_PROJECT
                )
                location = (
                    gemini_config.get("location")
                    or gemini_config.get("vertex_location")
                    or os.getenv("GOOGLE_CLOUD_LOCATION")
                    or VERTEX_LOCATION
                )
                api_version = gemini_config.get("api_version", "v1")

                self.gemini_client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                    http_options=types.HttpOptions(api_version=api_version),
                )
                self.gemini_available = True
                self.gemini_vertex_project = project
                self.gemini_vertex_location = location
                print(
                    "   Gemini Vertex AI client initialized "
                    f"({project}/{location}, model={self.gemini_model_name})"
                )
                return

            # Get primary API key from config or environment
            primary_api_key = (
                gemini_config.get("api_key") or
                os.getenv("GEMINI_API_KEY") or
                os.getenv("GOOGLE_API_KEY")
            )

            # Get fallback API keys from config (support comma-separated string)
            fallback_keys = gemini_config.get("fallback_api_keys", [])
            if isinstance(fallback_keys, str):
                fallback_keys = [k.strip() for k in fallback_keys.split(",") if k.strip()]
            
            # Build list of all API keys (primary + fallbacks)
            if primary_api_key:
                self.gemini_api_keys.append(primary_api_key)
            self.gemini_api_keys.extend(fallback_keys)

            if self.gemini_api_keys:
                # Set initial API key in environment
                os.environ["GOOGLE_API_KEY"] = self.gemini_api_keys[0]

                # Create client using new API
                self.gemini_client = genai.Client()

                self.gemini_available = True
                total_keys = len(self.gemini_api_keys)
                print(f"   Gemini client initialized with {total_keys} API key(s) (1 primary + {total_keys-1} fallback)")
            else:
                self.gemini_available = False
                print("   Gemini API key not found")

        except ImportError as e:
            self.gemini_available = False
            print("   Gemini library not installed - install with: pip install google-genai")
        except Exception as e:
            self.gemini_available = False
            print(f"   Gemini initialization failed: {e}")

    def switch_gemini_key(self) -> bool:
        """Rotate to next Gemini API key. Returns True if switched."""
        try:
            from google import genai
        except Exception:
            return False
        if not getattr(self, "gemini_api_keys", None):
            return False
        if len(self.gemini_api_keys) < 2:
            return False
        self.gemini_current_key_index = (self.gemini_current_key_index + 1) % len(self.gemini_api_keys)
        new_key = self.gemini_api_keys[self.gemini_current_key_index]
        os.environ["GOOGLE_API_KEY"] = new_key
        try:
            self.gemini_client = genai.Client()
            print(f"   Gemini key rotated to index {self.gemini_current_key_index}")
            return True
        except Exception as e:
            print(f"   Gemini key rotation failed: {e}")
            return False

    def _init_grok(self):
        """Initialize Grok client"""
        try:
            import httpx

            # Get API key from config or environment
            api_key = (
                self.config.get("grok", {}).get("api_key") or
                os.getenv("GROK_API_KEY") or
                os.getenv("XAI_API_KEY")
            )

            if api_key:
                # Store API key and base URL for later use
                self.grok_api_key = api_key
                self.grok_base_url = self.config.get("grok", {}).get("base_url", "https://api.x.ai/v1")
                self.grok_model_name = self.config.get("grok", {}).get("model", "grok-4-latest")
                
                # Create async HTTP client
                self.grok_client = httpx.AsyncClient()
                
                self.grok_available = True
                print("   Grok client initialized")
            else:
                self.grok_available = False
                print("   Grok API key not found")

        except ImportError:
            self.grok_available = False
            print("   httpx library not installed - install with: pip install httpx")
        except Exception as e:
            self.grok_available = False
            print(f"   Grok initialization failed: {e}")

    def _init_deepseek(self):
        """Initialize DeepSeek client (OpenAI-compatible API)

        Supports both the official DeepSeek endpoint and Alibaba DashScope's
        compatible-mode endpoint so a single DashScope API key can be reused
        for DeepSeek reasoning models (e.g., deepseek-v3.1) or Qwen family models.
        """
        try:
            from openai import AsyncOpenAI

            deepseek_config = self.config.get("deepseek", {})

            # Determine which API key / endpoint to use.
            api_key = (
                deepseek_config.get("api_key")
                or os.getenv("DEEPSEEK_API_KEY")
            )
            use_dashscope = False

            if not api_key:
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if api_key:
                    use_dashscope = True

            provider_hint = (deepseek_config.get("provider") or "").lower()
            if provider_hint in {"dashscope", "alibaba"}:
                use_dashscope = True

            if deepseek_config.get("use_dashscope"):
                use_dashscope = True

            if not api_key:
                self.deepseek_available = False
                print("   DeepSeek API key not found")
                return

            if use_dashscope and not deepseek_config.get("base_url"):
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            else:
                base_url = deepseek_config.get("base_url", "https://api.deepseek.com")

            base_url = base_url.rstrip("/")
            self.deepseek_api_key = api_key
            self.deepseek_base_url = base_url

            default_model = "deepseek-v3.1" if use_dashscope else "deepseek-chat"
            self.deepseek_model_name = deepseek_config.get("model", default_model)

            self.deepseek_extra_body = None
            self.deepseek_client = AsyncOpenAI(api_key=api_key, base_url=self.deepseek_base_url)
            self.deepseek_available = True

            provider_label = "DashScope" if use_dashscope else "DeepSeek"
            print(f"   DeepSeek client initialized via {provider_label} (model: {self.deepseek_model_name})")

        except ImportError:
            self.deepseek_available = False
            print("   openai library not installed - install with: pip install openai")
        except Exception as e:
            self.deepseek_available = False
            print(f"   DeepSeek initialization failed: {e}")

    def _init_anthropic(self):
        """Initialize Anthropic (Claude) client"""
        try:
            import httpx

            api_key = (
                self.config.get("anthropic", {}).get("api_key") or
                os.getenv("ANTHROPIC_API_KEY")
            )

            if api_key:
                self.anthropic_api_key = api_key
                self.anthropic_base_url = self.config.get("anthropic", {}).get("base_url", "https://api.anthropic.com")
                self.anthropic_model_name = self.config.get("anthropic", {}).get("model", "claude-3-5-sonnet-latest")
                self.anthropic_client = httpx.AsyncClient()
                self.anthropic_available = True
                print("   Anthropic client initialized")
            else:
                self.anthropic_available = False
                print("   Anthropic API key not found")

        except ImportError:
            self.anthropic_available = False
            print("   httpx library not installed - install with: pip install httpx")
        except Exception as e:
            self.anthropic_available = False
            print(f"   Anthropic initialization failed: {e}")

    # --------------------------- Batch helpers (Anthropic) --------------------------- #
    async def anthropic_create_batch(self, requests: list) -> str:
        """
        Create an Anthropic messages batch. Each item should include:
        {
          "custom_id": "...",
          "params": { "model": "...", "max_tokens": ..., "messages": [...] }
        }
        Returns batch_id.
        """
        if not getattr(self, "anthropic_available", False):
            raise RuntimeError("Anthropic not available for batch")
        import httpx
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {"requests": requests}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.anthropic_base_url}/v1/messages/batches",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["id"]

    async def anthropic_get_batch(self, batch_id: str) -> dict:
        if not getattr(self, "anthropic_available", False):
            raise RuntimeError("Anthropic not available for batch")
        import httpx
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{self.anthropic_base_url}/v1/messages/batches/{batch_id}",
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def anthropic_download_batch_output(self, batch_id: str) -> list:
        """
        Download completed batch results. Returns list of result items.
        Supports both older Anthropic responses (embedded `results` + `status`)
        and newer responses (`processing_status` + `results_url` returning JSONL).
        """
        data = await self.anthropic_get_batch(batch_id)

        # Older API: {"status": "completed", "results": [...]}
        if isinstance(data, dict) and data.get("results") is not None:
            status = data.get("status") or data.get("processing_status")
            if status not in {"completed", "ended"}:
                raise RuntimeError(f"Batch {batch_id} not completed yet (status={status!r})")
            return data.get("results", []) or []

        # Newer API: {"processing_status": "ended", "results_url": ".../results"}
        processing_status = data.get("processing_status") or data.get("status")
        if processing_status not in {"ended", "completed"}:
            raise RuntimeError(f"Batch {batch_id} not completed yet (processing_status={processing_status!r})")

        results_url = data.get("results_url")
        if not results_url:
            # Be explicit: caller can always use anthropic_get_batch for metadata.
            raise RuntimeError(f"Batch {batch_id} has no results_url; cannot download results")

        import httpx
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(results_url, headers=headers)
            resp.raise_for_status()
            raw = resp.text

        # Results endpoint is JSONL (one JSON object per line).
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items

    def _init_qwen(self):
        """Initialize Qwen (DashScope) client via OpenAI-compatible HTTP API"""
        try:
            from openai import AsyncOpenAI

            api_key = (
                self.config.get("qwen", {}).get("api_key") or
                os.getenv("DASHSCOPE_API_KEY")
            )

            if api_key:
                base_url = self.config.get("qwen", {}).get(
                    "base_url",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ).rstrip("/")
                self.qwen_api_key = api_key
                self.qwen_base_url = base_url
                self.qwen_model_name = self.config.get("qwen", {}).get(
                    "model",
                    "qwen3.5-122b-a10b"
                )
                self.qwen_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                self.qwen_available = True
                print("   Qwen client initialized (DashScope compatible API)")
            else:
                self.qwen_available = False
                print("   Qwen API key not found (set qwen.api_key or DASHSCOPE_API_KEY)")

        except ImportError:
            self.qwen_available = False
            print("   openai library not installed - install with: pip install openai")
        except Exception as e:
            self.qwen_available = False
            print(f"   Qwen initialization failed: {e}")

    def _init_doubao(self):
        """Initialize Doubao client via OpenAI-compatible Responses API (Volcengine Ark)."""
        try:
            from openai import AsyncOpenAI

            api_key = (
                self.config.get("doubao", {}).get("api_key") or
                os.getenv("ARK_API_KEY")
            )

            if api_key:
                base_url = self.config.get("doubao", {}).get(
                    "base_url",
                    "https://ark.cn-beijing.volces.com/api/v3"
                ).rstrip("/")
                self.doubao_api_key = api_key
                self.doubao_base_url = base_url
                self.doubao_model_name = self.config.get("doubao", {}).get(
                    "model",
                    "doubao-seed-2-0-pro-260215"
                )
                self.doubao_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                self.doubao_available = True
                print("   Doubao (Volcengine Ark) client initialized")
            else:
                self.doubao_available = False
                print("   Doubao API key not found (set doubao.api_key or ARK_API_KEY)")

        except ImportError:
            self.doubao_available = False
            print("   openai library not installed - install with: pip install openai")
        except Exception as e:
            self.doubao_available = False
            print(f"   Doubao initialization failed: {e}")

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        if isinstance(response, dict):
            output_text = response.get("output_text")
            if isinstance(output_text, str) and output_text:
                return output_text
            output = response.get("output", [])
        else:
            output = getattr(response, "output", [])

        text_parts: list[str] = []
        for item in output or []:
            content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
            for block in content or []:
                if isinstance(block, dict):
                    value = block.get("text") or block.get("content")
                else:
                    value = getattr(block, "text", None) or getattr(block, "content", None)
                if isinstance(value, str):
                    text_parts.append(value)
        return "".join(text_parts)

    async def call_doubao(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Optional[str]:
        """Call Doubao (Volcengine Ark) via the Responses API."""
        if not getattr(self, "doubao_available", False):
            raise ValueError("Doubao not available")

        generation_config = self.config.get("doubao", {}).get("generation", {})
        params = {
            "model": model_name or self.doubao_model_name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                        }
                    ],
                }
            ],
            "temperature": temperature if temperature is not None else generation_config.get("temperature", 0.7),
        }
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        try:
            response = await self.doubao_client.responses.create(**params)
            content = self._extract_response_text(response)
            if not content or not content.strip():
                raise ValueError(f"Doubao returned empty output (model={params['model']})")
            return content.strip()
        except Exception as e:
            print(f"Doubao API error: {e!r}")
            raise

    def _init_ollama(self):
        """Initialize Ollama client (local OpenAI-compatible API)"""
        try:
            import httpx

            # Ollama runs locally, no API key needed
            # Default base URL for Ollama
            self.ollama_base_url = self.config.get("ollama", {}).get("base_url", "http://localhost:11434/v1")
            self.ollama_model_name = self.config.get("ollama", {}).get("model", "deepseek-r1")

            # Create async HTTP client with longer timeout for local models
            self.ollama_client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout for reasoning

            self.ollama_available = True
            print(f"   Ollama client initialized (model: {self.ollama_model_name})")

        except ImportError:
            self.ollama_available = False
            print("   httpx library not installed - install with: pip install httpx")
        except Exception as e:
            self.ollama_available = False
            print(f"   Ollama initialization failed: {e}")

    def _init_vllm(self):
        """Initialize vLLM client (OpenAI-compatible server)"""
        try:
            import httpx

            config = self.config.get("vllm", {})
            base_url = config.get("base_url") or os.getenv("VLLM_BASE_URL") or "http://localhost:8000/v1"
            base_url = base_url.rstrip("/")
            self.vllm_base_url = base_url
            self.vllm_model_name = config.get("model") or os.getenv("VLLM_MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
            self.vllm_api_key = config.get("api_key") or os.getenv("VLLM_API_KEY")
            timeout = config.get("timeout", 180.0)

            self.vllm_client = httpx.AsyncClient(timeout=timeout)
            self.vllm_available = True
            print(f"   vLLM client initialized (base_url: {self.vllm_base_url})")

        except ImportError:
            self.vllm_available = False
            print("   httpx library not installed - install with: pip install httpx")
        except Exception as e:
            self.vllm_available = False
            print(f"   vLLM initialization failed: {e}")

    def _init_llama(self):
        """Initialize Llama (OpenRouter) client with fallback API keys."""
        try:
            from openai import AsyncOpenAI

            cfg = self.config.get("llama", {}) or {}
            primary_key = cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY")
            fallback_keys = cfg.get("fallback_api_keys", [])
            if isinstance(fallback_keys, str):
                fallback_keys = [k.strip() for k in fallback_keys.split(",") if k.strip()]

            self.llama_api_keys = []
            if primary_key:
                self.llama_api_keys.append(primary_key)
            self.llama_api_keys.extend(fallback_keys or [])
            self.llama_current_key_index = 0

            if not self.llama_api_keys:
                self.llama_available = False
                print("   Llama4 API key not found (set llama.api_key or OPENROUTER_API_KEY)")
                return

            base_url = cfg.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
            self.llama_base_url = base_url
            self.llama_model_name = cfg.get("model", "meta-llama/llama-4-maverick")

            active_key = self.llama_api_keys[self.llama_current_key_index]
            self.llama_client = AsyncOpenAI(api_key=active_key, base_url=base_url)
            self.llama_available = True
            total_keys = len(self.llama_api_keys)
            print(f"   Llama4 (OpenRouter) client initialized with {total_keys} key(s)")

        except ImportError:
            self.llama_available = False
            print("   openai library not installed - install with: pip install openai")
        except Exception as e:
            self.llama_available = False
            print(f"   Llama4 initialization failed: {e}")

    def _switch_llama_api_key(self) -> bool:
        """Rotate to the next Llama/OpenRouter API key, rebuild client, return True if switched."""
        try:
            from openai import AsyncOpenAI
        except Exception:
            return False

        if not getattr(self, "llama_api_keys", None) or len(self.llama_api_keys) < 2:
            return False

        self.llama_current_key_index = (self.llama_current_key_index + 1) % len(self.llama_api_keys)
        new_key = self.llama_api_keys[self.llama_current_key_index]
        try:
            self.llama_client = AsyncOpenAI(api_key=new_key, base_url=self.llama_base_url)
            print(f"   Llama API key rotated to index {self.llama_current_key_index + 1}/{len(self.llama_api_keys)}")
            return True
        except Exception as e:
            print(f"   Llama key rotation failed: {e}")
            return False

    async def call_llama_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.8,
        max_tokens: int = 16000,
        model_name: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call Llama via OpenRouter with fallback key rotation on 429s."""
        if not getattr(self, "llama_available", False):
            raise ValueError("Llama not available")

        attempts = max(2, (len(self.llama_api_keys) if getattr(self, "llama_api_keys", None) else 1) * 2)
        base_backoff = 2.0  # seconds; exponential backoff like JS example
        last_err: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                response = await self.llama_client.chat.completions.create(
                    model=model_name or self.llama_model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers or self.config.get("llama", {}).get("extra_headers", {}),
                    extra_body=extra_body if extra_body is not None else self.config.get("llama", {}).get("extra_body", {}),
                )
                content = response.choices[0].message.content if response.choices else None
                if content is None:
                    # Log warning for debugging
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        "Llama API returned None content (model=%s, max_tokens=%d, response_id=%s)",
                        model_name or self.llama_model_name,
                        max_tokens,
                        getattr(response, "id", "unknown"),
                    )
                return content or ""
            except Exception as e:
                last_err = e
                is_rate_limit = (
                    getattr(e, "status_code", None) == 429
                    or "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                )
                if is_rate_limit:
                    # Prefer header-derived reset; otherwise exponential backoff
                    wait_s = self._compute_rate_limit_wait(e, default_wait=base_backoff * (2 ** attempt), max_wait=60.0)
                    switched = self._switch_llama_api_key()
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    if switched:
                        continue
                    # no fallback key; retry same key after backoff
                    continue
                break

        raise last_err or RuntimeError("Llama call failed")

    @staticmethod
    def _compute_rate_limit_wait(err: Exception, default_wait: float = 5.0, max_wait: float = 60.0) -> float:
        """Best-effort wait time from rate-limit headers; fallback to default."""
        try:
            resp = getattr(err, "response", None)
            headers = getattr(resp, "headers", {}) if resp else {}
            reset_raw = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
            if reset_raw:
                try:
                    reset_val = float(reset_raw)
                    # Heuristic: if value is milliseconds since epoch and not absurd
                    if reset_val > 10_000_000:
                        reset_val = reset_val / 1000.0
                    wait = reset_val - time.time()
                    if wait > 0:
                        return min(wait, max_wait)
                except Exception:
                    pass
        except Exception:
            pass
        return default_wait

    # --------------------------- Batch helpers (OpenAI) --------------------------- #
    async def openai_create_batch(
        self,
        requests: list,
        *,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
    ) -> str:
        """
        Create an OpenAI batch job. Returns batch_id.
        Each request item should already include: custom_id, method, url, body.
        """
        if not self.openai_available:
            raise RuntimeError("OpenAI not available for batch")

        import httpx
        api_key = self.config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = (self.config.get("openai", {}).get("base_url") or "https://api.openai.com/v1").rstrip("/")

        # 1) Upload input file
        jsonl_lines = "\n".join(json.dumps(r) for r in requests)
        files = {"file": ("batchinput.jsonl", jsonl_lines.encode("utf-8"))}
        data = {"purpose": "batch"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            upload = await client.post(
                f"{base_url}/files",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
            )
            upload.raise_for_status()
            input_file_id = upload.json()["id"]

            # 2) Create batch
            batch_payload = {
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
            }
            batch = await client.post(
                f"{base_url}/batches",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=batch_payload,
            )
            batch.raise_for_status()
            return batch.json()["id"]

    async def openai_get_batch(self, batch_id: str) -> dict:
        if not self.openai_available:
            raise RuntimeError("OpenAI not available for batch")
        import httpx
        api_key = self.config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = (self.config.get("openai", {}).get("base_url") or "https://api.openai.com/v1").rstrip("/")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{base_url}/batches/{batch_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return resp.json()

    async def openai_download_batch_output(self, file_id: str) -> str:
        """Returns raw text content of the batch output file."""
        if not self.openai_available:
            raise RuntimeError("OpenAI not available for batch")
        import httpx
        api_key = self.config.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = (self.config.get("openai", {}).get("base_url") or "https://api.openai.com/v1").rstrip("/")
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(
                f"{base_url}/files/{file_id}/content",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return resp.text

    async def call_openai(self, prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
        """Call OpenAI API"""
        if not self.openai_available:
            raise ValueError("OpenAI not available")

        try:
            # Get model from config with proper nesting
            openai_config = self.config.get("openai", {})
            model_name = (
                openai_config.get("model", {}).get("text") if isinstance(openai_config.get("model"), dict)
                else openai_config.get("model", model)
            )

            # Ensure prompt is a valid string and not too long
            if not isinstance(prompt, str):
                prompt = str(prompt)

            # Clean prompt to avoid JSON parsing issues
            prompt = prompt.replace('\x00', '').strip()

            # Truncate extremely long prompts to avoid API issues
            if len(prompt) > 50000:
                prompt = prompt[:50000] + "... [truncated]"

            # Get config values with proper defaults and nesting
            # Increased default max_tokens from 1500 to 4000 to avoid truncating JSON responses
            max_tokens = openai_config.get("max_tokens", 6000)
            temperature = openai_config.get("generation", {}).get("temperature", 0.7)

            # Ensure numeric values are proper types
            if not isinstance(max_tokens, int):
                max_tokens = 6000  # Increased from 1500
            if not isinstance(temperature, (int, float)):
                temperature = 0.7

            # Build parameters based on model family
            is_gpt5 = model_name and "gpt-5" in model_name.lower()

            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}]
            }

            if is_gpt5:
                # GPT-5: max_completion_tokens ONLY (max_tokens not supported)
                params["max_completion_tokens"] = max_tokens
                # Temperature NOT configurable for GPT-5 (fixed at 1, remove if you pass 0.1/0.7)
                # top_p and frequency_penalty often unsupported (omit)
            else:
                # Older models (GPT-4, GPT-3.5): max_tokens, configurable temperature
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            response = await self.openai_client.chat.completions.create(**params)

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            print(f"Model: {model_name if 'model_name' in locals() else 'unknown'}")
            print(f"Prompt length: {len(prompt) if 'prompt' in locals() else 'unknown'}")
            raise

    async def call_openai_with_web_search(self, query: str) -> Optional[str]:
        """Call OpenAI API with web_search tool enabled (for GPT-5 models)"""
        if not self.openai_available:
            raise ValueError("OpenAI not available")

        try:
            openai_config = self.config.get("openai", {})
            model_name = (
                openai_config.get("model", {}).get("text") if isinstance(openai_config.get("model"), dict)
                else openai_config.get("model", "gpt-5")
            )

            # Check if client supports responses API (GPT-5)
            if not hasattr(self.openai_client, 'responses'):
                print("Warning: OpenAI client doesn't support responses API. Using standard chat completion.")
                return await self.call_openai(query)

            # Use new responses API with web_search tool
            response = await self.openai_client.responses.create(
                model=model_name,
                tools=[
                    {"type": "web_search"},
                ],
                input=query,
            )

            # Extract output text from response
            if hasattr(response, 'output_text'):
                return response.output_text
            elif hasattr(response, 'output') and isinstance(response.output, str):
                return response.output
            else:
                print(f"Unexpected response format: {response}")
                return None

        except AttributeError as e:
            print(f"OpenAI responses API not available: {e}")
            print("Falling back to standard chat completion...")
            return await self.call_openai(query)
        except Exception as e:
            print(f"OpenAI web search error: {e}")
            raise

    async def call_openai_vision(
        self,
        prompt: str,
        image_files: Optional[list] = None,
        model: str = "gpt-4o-mini"
    ) -> Optional[str]:
        """Call OpenAI API with vision support (gpt-4o, gpt-4o-mini, gpt-4-turbo)

        Args:
            prompt: Text prompt
            image_files: List of local image file paths
            model: Model to use (must support vision)
        """
        if not self.openai_available:
            raise ValueError("OpenAI not available")

        import base64
        import mimetypes

        try:
            # Get model from config
            openai_config = self.config.get("openai", {})
            model_name = (
                openai_config.get("model", {}).get("text") if isinstance(openai_config.get("model"), dict)
                else openai_config.get("model", model)
            )

            # Build content with text and images
            content_blocks = []

            # Add text prompt
            if prompt:
                content_blocks.append({
                    "type": "text",
                    "text": prompt
                })

            # Add images as base64
            for file_path in (image_files or []):
                try:
                    with open(file_path, 'rb') as f:
                        data_b64 = base64.b64encode(f.read()).decode('utf-8')
                    media_type, _ = mimetypes.guess_type(file_path)
                    if not media_type:
                        media_type = "image/png"

                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data_b64}"
                        }
                    })
                except Exception as e:
                    print(f"Failed to read image file {file_path}: {e}")

            # Get config values
            max_tokens = openai_config.get("max_tokens", 6000)
            temperature = openai_config.get("generation", {}).get("temperature", 0.7)

            # Build parameters
            is_gpt5 = model_name and "gpt-5" in model_name.lower()

            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": content_blocks}]
            }

            if is_gpt5:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            response = await self.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI Vision API error: {e}")
            print(f"Model: {model_name if 'model_name' in locals() else 'unknown'}")
            raise

    def _switch_gemini_api_key(self):
        """Switch to next available Gemini API key"""
        if len(self.gemini_api_keys) <= 1:
            return False  # No fallback available
        
        # Move to next key
        self.gemini_current_key_index = (self.gemini_current_key_index + 1) % len(self.gemini_api_keys)
        next_key = self.gemini_api_keys[self.gemini_current_key_index]
        
        # Update environment and recreate client
        os.environ["GOOGLE_API_KEY"] = next_key
        from google import genai
        self.gemini_client = genai.Client()
        
        # Reset rate limiting for new key
        self.gemini_daily_requests = 0
        self.gemini_daily_reset_time = time.time() + 86400
        self.gemini_request_times.clear()
        
        key_num = self.gemini_current_key_index + 1
        total_keys = len(self.gemini_api_keys)
        print(f"🔄 Switched to Gemini API key {key_num}/{total_keys}")
        return True

    async def _rate_limit_gemini(self):
        """Apply rate limiting for Gemini API with automatic fallback"""
        current_time = time.time()

        # Reset daily counter if needed
        if current_time >= self.gemini_daily_reset_time:
            self.gemini_daily_requests = 0
            self.gemini_daily_reset_time = current_time + 86400

        # Check daily limit (if configured)
        if self.gemini_rpd_limit is not None and self.gemini_daily_requests >= self.gemini_rpd_limit:
            # Try to switch to fallback key
            if self._switch_gemini_api_key():
                print(f"✅ Switched to fallback API key. Continuing...")
                return  # Continue with new key
            else:
                # No fallback available, must wait
                wait_time = self.gemini_daily_reset_time - current_time
                print(f"⚠️  Daily limit reached ({self.gemini_rpd_limit} requests) on all API keys. Waiting {wait_time/3600:.1f} hours...")
                await asyncio.sleep(wait_time)
                self.gemini_daily_requests = 0
                self.gemini_daily_reset_time = time.time() + 86400

        # Check RPM limit (requests in last 60 seconds)
        one_minute_ago = current_time - 60
        recent_requests = [t for t in self.gemini_request_times if t > one_minute_ago]

        if self.gemini_rpm_limit is not None and len(recent_requests) >= self.gemini_rpm_limit:
            # Calculate wait time to stay under RPM limit
            oldest_recent = min(recent_requests)
            wait_time = 60 - (current_time - oldest_recent) + 0.1  # Small buffer
            if wait_time > 0:
                print(f"⚠️  RPM limit approaching. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

        # Record this request
        self.gemini_request_times.append(time.time())
        self.gemini_daily_requests += 1

    async def rate_limit_gemini(self):
        """Public wrapper for Gemini rate limiting."""
        await self._rate_limit_gemini()

    def _build_gemini_safety_settings(self) -> Optional[List[Any]]:
        """
        Build safety settings for Gemini requests using config overrides.

        Returns a list of google.genai.types.SafetySetting entries or None if the
        types module is unavailable (which shouldn't happen when Gemini is enabled).
        """
        types = getattr(self, "genai_types", None)
        if types is None:
            return None

        # Default thresholds keep high-severity content blocked but allow benign items.
        default_categories = {
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
        }

        safety_cfg = (
            self.config.get("gemini", {}).get("safety_settings", {})
            if isinstance(self.config, dict)
            else {}
        )
        override_default = safety_cfg.get("default_threshold")
        category_overrides = safety_cfg.get("categories", {})

        def _resolve(enum_cls, value, label):
            if value is None:
                return None
            if isinstance(value, str):
                resolved = getattr(enum_cls, value, None)
                if resolved is None:
                    logging.warning(
                        "Gemini safety setting ignored: %s '%s' not supported",
                        label,
                        value,
                    )
                return resolved
            return value

        settings: List[Any] = []
        merged = {**default_categories, **category_overrides}
        for category, threshold in merged.items():
            effective_threshold = threshold or override_default
            if not effective_threshold:
                continue
            resolved_category = _resolve(
                types.HarmCategory, category, "category"
            )
            resolved_threshold = _resolve(
                types.HarmBlockThreshold, effective_threshold, "threshold"
            )
            if not resolved_category or not resolved_threshold:
                continue
            settings.append(
                types.SafetySetting(
                    category=resolved_category,
                    threshold=resolved_threshold,
                )
            )

        return settings or None

    # async def call_gemini_with_web_search(self, prompt: str) -> Optional[str]:
    #     """Call Gemini API with Google Search tool enabled"""
    #     if not self.gemini_available:
    #         raise ValueError("Gemini not available")

    #     try:
    #         # Apply rate limiting
    #         await self._rate_limit_gemini()

    #         # Configure generation with google_search tool
    #         generation_config = self.config.get("gemini", {}).get("generation", {})
    #         temperature = generation_config.get("temperature", 0.1)

    #         # Create config with google_search tool
    #         config = self.genai_types.GenerateContentConfig(
    #             max_output_tokens=8192,  # Increased to max for Gemini 2.0 Flash to avoid truncation
    #             temperature=temperature,
    #             top_k=2,
    #             top_p=0.95,
    #             response_modalities=['TEXT'],
    #             tools=[
    #                 {'google_search': {}},
    #             ],
    #         )

    #         # Call the API with google_search enabled
    #         response = await asyncio.to_thread(
    #             self.gemini_client.models.generate_content,
    #             model=self.gemini_model_name,
    #             contents=prompt,
    #             config=config
    #         )

    #         # Extract text from response
    #         if hasattr(response, 'text'):
    #             return response.text
    #         elif hasattr(response, 'candidates') and response.candidates:
    #             candidate = response.candidates[0]
    #             if hasattr(candidate, 'content') and candidate.content:
    #                 if hasattr(candidate.content, 'parts') and candidate.content.parts:
    #                     return candidate.content.parts[0].text

    #         return ""

    #     except Exception as e:
    #         print(f"Gemini web search error: {e}")
    #         print("Falling back to standard Gemini call...")
    #         return await self.call_gemini(prompt)

    async def call_gemini_with_web_search(self, prompt: str) -> Optional[str]:
        """Stable compatibility fallback for callers that request Gemini web search."""
        return await self.call_gemini(prompt)

    async def call_gemini(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Call Gemini API using new google.genai client with types.GenerateContentConfig"""
        if not self.gemini_available:
            raise ValueError("Gemini not available")

        # Apply rate limiting
        await self._rate_limit_gemini()

        # Configure generation parameters from config
        generation_config = self.config.get("gemini", {}).get("generation", {})
        temperature = generation_config.get("temperature", 0.1)

        # Create config using the new types.GenerateContentConfig
        config = self.genai_types.GenerateContentConfig(
            max_output_tokens=8192,  # Increased to max for Gemini 2.0 Flash to avoid truncation
            temperature=temperature,
            top_k=2,
            top_p=0.95,
        )
        model_name = self._resolve_gemini_model(model)

        keys = getattr(self, "gemini_api_keys", []) or []
        max_attempts = max(1, len(keys))

        for attempt in range(max_attempts):
            try:
                # Call the new API with the correct format
                response = await asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=model_name,
                    contents=prompt,  # Can pass string directly
                    config=config
                )
            except Exception as e:
                msg = str(e).lower()
                retriable = (
                    "429" in msg
                    or "too many requests" in msg
                    or "quota" in msg
                    or "resource_exhausted" in msg
                    or "503" in msg
                    or "service unavailable" in msg
                )
                retry_delay = self._extract_retry_delay_seconds(e) if retriable else None
                if retry_delay and (attempt + 1) < max_attempts:
                    LOGGER.warning(
                        "Gemini requested retry after %.2fs; waiting before retry.",
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                if retriable and (attempt + 1) < max_attempts and self.switch_gemini_key():
                    LOGGER.warning(
                        "Gemini %s; rotated to fallback key (attempt %s/%s) and retrying.",
                        msg,
                        attempt + 2,
                        max_attempts,
                    )
                    continue
                print(f"Gemini API error: {e}")
                raise

            # Extract text from response
            if hasattr(response, 'text'):
                text = response.text
                if text:
                    return text

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        part = candidate.content.parts[0]
                        text = getattr(part, 'text', None)
                        if text:
                            return text

            # Gather debugging info when Gemini returns no usable text
            debug_bits = []

            prompt_feedback = getattr(response, "prompt_feedback", None)
            if prompt_feedback:
                block_reason = getattr(prompt_feedback, "block_reason", None)
                if block_reason:
                    debug_bits.append(f"prompt_block={getattr(block_reason, 'name', block_reason)}")
                safety_ratings = getattr(prompt_feedback, "safety_ratings", None)
                if safety_ratings:
                    ratings = ", ".join(
                        f"{getattr(r.category, 'name', r.category)}:{getattr(r.threshold, 'name', r.threshold)}"
                        for r in safety_ratings
                    )
                    debug_bits.append(f"prompt_safety=[{ratings}]")

            candidates = getattr(response, "candidates", None) or []
            for idx, cand in enumerate(candidates):
                finish_reason = getattr(cand, "finish_reason", None)
                if finish_reason:
                    debug_bits.append(f"cand{idx}_finish={getattr(finish_reason, 'name', finish_reason)}")
                safety_ratings = getattr(cand, "safety_ratings", None)
                if safety_ratings:
                    ratings = ", ".join(
                        f"{getattr(r.category, 'name', r.category)}:{getattr(r.probability, 'name', r.probability)}"
                        for r in safety_ratings
                    )
                    debug_bits.append(f"cand{idx}_safety=[{ratings}]")

            usage = getattr(response, "usage_metadata", None)
            if usage:
                tokens_in = getattr(usage, "prompt_token_count", None)
                tokens_out = getattr(usage, "candidates_token_count", None)
                if tokens_in is not None:
                    debug_bits.append(f"prompt_tokens={tokens_in}")
                if tokens_out is not None:
                    debug_bits.append(f"candidate_tokens={tokens_out}")

            debug_msg = " | ".join(debug_bits) if debug_bits else "no response metadata"
            logging.warning(f"Gemini returned no text. Metadata: {debug_msg}")
            raise ValueError(f"Gemini returned no text (metadata: {debug_msg})")

    async def call_gemini_vision(
        self,
        prompt: str,
        image_files: Optional[list] = None,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """Call Gemini API with vision support

        Args:
            prompt: Text prompt
            image_files: List of local image file paths
        """
        if not self.gemini_available:
            raise ValueError("Gemini not available")

        from PIL import Image

        # Apply rate limiting
        await self._rate_limit_gemini()

        # Configure generation parameters
        generation_config = self.config.get("gemini", {}).get("generation", {})
        temperature = generation_config.get("temperature", 0.1)

        config = self.genai_types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=temperature,
            top_k=2,
            top_p=0.95,
            safety_settings=self._build_gemini_safety_settings(),
        )
        model_name = self._resolve_gemini_model(model)

        # Build content with text and images
        content_parts = []

        if prompt:
            content_parts.append(prompt)

        for file_path in (image_files or []):
            try:
                img = Image.open(file_path)
                content_parts.append(img)
            except Exception as e:
                print(f"Failed to read image file {file_path}: {e}")

        keys = getattr(self, "gemini_api_keys", []) or []
        max_attempts = max(1, len(keys))

        for attempt in range(max_attempts):
            try:
                # Call Gemini with multimodal content
                response = await asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=model_name,
                    contents=content_parts,
                    config=config
                )
            except Exception as e:
                msg = str(e).lower()
                retriable = (
                    "429" in msg
                    or "too many requests" in msg
                    or "quota" in msg
                    or "resource_exhausted" in msg
                    or "503" in msg
                    or "service unavailable" in msg
                )
                retry_delay = self._extract_retry_delay_seconds(e) if retriable else None
                if retry_delay and (attempt + 1) < max_attempts:
                    LOGGER.warning(
                        "Gemini requested retry after %.2fs; waiting before retry.",
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                if retriable and (attempt + 1) < max_attempts and self.switch_gemini_key():
                    LOGGER.warning(
                        "Gemini %s; rotated to fallback key (attempt %s/%s) and retrying.",
                        msg,
                        attempt + 2,
                        max_attempts,
                    )
                    continue
                print(f"Gemini Vision API error: {e}")
                raise

            # Extract text from response (same logic as call_gemini)
            if hasattr(response, 'text'):
                text = response.text
                if text:
                    return text

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        part = candidate.content.parts[0]
                        text = getattr(part, 'text', None)
                        if text:
                            return text

            # Collect metadata for troubleshooting (mirrors call_gemini)
            debug_bits = []

            prompt_feedback = getattr(response, "prompt_feedback", None)
            if prompt_feedback:
                block_reason = getattr(prompt_feedback, "block_reason", None)
                if block_reason:
                    debug_bits.append(
                        f"prompt_block={getattr(block_reason, 'name', block_reason)}"
                    )
                safety_ratings = getattr(prompt_feedback, "safety_ratings", None)
                if safety_ratings:
                    ratings = ", ".join(
                        f"{getattr(r.category, 'name', r.category)}:{getattr(r.threshold, 'name', r.threshold)}"
                        for r in safety_ratings
                    )
                    debug_bits.append(f"prompt_safety=[{ratings}]")

            candidates = getattr(response, "candidates", None) or []
            for idx, cand in enumerate(candidates):
                finish_reason = getattr(cand, "finish_reason", None)
                if finish_reason:
                    debug_bits.append(
                        f"cand{idx}_finish={getattr(finish_reason, 'name', finish_reason)}"
                    )
                safety_ratings = getattr(cand, "safety_ratings", None)
                if safety_ratings:
                    ratings = ", ".join(
                        f"{getattr(r.category, 'name', r.category)}:{getattr(r.probability, 'name', r.probability)}"
                        for r in safety_ratings
                    )
                    debug_bits.append(f"cand{idx}_safety=[{ratings}]")

            usage = getattr(response, "usage_metadata", None)
            if usage:
                tokens_in = getattr(usage, "prompt_token_count", None)
                tokens_out = getattr(usage, "candidates_token_count", None)
                if tokens_in is not None:
                    debug_bits.append(f"prompt_tokens={tokens_in}")
                if tokens_out is not None:
                    debug_bits.append(f"candidate_tokens={tokens_out}")

            debug_msg = " | ".join(debug_bits) if debug_bits else "no response metadata"
            logging.warning(f"Gemini vision returned no text. Metadata: {debug_msg}")
            raise ValueError(f"Gemini vision returned no text (metadata: {debug_msg})")

    async def call_grok(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """Call Grok API using httpx client"""
        if not self.grok_available:
            raise ValueError("Grok not available")

        try:
            # Get generation parameters from config
            generation_config = self.config.get("grok", {}).get("generation", {})
            effective_temperature = temperature
            if effective_temperature is None:
                effective_temperature = generation_config.get("temperature", 0)

            # Prepare the request payload
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                        "role": "user",
                        "content": prompt
            })

            payload = {
                "messages": messages,
                "model": self.grok_model_name,
                "stream": False,
                "temperature": effective_temperature
            }

            # Make the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.grok_api_key}"
            }

            timeout = self.config.get("grok", {}).get("timeout", 120.0)

            response = await self.grok_client.post(
                f"{self.grok_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout
            )

            # Check for HTTP errors
            try:
                response.raise_for_status()
            except Exception as http_err:
                detail = ""
                try:
                    detail = response.text
                except Exception:
                    pass
                raise RuntimeError(
                    f"Grok HTTP error {response.status_code}: {detail}"
                ) from http_err

            # Parse the response
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format: {response_data}")

        except Exception as e:
            print(f"Grok API error: {e!r}")
            if e.__class__.__name__ == "ReadTimeout":
                raise RuntimeError(
                    "Grok API request timed out. Increase grok.timeout in config/llm.yaml or try again."
                ) from e
            raise

    async def call_deepseek(self, prompt: str) -> Optional[str]:
        """Call DeepSeek API (OpenAI-compatible chat completions)"""
        if not self.deepseek_available:
            raise ValueError("DeepSeek not available")

        try:
            generation_config = self.config.get("deepseek", {}).get("generation", {})
            temperature = generation_config.get("temperature", 0)

            params = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": self.deepseek_model_name,
                "stream": False,
                "temperature": temperature
            }
            response = await self.deepseek_client.chat.completions.create(**params)
            if response.choices:
                return response.choices[0].message.content
            raise ValueError(f"Unexpected response format: {response}")

        except Exception as e:
            print(f"DeepSeek API error: {e}")
            raise

    async def call_anthropic(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Call Anthropic Messages API with a simple user prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or ""}
                ],
            }
        ]
        return await self.call_anthropic_messages(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

    async def call_anthropic_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        model_name: Optional[str] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """Generic Anthropic Messages API caller."""
        if not self.anthropic_available:
            raise ValueError("Anthropic not available")

        await self._apply_rate_limit_delay("_anthropic_last_call", "anthropic_call_delay")

        try:
            generation_config = self.config.get("anthropic", {}).get("generation", {})
            effective_temperature = (
                temperature if temperature is not None else generation_config.get("temperature", 0)
            )

            payload: Dict[str, Any] = {
                "model": model_name or self.anthropic_model_name,
                "max_tokens": max_tokens,
                "temperature": effective_temperature,
                "messages": messages,
            }
            if system_prompt:
                payload["system"] = system_prompt

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            }

            response = await self.anthropic_client.post(
                f"{self.anthropic_base_url}/v1/messages",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            data = response.json()
            if isinstance(data, dict) and isinstance(data.get("content"), list):
                text_parts = [
                    block.get("text", "")
                    for block in data["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                self._mark_rate_limit_timestamp("_anthropic_last_call")
                result = "".join(text_parts)
                if not result:
                    LOGGER.warning(
                        "Anthropic API returned empty text content (model=%s, max_tokens=%d, response_id=%s)",
                        model_name or self.anthropic_model_name,
                        max_tokens,
                        data.get("id", "N/A"),
                    )
                return result
            raise ValueError(f"Unexpected response format: {data}")

        except Exception as e:
            error_msg = f"Anthropic API error: {type(e).__name__}: {e}"
            print(error_msg)
            if hasattr(e, "response") and getattr(e.response, "text", None):
                print(f"Response body: {e.response.text}")
            raise

    async def call_anthropic_vision(
        self,
        prompt: str,
        image_urls: Optional[list] = None,
        image_files: Optional[list] = None,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """Call Anthropic Messages API with vision (images via URL or local files).

        Args:
            prompt: User text prompt
            image_urls: List of image URLs
            image_files: List of local image file paths
            max_tokens: Max output tokens
        """
        if not getattr(self, 'anthropic_available', False):
            raise ValueError("Anthropic not available")

        await self._apply_rate_limit_delay("_anthropic_last_call", "anthropic_call_delay")

        try:
            generation_config = self.config.get("anthropic", {}).get("generation", {})
            temperature = generation_config.get("temperature", 0)

            content_blocks = [{"type": "text", "text": prompt or ""}]

            # Add URL image blocks
            for url in (image_urls or []):
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": url}
                })

            # Add base64 image blocks for local files
            for file_path in (image_files or []):
                try:
                    with open(file_path, 'rb') as f:
                        data_b64 = base64.b64encode(f.read()).decode('utf-8')
                    media_type, _ = mimetypes.guess_type(file_path)
                    if not media_type:
                        media_type = "image/png"
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data_b64
                        }
                    })
                except Exception as e:
                    logging.warning(f"Failed to read image file {file_path}: {e}")

            payload = {
                "model": self.anthropic_model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": content_blocks}
                ]
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }

            response = await self.anthropic_client.post(
                f"{self.anthropic_base_url}/v1/messages",
                json=payload,
                headers=headers,
                timeout=300.0
            )

            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "content" in data and isinstance(data["content"], list):
                text_parts = [
                    block.get("text", "")
                    for block in data["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                self._mark_rate_limit_timestamp("_anthropic_last_call")
                return "".join(text_parts)
            raise ValueError(f"Unexpected response format: {data}")

        except Exception as e:
            error_msg = f"Anthropic vision API error: {type(e).__name__}: {e}"
            print(error_msg)
            if hasattr(e, "response") and getattr(e.response, "text", None):
                print(f"Response body: {e.response.text}")
            raise

    async def call_qwen(self, prompt: str) -> Optional[str]:
        """Call Qwen (DashScope) via OpenAI-compatible chat completions API"""
        if not getattr(self, 'qwen_available', False):
            raise ValueError("Qwen not available")

        try:
            generation_config = self.config.get("qwen", {}).get("generation", {})
            temperature = generation_config.get("temperature", 0)

            params = {
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": self.qwen_model_name,
                "stream": False,
                "temperature": temperature,
                "extra_body": {"enable_thinking": False},
            }
            response = await self.qwen_client.chat.completions.create(**params)
            if response.choices:
                return response.choices[0].message.content
            raise ValueError(f"Unexpected response format: {response}")

        except Exception as e:
            print(f"Qwen API error: {e}")
            raise

    async def call_vllm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> Optional[str]:
        """Call a vLLM OpenAI-compatible server"""
        if not getattr(self, 'vllm_available', False):
            raise ValueError("vLLM not available")

        try:
            generation_config = self.config.get("vllm", {}).get("generation", {})
            effective_temperature = temperature if temperature is not None else generation_config.get("temperature", 0.8)
            max_new_tokens = max_tokens or generation_config.get("max_tokens", 16000)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "messages": messages,
                "model": model_name or self.vllm_model_name,
                "stream": False,
                "temperature": effective_temperature,
                "max_tokens": max_new_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }

            headers = {"Content-Type": "application/json"}
            if self.vllm_api_key:
                headers["Authorization"] = f"Bearer {self.vllm_api_key}"

            response = await self.vllm_client.post(
                f"{self.vllm_base_url}/chat/completions",
                json=payload,
                headers=headers
            )

            response.raise_for_status()
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            raise ValueError(f"Unexpected response format: {data}")

        except Exception as e:
            print(f"vLLM API error: {e}")
            raise

    async def call_both_models(self, prompt: str, use_web_search: bool = False) -> dict:
        """Call both Gemini and OpenAI models and return both results

        Args:
            prompt: The prompt/query to send
            use_web_search: If True, enables web search for both models

        Returns:
            dict with keys 'gemini' and 'openai' containing responses
        """
        results = {}

        # Call both models concurrently
        tasks = []

        if self.gemini_available:
            if use_web_search:
                tasks.append(("gemini", self.call_gemini_with_web_search(prompt)))
            else:
                tasks.append(("gemini", self.call_gemini(prompt)))

        if self.openai_available:
            if use_web_search:
                tasks.append(("openai", self.call_openai_with_web_search(prompt)))
            else:
                tasks.append(("openai", self.call_openai(prompt)))

        if self.grok_available:
            tasks.append(("grok", self.call_grok(prompt)))

        if self.deepseek_available:
            tasks.append(("deepseek", self.call_deepseek(prompt)))

        if getattr(self, 'anthropic_available', False):
            tasks.append(("anthropic", self.call_anthropic(prompt)))

        if getattr(self, 'qwen_available', False):
            tasks.append(("qwen", self.call_qwen(prompt)))

        # Execute concurrently
        if tasks:
            responses = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            for i, (model_name, _) in enumerate(tasks):
                if isinstance(responses[i], Exception):
                    results[model_name] = f"Error: {str(responses[i])}"
                else:
                    results[model_name] = responses[i]

        return results

    async def call_llm(self, prompt: str, preferred_model: str = "gemini", use_web_search: bool = False, use_both: bool = False) -> Optional[str]:
        """Call LLM with fallback to available models

        Args:
            prompt: The prompt/query to send to the LLM
            preferred_model: "gemini" or "openai"
            use_web_search: If True, enables web_search tool for supported models
            use_both: If True, calls both models and returns dict with both responses
        """

        # If use_both is requested, call both models
        if use_both:
            return await self.call_both_models(prompt, use_web_search=use_web_search)

        # Check if any models are available
        if not self.is_available():
            raise ValueError("No LLM models available - check API keys and library installations")

        # Try preferred model first
        if preferred_model == "gemini" and self.gemini_available:
            try:
                if use_web_search:
                    return await self.call_gemini_with_web_search(prompt)
                else:
                    return await self.call_gemini(prompt)
            except Exception as e:
                print(f"Gemini failed: {e}")
                if self.openai_available:
                    print("Trying OpenAI fallback...")
                    if use_web_search:
                        return await self.call_openai_with_web_search(prompt)
                    else:
                        return await self.call_openai(prompt)
                else:
                    raise ValueError("Gemini failed and no OpenAI fallback available")

        elif preferred_model == "openai" and self.openai_available:
            try:
                if use_web_search:
                    return await self.call_openai_with_web_search(prompt)
                else:
                    return await self.call_openai(prompt)
            except Exception as e:
                print(f"OpenAI failed: {e}")
                if self.gemini_available:
                    print("Trying Gemini fallback...")
                    return await self.call_gemini(prompt)
                else:
                    raise ValueError("OpenAI failed and no Gemini fallback available")

        elif preferred_model == "grok" and self.grok_available:
            try:
                return await self.call_grok(prompt)
            except Exception as e:
                print(f"Grok failed: {e}")
                # Fallback preference: DeepSeek then OpenAI then Gemini
                if self.deepseek_available:
                    print("Trying DeepSeek fallback...")
                    return await self.call_deepseek(prompt)
                if self.openai_available:
                    print("Trying OpenAI fallback...")
                    return await self.call_openai(prompt)
                if self.gemini_available:
                    print("Trying Gemini fallback...")
                    return await self.call_gemini(prompt)
                raise ValueError("Grok failed and no fallback available")

        elif preferred_model == "deepseek" and self.deepseek_available:
            try:
                return await self.call_deepseek(prompt)
            except Exception as e:
                print(f"DeepSeek failed: {e}")
                # Fallback preference: Grok then OpenAI then Gemini
                if self.grok_available:
                    print("Trying Grok fallback...")
                    return await self.call_grok(prompt)
                if self.openai_available:
                    print("Trying OpenAI fallback...")
                    return await self.call_openai(prompt)
                if self.gemini_available:
                    print("Trying Gemini fallback...")
                    return await self.call_gemini(prompt)
                raise ValueError("DeepSeek failed and no fallback available")

        elif preferred_model == "anthropic" and getattr(self, 'anthropic_available', False):
            try:
                return await self.call_anthropic(prompt)
            except Exception as e:
                print(f"Anthropic failed: {e}")
                # Fallback preference: OpenAI, Gemini, Grok, DeepSeek
                if self.openai_available:
                    print("Trying OpenAI fallback...")
                    return await self.call_openai(prompt)
                if self.gemini_available:
                    print("Trying Gemini fallback...")
                    return await self.call_gemini(prompt)
                if self.grok_available:
                    print("Trying Grok fallback...")
                    return await self.call_grok(prompt)
                if self.deepseek_available:
                    print("Trying DeepSeek fallback...")
                    return await self.call_deepseek(prompt)
                raise ValueError("Anthropic failed and no fallback available")

        elif preferred_model == "qwen" and getattr(self, 'qwen_available', False):
            try:
                return await self.call_qwen(prompt)
            except Exception as e:
                print(f"Qwen failed: {e}")
                # Fallback preference: OpenAI, Gemini, Grok, DeepSeek, Anthropic
                if self.openai_available:
                    print("Trying OpenAI fallback...")
                    return await self.call_openai(prompt)
                if self.gemini_available:
                    print("Trying Gemini fallback...")
                    return await self.call_gemini(prompt)
                if self.grok_available:
                    print("Trying Grok fallback...")
                    return await self.call_grok(prompt)
                if self.deepseek_available:
                    print("Trying DeepSeek fallback...")
                    return await self.call_deepseek(prompt)
                if getattr(self, 'anthropic_available', False):
                    print("Trying Anthropic fallback...")
                    return await self.call_anthropic(prompt)
                raise ValueError("Qwen failed and no fallback available")

        elif preferred_model == "vllm" and getattr(self, 'vllm_available', False):
            try:
                return await self.call_vllm(prompt)
            except Exception as e:
                print(f"vLLM failed: {e}")
                # Fallback preference: OpenAI, Gemini, Hugging Face local (via Qwen/DeepSeek) etc.
                if self.openai_available:
                    print("Trying OpenAI fallback...")
                    return await self.call_openai(prompt)
                if self.gemini_available:
                    print("Trying Gemini fallback...")
                    return await self.call_gemini(prompt)
                if self.grok_available:
                    print("Trying Grok fallback...")
                    return await self.call_grok(prompt)
                if self.deepseek_available:
                    print("Trying DeepSeek fallback...")
                    return await self.call_deepseek(prompt)
                if getattr(self, 'qwen_available', False):
                    print("Trying Qwen fallback...")
                    return await self.call_qwen(prompt)
                raise ValueError("vLLM failed and no fallback available")

        # Fallback to any available model
        if self.gemini_available and preferred_model != "gemini":
            try:
                if use_web_search:
                    return await self.call_gemini_with_web_search(prompt)
                else:
                    return await self.call_gemini(prompt)
            except Exception as e:
                print(f"Gemini fallback failed: {e}")

        if self.openai_available and preferred_model != "openai":
            try:
                if use_web_search:
                    return await self.call_openai_with_web_search(prompt)
                else:
                    return await self.call_openai(prompt)
            except Exception as e:
                print(f"OpenAI fallback failed: {e}")

        if self.grok_available and preferred_model != "grok":
            try:
                return await self.call_grok(prompt)
            except Exception as e:
                print(f"Grok fallback failed: {e}")

        if self.deepseek_available and preferred_model != "deepseek":
            try:
                return await self.call_deepseek(prompt)
            except Exception as e:
                print(f"DeepSeek fallback failed: {e}")

        if getattr(self, 'anthropic_available', False) and preferred_model != "anthropic":
            try:
                return await self.call_anthropic(prompt)
            except Exception as e:
                print(f"Anthropic fallback failed: {e}")

        if getattr(self, 'qwen_available', False) and preferred_model != "qwen":
            try:
                return await self.call_qwen(prompt)
            except Exception as e:
                print(f"Qwen fallback failed: {e}")

        if getattr(self, 'vllm_available', False) and preferred_model != "vllm":
            try:
                return await self.call_vllm(prompt)
            except Exception as e:
                print(f"vLLM fallback failed: {e}")

        raise ValueError(f"All available LLM models failed. Available: {self.get_available_models()}")

    def get_available_models(self) -> list:
        """Get list of available models"""
        available = []
        if self.openai_available:
            available.append("openai")
        if self.gemini_available:
            available.append("gemini")
        if getattr(self, 'grok_available', False):
            available.append("grok")
        if getattr(self, 'deepseek_available', False):
            available.append("deepseek")
        if getattr(self, 'anthropic_available', False):
            available.append("anthropic")
        if getattr(self, 'qwen_available', False):
            available.append("qwen")
        if getattr(self, 'vllm_available', False):
            available.append("vllm")
        return available

    def is_available(self) -> bool:
        """Check if any LLM is available"""
        return (
            getattr(self, 'openai_available', False) or
            getattr(self, 'gemini_available', False) or
            getattr(self, 'grok_available', False) or
            getattr(self, 'deepseek_available', False) or
            getattr(self, 'anthropic_available', False) or
            getattr(self, 'qwen_available', False) or
            getattr(self, 'vllm_available', False)
        )


# Test function
async def test_llm_manager():
    """Test the LLM manager"""
    manager = LLMManager()

    print(f"Available models: {manager.get_available_models()}")

    if manager.is_available():
        try:
            response = await manager.call_llm("Say hello in a friendly way")
            print(f"Test response: {response}")
        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("No LLM models available for testing")


if __name__ == "__main__":
    asyncio.run(test_llm_manager())
