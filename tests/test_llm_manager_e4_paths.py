from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from llm_manager import LLMManager  # noqa: E402


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.params = None

    async def create(self, **params):
        self.params = params
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok"))
            ]
        )


class _FakeResponses:
    def __init__(self) -> None:
        self.params = None

    async def create(self, **params):
        self.params = params
        return SimpleNamespace(output_text="ok")


class LLMManagerE4PathsTest(unittest.TestCase):
    def manager_without_init(self) -> LLMManager:
        return LLMManager.__new__(LLMManager)

    def test_deepseek_uses_openai_compatible_chat_completions(self) -> None:
        manager = self.manager_without_init()
        completions = _FakeChatCompletions()
        manager.deepseek_available = True
        manager.deepseek_model_name = "deepseek-chat"
        manager.deepseek_extra_body = {"thinking": {"type": "enabled"}}
        manager.deepseek_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        manager.config = {"deepseek": {"generation": {"temperature": 0}}}

        result = asyncio.run(manager.call_deepseek("score this"))

        self.assertEqual(result, "ok")
        self.assertEqual(completions.params["model"], "deepseek-chat")
        self.assertEqual(completions.params["messages"][0]["content"], "score this")
        self.assertNotIn("extra_body", completions.params)

    def test_qwen_uses_openai_compatible_chat_completions(self) -> None:
        manager = self.manager_without_init()
        completions = _FakeChatCompletions()
        manager.qwen_available = True
        manager.qwen_model_name = "qwen3.5-122b-a10b"
        manager.qwen_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        manager.config = {"qwen": {"generation": {"temperature": 0}, "extra_body": {"enable_thinking": True}}}

        result = asyncio.run(manager.call_qwen("score this"))

        self.assertEqual(result, "ok")
        self.assertEqual(completions.params["model"], "qwen3.5-122b-a10b")
        self.assertEqual(completions.params["messages"][0]["content"], "score this")
        self.assertEqual(completions.params["extra_body"], {"enable_thinking": False})

    def test_doubao_uses_responses_api(self) -> None:
        manager = self.manager_without_init()
        responses = _FakeResponses()
        manager.doubao_available = True
        manager.doubao_model_name = "doubao-seed-2-0-pro-260215"
        manager.doubao_client = SimpleNamespace(responses=responses)
        manager.config = {"doubao": {"generation": {"temperature": 0}}}

        result = asyncio.run(manager.call_doubao("score this", max_tokens=1024))

        self.assertEqual(result, "ok")
        self.assertEqual(responses.params["model"], "doubao-seed-2-0-pro-260215")
        self.assertEqual(responses.params["input"][0]["content"][0]["text"], "score this")
        self.assertEqual(responses.params["max_output_tokens"], 1024)

    def test_llama_allows_empty_extra_body_to_disable_config_thinking(self) -> None:
        manager = self.manager_without_init()
        completions = _FakeChatCompletions()
        manager.llama_available = True
        manager.llama_api_keys = ["key"]
        manager.llama_model_name = "meta-llama/llama-4-scout"
        manager.llama_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        manager.config = {"llama": {"extra_body": {"enable_thinking": True}}}

        result = asyncio.run(
            manager.call_llama_chat(
                [{"role": "user", "content": "score this"}],
                temperature=0,
                max_tokens=1024,
                extra_body={},
            )
        )

        self.assertEqual(result, "ok")
        self.assertEqual(completions.params["extra_body"], {})

    def test_gemini_web_search_path_falls_back_to_plain_gemini(self) -> None:
        manager = self.manager_without_init()
        calls = []

        async def fake_call_gemini(prompt: str):
            calls.append(prompt)
            return "gemini-ok"

        manager.call_gemini = fake_call_gemini

        result = asyncio.run(manager.call_gemini_with_web_search("hello"))

        self.assertEqual(result, "gemini-ok")
        self.assertEqual(calls, ["hello"])


if __name__ == "__main__":
    unittest.main()
