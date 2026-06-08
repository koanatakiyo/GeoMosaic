from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.direct_scoring import (  # noqa: E402
    build_direct_scoring_tasks,
    claim_text_quality,
    compact_direct_scoring_results,
    execute_direct_scoring,
    existing_completed_task_models,
    normalize_direct_scoring_response,
    reset_direct_scoring_model_rows,
)
import geomosaic_hg.direct_scoring as direct_scoring_module  # noqa: E402
from run_direct_scoring import DEFAULT_MODEL_IDS, provider_available  # noqa: E402
from run_direct_scoring import build_model_caller  # noqa: E402


class DirectScoringE4Test(unittest.TestCase):
    def test_e4_scorer_pool_excludes_extraction_models(self) -> None:
        self.assertEqual(
            set(DEFAULT_MODEL_IDS),
            {"grok", "openai", "llama", "doubao", "deepseek", "qwen"},
        )
        self.assertNotIn("gemini", DEFAULT_MODEL_IDS)
        self.assertEqual(DEFAULT_MODEL_IDS["qwen"], "qwen3.5-122b-a10b")
        self.assertEqual(DEFAULT_MODEL_IDS["doubao"], "doubao-seed-2-0-pro-260215")

    def test_runner_preflight_checks_provider_availability(self) -> None:
        manager = SimpleNamespace(openai_available=False, qwen_available=True)

        self.assertFalse(provider_available(manager, "openai"))
        self.assertTrue(provider_available(manager, "qwen"))

    def test_runner_reuses_one_event_loop_for_httpx_style_clients(self) -> None:
        manager = SimpleNamespace(grok_model_name="grok-4-1-fast")
        loop_ids = []

        async def fake_call_grok(prompt: str, temperature: int = 0):
            loop_ids.append(id(__import__("asyncio").get_running_loop()))
            return prompt

        manager.call_grok = fake_call_grok
        caller = build_model_caller(manager, "grok")

        self.assertEqual(caller("first", "grok-4-1-fast"), "first")
        self.assertEqual(caller("second", "grok-4-1-fast"), "second")
        self.assertEqual(len(set(loop_ids)), 1)

    def test_runner_passes_max_output_tokens_to_llama(self) -> None:
        calls = []

        async def fake_call_llama_chat(messages, *, temperature, max_tokens, model_name, extra_body=None, **_):
            calls.append({"messages": messages, "max_tokens": max_tokens, "model_name": model_name, "extra_body": extra_body})
            return "ok"

        manager = SimpleNamespace(call_llama_chat=fake_call_llama_chat)
        caller = build_model_caller(manager, "llama", max_output_tokens=24000)

        self.assertEqual(caller("prompt", "meta-llama/llama-4-scout"), "ok")
        self.assertEqual(calls[0]["max_tokens"], 24000)
        self.assertEqual(calls[0]["extra_body"], {})

    def test_runner_passes_max_output_tokens_to_doubao(self) -> None:
        calls = []

        async def fake_call_doubao(prompt, *, temperature, max_tokens, model_name):
            calls.append({"prompt": prompt, "max_tokens": max_tokens, "model_name": model_name})
            return "ok"

        manager = SimpleNamespace(call_doubao=fake_call_doubao)
        caller = build_model_caller(manager, "doubao", max_output_tokens=24000)

        self.assertEqual(caller("prompt", "doubao-seed-2-0-pro-260215"), "ok")
        self.assertEqual(calls[0]["max_tokens"], 24000)

    def test_plans_document_bundle_tasks_with_representative_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            manifest_rows = [
                {
                    "event_id": "crimea",
                    "document_id": "un-res",
                    "source_filename": "crimea__multilateral__official_response__fr__un-res.pdf",
                    "language": "fr",
                    "viewpoint_origin": "multilateral",
                    "evidence_scope": "official_response",
                    "source_url": "https://example/fr",
                },
                {
                    "event_id": "crimea",
                    "document_id": "un-res",
                    "source_filename": "crimea__multilateral__official_response__en__un-res.pdf",
                    "language": "en",
                    "viewpoint_origin": "multilateral",
                    "evidence_scope": "official_response",
                    "source_url": "https://example/en",
                },
            ]
            doc_rows = [
                {
                    "event_id": "crimea",
                    "document_id": "un-res",
                    "source_filename": "crimea__multilateral__official_response__fr__un-res.pdf",
                    "language": "fr",
                    "text": "French text",
                    "char_count": 11,
                },
                {
                    "event_id": "crimea",
                    "document_id": "un-res",
                    "source_filename": "crimea__multilateral__official_response__en__un-res.pdf",
                    "language": "en",
                    "text": "English text about territorial integrity.",
                    "char_count": 39,
                },
            ]
            claim_rows = [
                {
                    "claim_id": "crimea:official:multilateral:UN:E1",
                    "event_id": "crimea",
                    "claim_text": "The document affirms Ukraine's territorial integrity.",
                }
            ]
            (parsed / "official_doc_text.jsonl").write_text(
                "\n".join(json.dumps(row) for row in doc_rows) + "\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            claims = root / "claims.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")
            claims.write_text("\n".join(json.dumps(row) for row in claim_rows) + "\n", encoding="utf-8")

            summary = build_direct_scoring_tasks(
                parsed_dir=parsed,
                manifest_path=manifest,
                claims_path=claims,
                output_dir=root / "direct_scoring",
                source_layers=["official"],
            )

            tasks = [
                json.loads(line)
                for line in (root / "direct_scoring" / "direct_scoring_tasks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(summary["tasks"], 1)
            self.assertEqual(tasks[0]["task_type"], "direct_scoring")
            self.assertEqual(tasks[0]["source_vp"], "multilateral")
            self.assertEqual(tasks[0]["scored_vp"], "UN")
            self.assertEqual(tasks[0]["documents"][0]["language"], "en")
            self.assertNotIn("passages", tasks[0])
            self.assertIn("English text", tasks[0]["bundle_text"])

    def test_claim_text_quality_does_not_penalize_ordinary_claim_word(self) -> None:
        ordinary = "The document affirms the claim that territorial integrity matters."
        templated = "claim D1 evaluated for UN viewpoint"

        self.assertGreater(claim_text_quality(ordinary), claim_text_quality(templated))

    def test_map_reduce_tasks_do_not_duplicate_full_bundle_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            manifest_rows = [
                {
                    "event_id": "scs",
                    "document_id": "award",
                    "source_filename": "scs__legal_international__anchor_document__en__award.pdf",
                    "language": "en",
                    "viewpoint_origin": "legal_international",
                    "evidence_scope": "anchor_document",
                    "source_url": "https://example/award",
                }
            ]
            doc_rows = [
                {
                    "event_id": "scs",
                    "document_id": "award",
                    "source_filename": "scs__legal_international__anchor_document__en__award.pdf",
                    "language": "en",
                    "text": "0123456789" * 10,
                    "char_count": 100,
                }
            ]
            claim_rows = [
                {
                    "claim_id": "scs:official:legal_international:China:D1",
                    "event_id": "scs",
                    "claim_text": "The award addresses maritime claims.",
                }
            ]
            (parsed / "official_doc_text.jsonl").write_text("\n".join(json.dumps(row) for row in doc_rows) + "\n", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            claims = root / "claims.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")
            claims.write_text("\n".join(json.dumps(row) for row in claim_rows) + "\n", encoding="utf-8")

            build_direct_scoring_tasks(parsed, manifest, claims, root / "direct_scoring", max_bundle_chars=40)
            task = json.loads((root / "direct_scoring" / "direct_scoring_tasks.jsonl").read_text(encoding="utf-8"))

            self.assertEqual(task["strategy"], "document_map_reduce")
            self.assertEqual(task["bundle_text"], "")
            self.assertGreater(len(task["chunks"]), 1)

    def test_build_direct_scoring_tasks_reuses_bundle_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            doc_rows = [
                {
                    "event_id": "crimea",
                    "document_id": "doc",
                    "source_filename": "crimea__multilateral__official_response__en__doc.txt",
                    "language": "en",
                    "text": "Text.",
                    "char_count": 5,
                }
            ]
            manifest_rows = [
                {
                    "event_id": "crimea",
                    "document_id": "doc",
                    "source_filename": "crimea__multilateral__official_response__en__doc.txt",
                    "language": "en",
                    "viewpoint_origin": "multilateral",
                    "evidence_scope": "official_response",
                }
            ]
            claim_rows = [
                {"claim_id": "crimea:official:multilateral:UN:D1", "event_id": "crimea", "claim_text": "Text."}
            ]
            (parsed / "official_doc_text.jsonl").write_text("\n".join(json.dumps(row) for row in doc_rows) + "\n", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            claims = root / "claims.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")
            claims.write_text("\n".join(json.dumps(row) for row in claim_rows) + "\n", encoding="utf-8")

            original = direct_scoring_module.build_document_bundles
            calls = {"count": 0}

            def wrapped(docs):
                calls["count"] += 1
                return original(docs)

            try:
                direct_scoring_module.build_document_bundles = wrapped
                build_direct_scoring_tasks(parsed, manifest, claims, root / "direct_scoring")
            finally:
                direct_scoring_module.build_document_bundles = original

            self.assertEqual(calls["count"], 1)

    def test_normalizes_direct_scoring_json_response(self) -> None:
        response = """```json
        {"scores": [
          {"claim_id": "E1", "score": 2, "max": 2, "justification": "explicit"},
          {"claim_id": "E2", "score": 5, "max": 2, "justification": "clamped"},
          {"claim_id": "E3", "score": null, "max": 2, "justification": "PARSE_FAIL"}
        ]}
        ```"""

        scores = normalize_direct_scoring_response(response, expected_claims=["E1", "E2", "E3", "E4"])

        self.assertEqual(scores["E1"]["score"], 2)
        self.assertEqual(scores["E2"]["score"], 2)
        self.assertIsNone(scores["E3"]["score"])
        self.assertIsNone(scores["E4"]["score"])

    def test_normalizes_response_with_raw_control_character(self) -> None:
        response = '{"scores":[{"claim_id":"E1","score":2,"max":2,"justification":"line\x01break"}]}'

        scores = normalize_direct_scoring_response(response, expected_claims=["E1"])

        self.assertEqual(scores["E1"]["score"], 2)

    def test_execute_direct_scoring_writes_claim_audit_compatible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            output = root / "scores.jsonl"
            summary = root / "summary.json"
            task = {
                "task_id": "direct_score_crimea_multilateral_UN",
                "task_type": "direct_scoring",
                "event_id": "crimea",
                "source_vp": "multilateral",
                "scored_vp": "UN",
                "run_id": 1,
                "strategy": "full_text",
                "documents": [{"document_id": "un-res", "language": "en"}],
                "bundle_text": "The resolution affirms territorial integrity.",
                "claims": [
                    {"claim_id": "E1", "claim_text": "Ukraine territorial integrity is affirmed.", "max": 2}
                ],
            }
            tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")

            def fake_caller(prompt: str, model_id: str) -> str:
                self.assertIn("GeoGround-style direct scoring", prompt)
                self.assertIn("The resolution affirms", prompt)
                return '{"scores":[{"claim_id":"E1","score":2,"max":2,"justification":"directly stated"}]}'

            result = execute_direct_scoring(
                tasks_path=tasks,
                output_path=output,
                summary_path=summary,
                model_name="fake-model",
                model_id="fake-id",
                model_caller=fake_caller,
                resume=False,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["completed_tasks"], 1)
            self.assertEqual(rows[0]["event"], "crimea")
            self.assertEqual(rows[0]["model"], "fake-model")
            self.assertEqual(rows[0]["source_vp"], "multilateral")
            self.assertEqual(rows[0]["scored_vp"], "UN")
            self.assertEqual(rows[0]["claim_id"], "E1")
            self.assertEqual(rows[0]["score"], 2)
            self.assertEqual(rows[0]["max"], 2)
            self.assertEqual(rows[0]["protocol"], "geoground_direct_scoring_v0")

    def test_completed_task_detection_requires_all_expected_claim_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores.jsonl"
            rows = [
                {"task_id": "task1", "model": "qwen", "claim_id": "D1", "score": 2},
                {"task_id": "task1", "model": "qwen", "claim_id": "D2", "score": None},
                {"task_id": "task2", "model": "qwen", "claim_id": "D1", "score": 0},
                {"task_id": "task2", "model": "qwen", "claim_id": "D2", "score": 1},
            ]
            output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            completed = existing_completed_task_models(output, expected_claim_counts={"task1": 2, "task2": 2})

            self.assertNotIn(("task1", "qwen"), completed)
            self.assertIn(("task2", "qwen"), completed)

    def test_compact_results_removes_only_resolved_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores.jsonl"
            rows = [
                {"task_id": "task1", "model": "qwen", "status": "failed", "error": "old"},
                {"task_id": "task1", "model": "qwen", "claim_id": "D1", "score": 2},
                {"task_id": "task2", "model": "qwen", "status": "failed", "error": "still bad"},
            ]
            output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = compact_direct_scoring_results(output)
            compacted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["removed_failed_rows"], 1)
            self.assertEqual(len(compacted), 2)
            self.assertFalse(any(row.get("task_id") == "task1" and row.get("status") == "failed" for row in compacted))
            self.assertTrue(any(row.get("task_id") == "task2" and row.get("status") == "failed" for row in compacted))

    def test_compact_results_keeps_latest_completed_claim_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores.jsonl"
            rows = [
                {"task_id": "task1", "model": "qwen", "claim_id": "D1", "score": 0, "justification": "old"},
                {"task_id": "task1", "model": "qwen", "claim_id": "D1", "score": 2, "justification": "new"},
                {"task_id": "task1", "model": "qwen", "claim_id": "D2", "score": 1, "justification": "only"},
            ]
            output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = compact_direct_scoring_results(output)
            compacted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["removed_duplicate_completed_rows"], 1)
            self.assertEqual(len(compacted), 2)
            self.assertFalse(any(row.get("justification") == "old" for row in compacted))
            self.assertTrue(any(row.get("claim_id") == "D1" and row.get("score") == 2 for row in compacted))

    def test_compact_results_removes_resolved_missing_score_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scores.jsonl"
            rows = [
                {"task_id": "task1", "model": "llama", "claim_id": "C1", "score": None, "justification": "MISSING_SCORE"},
                {"task_id": "task1", "model": "llama", "claim_id": "C1", "score": 1, "justification": "later valid"},
                {"task_id": "task1", "model": "llama", "claim_id": "C2", "score": None, "justification": "still missing"},
            ]
            output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = compact_direct_scoring_results(output)
            compacted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["removed_resolved_missing_score_rows"], 1)
            self.assertEqual(len(compacted), 2)
            self.assertTrue(any(row.get("claim_id") == "C1" and row.get("score") == 1 for row in compacted))
            self.assertTrue(any(row.get("claim_id") == "C2" and row.get("score") is None for row in compacted))

    def test_reset_model_rows_backs_up_and_removes_selected_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "scores.jsonl"
            backup = root / "removed.jsonl"
            rows = [
                {"task_id": "task1", "model": "deepseek", "claim_id": "D1", "score": 2},
                {"task_id": "task2", "model": "qwen", "claim_id": "D1", "score": 1},
                {"task_id": "task3", "model": "grok", "claim_id": "D1", "score": 0},
            ]
            output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = reset_direct_scoring_model_rows(output, models={"deepseek", "qwen"}, backup_path=backup)
            kept = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            removed = [json.loads(line) for line in backup.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["removed_rows"], 2)
            self.assertEqual(summary["kept_rows"], 1)
            self.assertEqual([row["model"] for row in kept], ["grok"])
            self.assertEqual({row["model"] for row in removed}, {"deepseek", "qwen"})


if __name__ == "__main__":
    unittest.main()
