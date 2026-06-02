from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.claim_grounding import (  # noqa: E402
    clean_human_evidence_text,
    adjudicate_task_judgments,
    execute_claim_grounding,
    export_claim_grounding_audit,
    normalize_grounding_response,
    plan_claim_grounding_tasks,
)
from run_claim_grounding import AnthropicClaudeGroundingCaller, DEFAULT_CLAUDE_MODEL, load_anthropic_api_key  # noqa: E402
from plan_claim_grounding import parse_source_layers_arg  # noqa: E402


class ClaimGroundingStageCTest(unittest.TestCase):
    def test_plans_official_claim_tasks_from_parsed_passages_without_stage_b_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            out = root / "claim_grounding"
            parsed.mkdir()

            claims = [
                {
                    "claim_id": "crimea:official:un:Russia:A1",
                    "claim_text": "The UN resolution affirms Ukraine territorial integrity and rejects the Crimea referendum.",
                    "event_id": "crimea",
                },
                {
                    "claim_id": "crimea:news:reuters:Russia:A1",
                    "claim_text": "A news-only claim should not be planned when source_layers=official.",
                    "event_id": "crimea",
                },
            ]
            passages = [
                {
                    "passage_id": "passage_good_en_0000",
                    "document_id": "good",
                    "event_id": "crimea",
                    "language": "en",
                    "source_filename": "crimea__multilateral__official_response__en__good.pdf",
                    "page_start": 2,
                    "page_end": 2,
                    "char_start": 100,
                    "char_end": 240,
                    "text": "The resolution affirms the territorial integrity of Ukraine and states the referendum has no validity.",
                },
                {
                    "passage_id": "passage_weak_en_0000",
                    "document_id": "weak",
                    "event_id": "crimea",
                    "language": "en",
                    "source_filename": "crimea__russia__anchor_document__en__weak.txt",
                    "page_start": 1,
                    "page_end": 1,
                    "char_start": 0,
                    "char_end": 50,
                    "text": "A short official notice about a meeting.",
                },
                {
                    "passage_id": "passage_other_en_0000",
                    "document_id": "other",
                    "event_id": "ukraine",
                    "language": "en",
                    "source_filename": "ukraine__western__official_response__en__other.txt",
                    "page_start": 1,
                    "page_end": 1,
                    "char_start": 0,
                    "char_end": 80,
                    "text": "Ukraine passage should not be mixed into Crimea tasks.",
                },
            ]

            (root / "claims.jsonl").write_text("\n".join(json.dumps(row) for row in claims) + "\n", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("\n".join(json.dumps(row) for row in passages) + "\n", encoding="utf-8")

            summary = plan_claim_grounding_tasks(
                claims_path=root / "claims.jsonl",
                parsed_dir=parsed,
                output_dir=out,
                source_layers={"official"},
                top_k=2,
            )

            rows = [json.loads(line) for line in (out / "claim_grounding_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["tasks"], 1)
            self.assertEqual(rows[0]["claim_id"], "crimea:official:un:Russia:A1")
            self.assertTrue(rows[0]["load_bearing"])
            self.assertFalse(rows[0]["stage_b_inputs_used"])
            self.assertEqual(rows[0]["stage_b_policy"], "exclude_stage_b_outputs")
            self.assertNotIn("summary", rows[0])
            self.assertEqual([p["passage_id"] for p in rows[0]["passages"]], ["passage_good_en_0000", "passage_weak_en_0000"])
            self.assertTrue(rows[0]["passages"][0]["score"] > rows[0]["passages"][1]["score"])

    def test_normalizes_grounding_response_by_whitelist_and_label_alias(self) -> None:
        normalized = normalize_grounding_response(
            '```json{"label":"contradiction","confidence":"0.77","rationale":"Because of passage A.",'
            '"cited_passage_ids":["p1",null,""],"document_position":"supports","summary":"forbidden"}```'
        )

        self.assertEqual(normalized["label"], "contradict")
        self.assertAlmostEqual(normalized["confidence"], 0.77)
        self.assertEqual(normalized["cited_passage_ids"], ["p1"])
        self.assertNotIn("document_position", normalized)
        self.assertNotIn("summary", normalized)

    def test_two_model_disagreement_requires_human_audit(self) -> None:
        task = {
            "task_id": "claim_ground_crimea_x",
            "event_id": "crimea",
            "claim_id": "crimea:official:un:Russia:A1",
            "claim_text": "claim",
            "audit_sample": False,
        }
        judgments = [
            {"task_id": task["task_id"], "model_name": "gemini", "label": "support", "status": "completed"},
            {"task_id": task["task_id"], "model_name": "claude", "label": "context", "status": "completed"},
        ]

        row = adjudicate_task_judgments(task, judgments, expected_models=["gemini", "claude"])

        self.assertEqual(row["status"], "disagreement")
        self.assertTrue(row["needs_human_audit"])
        self.assertEqual(row["human_audit_reason"], "model_disagreement")
        self.assertEqual(row["labels_by_model"], {"claude": "context", "gemini": "support"})

    def test_execute_limit_counts_tasks_not_individual_model_judgments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_path = root / "tasks.jsonl"
            output = root / "judgments.jsonl"
            adjudication = root / "adjudication.jsonl"
            summary_path = root / "summary.json"
            tasks = [
                {
                    "task_id": f"task-{idx}",
                    "event_id": "crimea",
                    "claim_id": f"crimea:official:un:Russia:A{idx}",
                    "claim_text": "claim",
                    "claim_source_layer": "official",
                    "passages": [{"passage_id": "p1", "text": "passage"}],
                    "audit_sample": False,
                }
                for idx in range(3)
            ]
            tasks_path.write_text("\n".join(json.dumps(row) for row in tasks) + "\n", encoding="utf-8")

            def support_caller(prompt: str, model_id: str) -> dict:
                return {"label": "support", "confidence": 0.9, "rationale": model_id, "cited_passage_ids": ["p1"]}

            summary = execute_claim_grounding(
                tasks_path=tasks_path,
                output_path=output,
                adjudication_path=adjudication,
                summary_path=summary_path,
                model_callers={"gemini": support_caller, "claude": support_caller},
                model_ids={"gemini": "gemini-test", "claude": "claude-test"},
                resume=False,
                limit=2,
            )

            judgment_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            adjudication_rows = [json.loads(line) for line in adjudication.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["tasks_total"], 3)
            self.assertEqual(summary["tasks_selected"], 2)
            self.assertTrue(summary["partial_adjudication"])
            self.assertEqual(summary["adjudication_scope"], "selected_tasks")
            self.assertEqual(summary["completed"], 4)
            self.assertEqual(len(judgment_rows), 4)
            self.assertEqual(len(adjudication_rows), 2)

    def test_default_claude_model_uses_configurable_46_family_fallback(self) -> None:
        self.assertEqual(DEFAULT_CLAUDE_MODEL, "claude-sonnet-4-6")

    def test_anthropic_caller_uses_tool_schema_and_reads_tool_use(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "ground_claim",
                                "input": {
                                    "label": "support",
                                    "confidence": 0.88,
                                    "rationale": "The passage directly states it.",
                                    "cited_passage_ids": ["p1"],
                                },
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(req, timeout=120):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        import run_claim_grounding

        original_urlopen = run_claim_grounding.request.urlopen
        run_claim_grounding.request.urlopen = fake_urlopen
        try:
            caller = AnthropicClaudeGroundingCaller(
                api_key="test-key",
                temperature=0.0,
                max_output_tokens=256,
                retries=0,
                retry_backoff_seconds=0.0,
                timeout=17,
            )
            raw = caller("prompt", "claude-test")
        finally:
            run_claim_grounding.request.urlopen = original_urlopen

        payload = captured["payload"]
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "ground_claim"})
        self.assertEqual(payload["tools"][0]["input_schema"]["required"], ["label", "confidence", "rationale", "cited_passage_ids"])
        self.assertEqual(captured["timeout"], 17)
        self.assertEqual(json.loads(raw)["label"], "support")

    def test_loads_anthropic_key_from_llm_yaml_when_env_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "llm.yaml"
            config.write_text("anthropic:\n  api_key: yaml-secret\n", encoding="utf-8")

            self.assertEqual(load_anthropic_api_key("GEOMOSAIC_TEST_MISSING_KEY", config), "yaml-secret")

    def test_empty_source_layers_warns_and_preserves_official_default(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            parsed = parse_source_layers_arg("")

        self.assertEqual(parsed, {"official"})
        self.assertIn("empty --source-layers", stderr.getvalue())

    def test_export_audit_queue_flags_invalid_citations_and_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            judgments = root / "judgments.jsonl"
            adjudication = root / "adjudication.jsonl"
            output = root / "audit"
            task = {
                "task_id": "task-1",
                "event_id": "iraq",
                "claim_id": "iraq:official:us:Russia:A1",
                "claim_text": "claim",
                "claim_source_layer": "official",
                "passages": [{"passage_id": "p1", "text": "allowed"}],
            }
            judgment_rows = [
                {
                    "task_id": "task-1",
                    "model_name": "gemini",
                    "label": "support",
                    "confidence": 1.0,
                    "rationale": "uses an unavailable passage",
                    "cited_passage_ids": ["p1", "p9"],
                    "status": "completed",
                },
                {
                    "task_id": "task-1",
                    "model_name": "claude",
                    "label": "insufficient",
                    "confidence": 0.61,
                    "rationale": "not enough evidence",
                    "cited_passage_ids": [],
                    "status": "completed",
                },
            ]
            adjudication_row = {
                "task_id": "task-1",
                "event_id": "iraq",
                "claim_id": task["claim_id"],
                "claim_text": task["claim_text"],
                "status": "disagreement",
                "labels_by_model": {"gemini": "support", "claude": "insufficient"},
                "needs_human_audit": True,
            }
            tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
            judgments.write_text("\n".join(json.dumps(row) for row in judgment_rows) + "\n", encoding="utf-8")
            adjudication.write_text(json.dumps(adjudication_row) + "\n", encoding="utf-8")

            summary = export_claim_grounding_audit(
                tasks_path=tasks,
                judgments_path=judgments,
                adjudication_path=adjudication,
                output_dir=output,
                hk_insufficient_sample_size=0,
            )

            rows = [json.loads(line) for line in (output / "human_audit_queue.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["audit_rows"], 1)
            self.assertEqual(summary["invalid_citation_tasks"], 1)
            self.assertEqual(rows[0]["audit_flags"], ["model_disagreement", "invalid_citation:gemini"])
            self.assertEqual(rows[0]["audit_mode"], "diagnostic_triage")
            self.assertIn("passage_quality_issue", rows[0]["diagnostic_label_options"])
            self.assertIn("retrieval_failure", rows[0]["diagnostic_label_options"])
            self.assertIn("not ground truth", rows[0]["audit_instruction"])
            self.assertEqual(rows[0]["model_judgments"]["gemini"]["invalid_cited_passage_ids"], ["p9"])
            self.assertEqual(rows[0]["confidence_use_policy"]["gemini"], "display_only_do_not_sort_or_weight")
            self.assertIn("human_evidence_text", rows[0]["passages"][0])

    def test_export_audit_queue_samples_hongkong_insufficient_agreements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = root / "tasks.jsonl"
            judgments = root / "judgments.jsonl"
            adjudication = root / "adjudication.jsonl"
            output = root / "audit"
            task_rows = []
            judgment_rows = []
            adjudication_rows = []
            for idx in range(3):
                task_id = f"hk-{idx}"
                task_rows.append(
                    {
                        "task_id": task_id,
                        "event_id": "hongkong",
                        "claim_id": f"hongkong:official:china:China:A{idx}",
                        "claim_text": "claim",
                        "claim_source_layer": "official",
                        "passages": [{"passage_id": f"p{idx}", "text": "allowed"}],
                    }
                )
                for model in ("gemini", "claude"):
                    judgment_rows.append(
                        {
                            "task_id": task_id,
                            "model_name": model,
                            "label": "insufficient",
                            "confidence": 0.9,
                            "rationale": "not enough evidence",
                            "cited_passage_ids": [],
                            "status": "completed",
                        }
                    )
                adjudication_rows.append(
                    {
                        "task_id": task_id,
                        "event_id": "hongkong",
                        "claim_id": task_rows[-1]["claim_id"],
                        "claim_text": "claim",
                        "status": "agreement",
                        "canonical_label": "insufficient",
                        "labels_by_model": {"gemini": "insufficient", "claude": "insufficient"},
                        "needs_human_audit": False,
                    }
                )
            tasks.write_text("\n".join(json.dumps(row) for row in task_rows) + "\n", encoding="utf-8")
            judgments.write_text("\n".join(json.dumps(row) for row in judgment_rows) + "\n", encoding="utf-8")
            adjudication.write_text("\n".join(json.dumps(row) for row in adjudication_rows) + "\n", encoding="utf-8")

            summary = export_claim_grounding_audit(
                tasks_path=tasks,
                judgments_path=judgments,
                adjudication_path=adjudication,
                output_dir=output,
                hk_insufficient_sample_size=2,
                seed=7,
            )

            rows = [json.loads(line) for line in (output / "human_audit_queue.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["hongkong_insufficient_agreement_total"], 3)
            self.assertEqual(summary["hongkong_insufficient_agreement_sample"], 2)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("hongkong_insufficient_agreement_sample" in row["audit_flags"] for row in rows))

    def test_clean_human_evidence_text_trims_connector_fragment_and_pdf_footer(self) -> None:
        raw = (
            "  and  to \nan inclusive national political dialogue; \n\n"
            "4.  Notes  that  Ukraine  has  not  authorized  the  referendum on the status of\n\nCrimea.\n\n"
            "5.  Declares  that  all  Stat es  should not recognize it.\n\n"
            "2/2 \n\n14-26657 \n \n\f"
        )

        cleaned = clean_human_evidence_text(raw)

        self.assertTrue(cleaned.startswith("4. Notes"))
        self.assertIn("status of Crimea", cleaned)
        self.assertIn("5. Declares that all States should not recognize it.", cleaned)
        self.assertNotIn("2/2", cleaned)
        self.assertNotIn("14-26657", cleaned)
        self.assertNotIn("  ", cleaned)


if __name__ == "__main__":
    unittest.main()
