from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.direct_scoring import (  # noqa: E402
    build_direct_scoring_tasks,
    execute_direct_scoring,
    normalize_direct_scoring_response,
)


class DirectScoringE4Test(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
