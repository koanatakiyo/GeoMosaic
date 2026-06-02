from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.metadata_extraction import (
    base_prompt,
    execute_metadata_extraction,
    finalize_failed_map_reduce_from_batches,
    normalize_metadata_response,
)


def fixture_response(label: str = "fixture") -> dict:
    return {
        "summary": f"{label} summary",
        "actors": ["Actor A", "Actor B"],
        "dates": ["2020-01-01"],
        "language_note": "English sample.",
        "section_outline": ["Opening", "Operative text"],
        "candidate_passage_hints": ["territorial claim", "official response"],
        "document_position": "supports",
        "claim_relation": "support",
    }


class MetadataExtractionExecutionTest(unittest.TestCase):
    def test_compact_prompt_constraints_are_versioned(self) -> None:
        v0_prompt = "\n".join(base_prompt({"prompt_version": "official_doc_metadata_v0"}))
        compact_prompt = "\n".join(base_prompt({"prompt_version": "official_doc_metadata_v1_compact"}))

        self.assertNotIn("Length constraints", v0_prompt)
        self.assertIn("Length constraints", compact_prompt)

    def test_normalizes_fenced_json_and_skips_null_list_items(self) -> None:
        metadata = normalize_metadata_response(
            '```json{"summary":"ok","actors":["Actor A",null,""],"dates":[null,"2020-01-01"],'
            '"language_note":"note","section_outline":[],"candidate_passage_hints":[null,"hint"],'
            '"document_position":"supports"}```'
        )

        self.assertEqual(metadata["summary"], "ok")
        self.assertEqual(metadata["actors"], ["Actor A"])
        self.assertEqual(metadata["dates"], ["2020-01-01"])
        self.assertEqual(metadata["candidate_passage_hints"], ["hint"])
        self.assertNotIn("document_position", metadata)

    def test_executes_full_text_task_without_load_bearing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            output = root / "official_doc_metadata_enrichment.jsonl"
            batch_output = root / "official_doc_metadata_batch_outputs.jsonl"
            summary_output = root / "summary.json"
            tasks_path = root / "tasks.jsonl"

            doc = {
                "document_id": "doc-a",
                "event_id": "crimea",
                "language": "en",
                "source_filename": "crimea__russia__anchor_document__en__doc-a.txt",
                "text": "Crimea official document text.",
                "char_count": 30,
                "page_count": 1,
                "parse_quality": "ok",
                "extra": {"text_sha256": "doc-text-hash"},
            }
            passage = {
                "passage_id": "passage_doc-a_en_0000",
                "document_id": "doc-a",
                "event_id": "crimea",
                "language": "en",
                "source_filename": doc["source_filename"],
                "passage_index": 0,
                "page_start": 1,
                "page_end": 1,
                "char_start": 0,
                "char_end": 30,
                "text": doc["text"],
                "extra": {"text_sha256": "passage-hash"},
            }
            task = {
                "task_id": "metadata_extract_crimea_doc-a_en_x",
                "task_type": "official_doc_metadata_extraction",
                "event_id": "crimea",
                "document_id": "doc-a",
                "language": "en",
                "source_filename": doc["source_filename"],
                "input_strategy": "document_full_text",
                "model_id": "gemini-2.5-flash",
                "prompt_version": "official_doc_metadata_v0",
                "schema_version": "official_doc_metadata_v0",
                "input_text_sha256": "doc-text-hash",
                "load_bearing": False,
            }
            (parsed / "official_doc_text.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
            (parsed / "passages.jsonl").write_text(json.dumps(passage) + "\n", encoding="utf-8")
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            calls: list[str] = []

            def fake_caller(prompt: str, model_id: str) -> dict:
                calls.append(prompt)
                return fixture_response("full")

            summary = execute_metadata_extraction(
                tasks_path=tasks_path,
                parsed_dir=parsed,
                output_path=output,
                batch_output_path=batch_output,
                summary_path=summary_output,
                model_caller=fake_caller,
                resume=False,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["llm_calls_made"], 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(rows[0]["task_id"], task["task_id"])
            self.assertEqual(rows[0]["metadata"]["summary"], "full summary")
            self.assertFalse(rows[0]["load_bearing"])
            self.assertTrue(rows[0]["stage_c_excluded"])
            self.assertEqual(rows[0]["stage_c_policy"], "never_use_stage_b_outputs")
            self.assertNotIn("document_position", rows[0]["metadata"])
            self.assertEqual(rows[0]["status"], "completed")

    def test_executes_long_doc_as_map_reduce_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            output = root / "official_doc_metadata_enrichment.jsonl"
            batch_output = root / "official_doc_metadata_batch_outputs.jsonl"
            summary_output = root / "summary.json"
            tasks_path = root / "tasks.jsonl"

            doc = {
                "document_id": "long-doc",
                "event_id": "scs",
                "language": "en",
                "source_filename": "scs__legal_international__anchor_document__en__long-doc.pdf",
                "text": "unused long text",
                "char_count": 100000,
                "page_count": 10,
                "parse_quality": "ok",
                "extra": {"text_sha256": "long-hash"},
            }
            passages = []
            for idx in range(3):
                passages.append(
                    {
                        "passage_id": f"passage_long-doc_en_{idx:04d}",
                        "document_id": "long-doc",
                        "event_id": "scs",
                        "language": "en",
                        "source_filename": doc["source_filename"],
                        "passage_index": idx,
                        "page_start": idx + 1,
                        "page_end": idx + 1,
                        "char_start": idx * 100,
                        "char_end": idx * 100 + 50,
                        "text": f"passage text {idx}",
                        "extra": {"text_sha256": f"p{idx}"},
                    }
                )
            task = {
                "task_id": "metadata_extract_scs_long-doc_en_x",
                "task_type": "official_doc_metadata_extraction",
                "event_id": "scs",
                "document_id": "long-doc",
                "language": "en",
                "source_filename": doc["source_filename"],
                "input_strategy": "passage_batch_map_reduce",
                "batch_passages": 2,
                "model_id": "gemini-2.5-flash",
                "prompt_version": "official_doc_metadata_v0",
                "schema_version": "official_doc_metadata_v0",
                "input_text_sha256": "long-hash",
                "load_bearing": False,
            }
            (parsed / "official_doc_text.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("\n".join(json.dumps(row) for row in passages) + "\n", encoding="utf-8")
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            calls: list[str] = []

            def fake_caller(prompt: str, model_id: str) -> dict:
                calls.append(prompt)
                if "Reduce these batch-level metadata extractions" in prompt:
                    return fixture_response("reduced")
                return fixture_response("batch")

            summary = execute_metadata_extraction(
                tasks_path=tasks_path,
                parsed_dir=parsed,
                output_path=output,
                batch_output_path=batch_output,
                summary_path=summary_output,
                model_caller=fake_caller,
                resume=False,
            )

            final_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            batch_rows = [json.loads(line) for line in batch_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["llm_calls_made"], 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(batch_rows), 2)
            self.assertEqual(final_rows[0]["metadata"]["summary"], "reduced summary")
            self.assertEqual(final_rows[0]["input_strategy"], "passage_batch_map_reduce")
            self.assertEqual(final_rows[0]["batch_count"], 2)
            self.assertFalse(final_rows[0]["load_bearing"])

    def test_resume_skips_existing_completed_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            output = root / "official_doc_metadata_enrichment.jsonl"
            batch_output = root / "official_doc_metadata_batch_outputs.jsonl"
            summary_output = root / "summary.json"
            tasks_path = root / "tasks.jsonl"
            task = {
                "task_id": "metadata_extract_existing",
                "event_id": "crimea",
                "document_id": "doc-a",
                "language": "en",
                "source_filename": "doc-a.txt",
                "input_strategy": "document_full_text",
                "model_id": "gemini-2.5-flash",
            }
            existing = {
                "task_id": "metadata_extract_existing",
                "status": "completed",
                "metadata": fixture_response("existing"),
                "load_bearing": False,
            }
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            (parsed / "official_doc_text.jsonl").write_text("", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("", encoding="utf-8")
            output.write_text(json.dumps(existing) + "\n", encoding="utf-8")
            calls: list[str] = []

            def fake_caller(prompt: str, model_id: str) -> dict:
                calls.append(prompt)
                return fixture_response("new")

            summary = execute_metadata_extraction(
                tasks_path=tasks_path,
                parsed_dir=parsed,
                output_path=output,
                batch_output_path=batch_output,
                summary_path=summary_output,
                model_caller=fake_caller,
                resume=True,
            )

            self.assertEqual(summary["skipped_existing"], 1)
            self.assertEqual(summary["llm_calls_made"], 0)
            self.assertEqual(calls, [])

    def test_missing_doc_records_friendly_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            output = root / "official_doc_metadata_enrichment.jsonl"
            batch_output = root / "official_doc_metadata_batch_outputs.jsonl"
            summary_output = root / "summary.json"
            tasks_path = root / "tasks.jsonl"
            task = {
                "task_id": "metadata_extract_missing",
                "event_id": "crimea",
                "document_id": "missing-doc",
                "language": "en",
                "source_filename": "missing-doc.txt",
                "input_strategy": "document_full_text",
                "model_id": "gemini-2.5-flash",
            }
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            (parsed / "official_doc_text.jsonl").write_text("", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("", encoding="utf-8")

            def fake_caller(prompt: str, model_id: str) -> dict:
                raise AssertionError("model should not be called")

            summary = execute_metadata_extraction(
                tasks_path=tasks_path,
                parsed_dir=parsed,
                output_path=output,
                batch_output_path=batch_output,
                summary_path=summary_output,
                model_caller=fake_caller,
                resume=False,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["failed"], 1)
            self.assertIn("No parsed official document found", rows[0]["error"])
            self.assertEqual(rows[0]["status"], "failed")

    def test_malformed_json_is_retried_and_old_failed_row_compacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            output = root / "official_doc_metadata_enrichment.jsonl"
            batch_output = root / "official_doc_metadata_batch_outputs.jsonl"
            summary_output = root / "summary.json"
            tasks_path = root / "tasks.jsonl"
            doc = {
                "document_id": "doc-a",
                "event_id": "crimea",
                "language": "en",
                "source_filename": "doc-a.txt",
                "text": "Crimea official document text.",
                "char_count": 30,
                "page_count": 1,
                "parse_quality": "ok",
                "extra": {"text_sha256": "doc-text-hash"},
            }
            task = {
                "task_id": "metadata_extract_retry",
                "event_id": "crimea",
                "document_id": "doc-a",
                "language": "en",
                "source_filename": "doc-a.txt",
                "input_strategy": "document_full_text",
                "model_id": "gemini-2.5-flash",
            }
            old_failed = {
                "task_id": "metadata_extract_retry",
                "status": "failed",
                "error": "old failure",
                "load_bearing": False,
            }
            (parsed / "official_doc_text.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("", encoding="utf-8")
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            output.write_text(json.dumps(old_failed) + "\n", encoding="utf-8")
            calls: list[str] = []

            def fake_caller(prompt: str, model_id: str) -> str:
                calls.append(prompt)
                if len(calls) == 1:
                    return '{"summary": "unterminated'
                return json.dumps(fixture_response("retry"))

            summary = execute_metadata_extraction(
                tasks_path=tasks_path,
                parsed_dir=parsed,
                output_path=output,
                batch_output_path=batch_output,
                summary_path=summary_output,
                model_caller=fake_caller,
                resume=True,
                parse_retries=1,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["llm_calls_made"], 2)
            self.assertEqual(summary["output_rows_after_compaction"], 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["metadata"]["summary"], "retry summary")

    def test_compaction_keeps_existing_completed_when_retry_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            output = root / "official_doc_metadata_enrichment.jsonl"
            batch_output = root / "official_doc_metadata_batch_outputs.jsonl"
            summary_output = root / "summary.json"
            tasks_path = root / "tasks.jsonl"
            doc = {
                "document_id": "doc-a",
                "event_id": "crimea",
                "language": "en",
                "source_filename": "doc-a.txt",
                "text": "Crimea official document text.",
                "char_count": 30,
                "page_count": 1,
                "parse_quality": "ok",
                "extra": {"text_sha256": "doc-text-hash"},
            }
            task = {
                "task_id": "metadata_extract_retry",
                "event_id": "crimea",
                "document_id": "doc-a",
                "language": "en",
                "source_filename": "doc-a.txt",
                "input_strategy": "document_full_text",
                "model_id": "gemini-2.5-flash",
            }
            old_completed = {
                "task_id": "metadata_extract_retry",
                "status": "completed",
                "metadata": fixture_response("existing"),
                "load_bearing": False,
            }
            (parsed / "official_doc_text.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("", encoding="utf-8")
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            output.write_text(json.dumps(old_completed) + "\n", encoding="utf-8")

            def fake_caller(prompt: str, model_id: str) -> str:
                return '{"summary": "still malformed'

            summary = execute_metadata_extraction(
                tasks_path=tasks_path,
                parsed_dir=parsed,
                output_path=output,
                batch_output_path=batch_output,
                summary_path=summary_output,
                model_caller=fake_caller,
                resume=False,
                parse_retries=0,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["output_rows_after_compaction"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["metadata"]["summary"], "existing summary")

    def test_finalize_failed_map_reduce_from_completed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_path = root / "tasks.jsonl"
            output = root / "official_doc_metadata_enrichment.jsonl"
            batch_output = root / "official_doc_metadata_batch_outputs.jsonl"
            summary_output = root / "finalize_summary.json"
            task = {
                "task_id": "metadata_extract_long",
                "task_type": "official_doc_metadata_extraction",
                "event_id": "scs",
                "document_id": "long-doc",
                "language": "en",
                "source_filename": "long-doc.pdf",
                "input_strategy": "passage_batch_map_reduce",
                "model_id": "gemini-2.5-flash",
                "prompt_version": "official_doc_metadata_v0",
                "schema_version": "official_doc_metadata_v0",
                "input_text_sha256": "hash",
            }
            failed = {
                "task_id": "metadata_extract_long",
                "event_id": "scs",
                "document_id": "long-doc",
                "language": "en",
                "source_filename": "long-doc.pdf",
                "input_strategy": "passage_batch_map_reduce",
                "status": "failed",
                "error": "reduce failed",
            }
            batches = [
                {
                    "task_id": "metadata_extract_long",
                    "batch_id": "metadata_extract_long_batch_0000",
                    "batch_index": 0,
                    "batch_total": 2,
                    "llm_calls": 1,
                    "metadata": {
                        "summary": "batch zero",
                        "actors": ["Actor A"],
                        "dates": ["2016-07-12"],
                        "language_note": "English.",
                        "section_outline": ["Part I"],
                        "candidate_passage_hints": ["award"],
                    },
                },
                {
                    "task_id": "metadata_extract_long",
                    "batch_id": "metadata_extract_long_batch_0001",
                    "batch_index": 1,
                    "batch_total": 2,
                    "llm_calls": 2,
                    "metadata": {
                        "summary": "batch one",
                        "actors": ["Actor A", "Actor B"],
                        "dates": ["2016-07-12", "2016-07-13"],
                        "language_note": "English.",
                        "section_outline": ["Part II"],
                        "candidate_passage_hints": ["tribunal"],
                    },
                },
            ]
            tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            output.write_text(json.dumps(failed) + "\n", encoding="utf-8")
            batch_output.write_text("\n".join(json.dumps(row) for row in batches) + "\n", encoding="utf-8")

            summary = finalize_failed_map_reduce_from_batches(
                tasks_path=tasks_path,
                output_path=output,
                batch_output_path=batch_output,
                summary_path=summary_output,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["finalized"], 1)
            self.assertEqual(summary["remaining_failed"], 0)
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["reduce_strategy"], "deterministic_batch_merge_fallback")
            self.assertEqual(rows[0]["llm_calls"], 3)
            self.assertEqual(rows[0]["metadata"]["actors"], ["Actor A", "Actor B"])


if __name__ == "__main__":
    unittest.main()
