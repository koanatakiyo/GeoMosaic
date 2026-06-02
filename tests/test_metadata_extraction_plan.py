from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.metadata_extraction_plan import plan_metadata_extraction


class MetadataExtractionPlanTest(unittest.TestCase):
    def test_dry_run_plans_official_docs_and_skips_pointer_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            external_assets = root / "external_assets.jsonl"
            output_dir = root / "metadata_plan"

            docs = [
                {
                    "document_id": "official-doc-a",
                    "event_id": "scs",
                    "language": "en",
                    "source_filename": "scs__legal_international__anchor_document__en__official-doc-a.pdf",
                    "local_path": "data/0_external/official_doc_materialized/files/scs/official-doc-a.pdf",
                    "parser": "pdfminer.six",
                    "text": "South China Sea award text",
                    "char_count": 26,
                    "page_count": 2,
                    "parse_quality": "ok",
                    "sha256": "file-hash",
                    "extra": {
                        "text_sha256": "text-hash",
                        "page_spans": [{"page_number": 1, "char_start": 0, "char_end": 26}],
                    },
                },
                {
                    "document_id": "short-failed",
                    "event_id": "scs",
                    "language": "en",
                    "source_filename": "failed.pdf",
                    "local_path": "",
                    "parser": "unknown",
                    "text": "",
                    "char_count": 0,
                    "page_count": None,
                    "parse_quality": "failed",
                    "sha256": "",
                    "extra": {"text_sha256": "", "page_spans": []},
                },
            ]
            passages = [
                {
                    "passage_id": "passage_official-doc-a_en_0000",
                    "document_id": "official-doc-a",
                    "event_id": "scs",
                    "language": "en",
                    "source_filename": docs[0]["source_filename"],
                    "page_start": 1,
                    "page_end": 1,
                    "passage_index": 0,
                    "char_start": 0,
                    "char_end": 26,
                    "text": "South China Sea award text",
                    "extra": {"text_sha256": "passage-hash"},
                }
            ]
            external_rows = [
                {
                    "asset_id": "asset_gdelt_article",
                    "event_id": "scs",
                    "modality": "text",
                    "asset_source": "GDELT_DOC",
                    "extra": {"record_type": "news_pointer", "source_temporal_coverage": "retrospective_context"},
                },
                {
                    "asset_id": "asset_gdelt_image",
                    "event_id": "scs",
                    "modality": "image_restricted_pointer",
                    "asset_source": "GDELT_DOC",
                    "extra": {"record_type": "image_restricted_pointer", "source_temporal_coverage": "retrospective_context"},
                },
            ]
            (parsed / "official_doc_text.jsonl").write_text("\n".join(json.dumps(row) for row in docs) + "\n", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("\n".join(json.dumps(row) for row in passages) + "\n", encoding="utf-8")
            external_assets.write_text("\n".join(json.dumps(row) for row in external_rows) + "\n", encoding="utf-8")

            summary = plan_metadata_extraction(
                parsed_dir=parsed,
                external_assets_path=external_assets,
                output_dir=output_dir,
                dry_run=True,
                model_id="gemini-2.5-flash",
                max_full_text_chars=100,
            )

            tasks = [
                json.loads(line)
                for line in (output_dir / "metadata_extraction_tasks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["tasks_total"], 1)
            self.assertEqual(summary["documents_skipped_by_quality"], {"failed": 1})
            self.assertEqual(summary["pointer_assets_skipped"], 2)
            self.assertEqual(summary["pointer_assets_by_record_type"], {"image_restricted_pointer": 1, "news_pointer": 1})
            self.assertEqual(tasks[0]["task_type"], "official_doc_metadata_extraction")
            self.assertEqual(tasks[0]["load_bearing"], False)
            self.assertEqual(tasks[0]["model_id"], "gemini-2.5-flash")
            self.assertEqual(tasks[0]["input_strategy"], "document_full_text")
            self.assertEqual(tasks[0]["fields_planned"], ["summary", "actors", "dates", "language_note", "section_outline", "candidate_passage_hints"])
            self.assertEqual(tasks[0]["passage_count"], 1)
            self.assertEqual(tasks[0]["pointer_extraction_policy"], "skip_pointers")
            self.assertTrue((output_dir / "metadata_extraction_dry_run_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
