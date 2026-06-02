from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.io import sha256_file
from geomosaic_hg.official_doc_parsing import page_range_for_span, parse_materialized_documents, split_passages
from geomosaic_hg.official_doc_qa import qa_parsed_official_docs
from organize_manual_materialized_docs import corrected_materialized_filename, organize, parse_materialized_filename


class OfficialDocParsingTest(unittest.TestCase):
    def test_manual_materialization_filename_corrections_remove_placeholder_x(self) -> None:
        jcpoa = corrected_materialized_filename("jcpoa__x__x__en__jcpoa-iaea-infcirc-887.pdf")
        scs = corrected_materialized_filename("scs__x__x__zh-Hans-CN__scs-fmprc-position-paper-jurisdiction-2014-12-07.txt")

        self.assertEqual(parse_materialized_filename(jcpoa)["viewpoint_origin"], "multilateral")
        self.assertEqual(parse_materialized_filename(jcpoa)["evidence_scope"], "anchor_document")
        self.assertEqual(parse_materialized_filename(scs)["viewpoint_origin"], "china")
        self.assertEqual(parse_materialized_filename(scs)["evidence_scope"], "legal_background")

    def test_organize_syncs_expected_filename_after_curation_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "manual"
            input_dir.mkdir()
            original_name = "jcpoa__x__x__en__jcpoa-iaea-infcirc-887.pdf"
            corrected_name = corrected_materialized_filename(original_name)
            (input_dir / original_name).write_bytes(b"%PDF-1.4\n% fixture\n")
            state = {
                "state": {
                    "task-1": {
                        "filename": original_name,
                        "work_status": "materialized",
                        "materialization_method": "manual_download",
                        "source_url": "https://example.test/doc.pdf",
                    }
                }
            }
            state_path = input_dir / "geomosaic_manual_materialization_state.json"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            output_dir = root / "organized"

            organize(input_dir, state_path, output_dir)

            rows = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["source_filename"], corrected_name)
            self.assertEqual(row["expected_source_filename"], corrected_name)
            self.assertEqual(row["state_source_filename"], original_name)
            self.assertEqual(row["actual_source_filename"], original_name)
            self.assertNotIn("__x__x__", row["source_filename"])

    def test_split_passages_preserves_offsets_and_overlap(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

        passages = split_passages(text, max_chars=24, overlap_chars=6)

        self.assertGreater(len(passages), 1)
        self.assertEqual(passages[0]["char_start"], 0)
        for passage in passages:
            self.assertEqual(text[passage["char_start"] : passage["char_end"]], passage["text"])
            self.assertLessEqual(len(passage["text"]), 30)
        self.assertLess(passages[1]["char_start"], passages[0]["char_end"])

    def test_parse_materialized_documents_reads_manifest_text_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            materialized = root / "official_doc_materialized"
            text_dir = materialized / "manual_text" / "crimea"
            text_dir.mkdir(parents=True)
            source = text_dir / "crimea__russia__anchor_document__en__fixture.txt"
            source.write_text("Title\n\nCrimea official statement text. " * 8, encoding="utf-8")
            source_zh = text_dir / "crimea__russia__anchor_document__zh__fixture.txt"
            source_zh.write_text("标题\n\n克里米亚官方声明文本。" * 8, encoding="utf-8")
            manifest = materialized / "manifest.jsonl"
            row = {
                "event_id": "crimea",
                "language": "en",
                "document_id": "fixture",
                "source_filename": source.name,
                "local_path": source.as_posix(),
                "output_kind": "manual_text",
                "organize_status": "copied",
                "materialization_method": "manual_copy_text",
                "sha256": "manual-fixture",
            }
            row_zh = {**row, "language": "zh", "source_filename": source_zh.name, "local_path": source_zh.as_posix()}
            missing = {
                **row,
                "document_id": "missing",
                "source_filename": "missing.txt",
                "local_path": "",
                "organize_status": "missing_source_file",
            }
            manifest.write_text(json.dumps(row) + "\n" + json.dumps(row_zh) + "\n" + json.dumps(missing) + "\n", encoding="utf-8")
            output = root / "parsed"

            summary = parse_materialized_documents(
                manifest,
                output,
                max_passage_chars=80,
                overlap_chars=10,
                min_ok_chars=20,
            )

            self.assertEqual(summary["documents_total"], 2)
            self.assertEqual(summary["documents_ok"], 2)
            self.assertEqual(summary["manifest_rows_skipped"], 1)
            self.assertEqual(summary["by_event"], {"crimea": 2})
            self.assertGreater(summary["total_char_count"], 0)
            self.assertEqual(len(summary["shortest_documents"]), 2)
            docs = [json.loads(line) for line in (output / "official_doc_text.jsonl").read_text(encoding="utf-8").splitlines()]
            passages = [json.loads(line) for line in (output / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0]["parser"], "manual_text")
            self.assertEqual(docs[0]["parse_quality"], "ok")
            self.assertIn("sha256", docs[0])
            self.assertNotIn("source_sha256", docs[0])
            self.assertNotIn("file_sha256", docs[0])
            self.assertNotIn("expected_source_filename", docs[0])
            self.assertNotIn("page_spans", docs[0])
            self.assertNotIn("text_sha256", docs[0])
            self.assertNotIn("parser_warning", docs[0])
            self.assertEqual(docs[0]["extra"]["expected_source_filename"], source.name)
            self.assertIn("page_spans", docs[0]["extra"])
            self.assertIn("text_sha256", docs[0]["extra"])
            self.assertGreater(len(passages), 1)
            self.assertEqual(len({p["passage_id"] for p in passages}), len(passages))
            for passage in passages:
                self.assertIn(f"_{passage['language']}_", passage["passage_id"])
                self.assertEqual(passage["page_start"], 1)
                self.assertEqual(passage["page_end"], 1)
                self.assertNotIn("text_sha256", passage)
                self.assertIn("text_sha256", passage["extra"])
            self.assertTrue((output / "parse_summary.json").exists())

    def test_page_range_for_span_maps_cross_page_offsets(self) -> None:
        page_spans = [
            {"page_number": 1, "char_start": 0, "char_end": 10},
            {"page_number": 2, "char_start": 10, "char_end": 30},
        ]

        self.assertEqual(page_range_for_span(page_spans, 2, 8), (1, 1))
        self.assertEqual(page_range_for_span(page_spans, 8, 12), (1, 2))

    def test_parse_official_docs_cli_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixture.txt"
            source.write_text("Official document body for parser CLI test.", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "event_id": "scs",
                        "language": "en",
                        "document_id": "cli-fixture",
                        "source_filename": source.name,
                        "local_path": source.as_posix(),
                        "output_kind": "manual_text",
                        "organize_status": "copied",
                        "materialization_method": "manual_copy_text",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "parsed"

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "script" / "parse_official_docs.py"),
                    "--manifest",
                    manifest.as_posix(),
                    "--output-dir",
                    output.as_posix(),
                    "--max-passage-chars",
                    "120",
                    "--overlap-chars",
                    "10",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"documents_total": 1', result.stdout)
            self.assertTrue((output / "official_doc_text.jsonl").exists())

    def test_qa_parsed_official_docs_reports_clean_outputs_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parsed = root / "parsed"
            parsed.mkdir()
            docs = [
                {
                    "document_id": "doc-a",
                    "event_id": "crimea",
                    "language": "en",
                    "source_filename": "crimea__multilateral__official_response__en__doc-a.txt",
                    "local_path": (root / "doc-a.txt").as_posix(),
                    "parser": "manual_text",
                    "text": "alpha beta gamma",
                    "char_count": 16,
                    "page_count": 1,
                    "parse_quality": "ok",
                    "sha256": "abc",
                    "extra": {
                        "expected_source_filename": "crimea__multilateral__official_response__en__doc-a.txt",
                        "text_sha256": "txt",
                        "page_spans": [{"page_number": 1, "char_start": 0, "char_end": 16}],
                    },
                },
                {
                    "document_id": "doc-a",
                    "event_id": "crimea",
                    "language": "fr",
                    "source_filename": "crimea__multilateral__official_response__fr__doc-a.txt",
                    "local_path": (root / "doc-a-fr.txt").as_posix(),
                    "parser": "manual_text",
                    "text": "alpha beta gamma",
                    "char_count": 16,
                    "page_count": 1,
                    "parse_quality": "ok",
                    "sha256": "def",
                    "extra": {
                        "expected_source_filename": "crimea__multilateral__official_response__fr__doc-a.txt",
                        "text_sha256": "txt-fr",
                        "page_spans": [{"page_number": 1, "char_start": 0, "char_end": 16}],
                    },
                },
            ]
            passages = [
                {
                    "passage_id": "passage_doc-a_en_0000",
                    "document_id": "doc-a",
                    "event_id": "crimea",
                    "language": "en",
                    "source_filename": docs[0]["source_filename"],
                    "page_start": 1,
                    "page_end": 1,
                    "passage_index": 0,
                    "char_start": 0,
                    "char_end": 16,
                    "text": "alpha beta gamma",
                    "extra": {"text_sha256": "p"},
                }
            ]
            (root / "doc-a.txt").write_text("alpha beta gamma", encoding="utf-8")
            (root / "doc-a-fr.txt").write_text("alpha beta gamma", encoding="utf-8")

            docs[0]["sha256"] = sha256_file(root / "doc-a.txt")
            docs[1]["sha256"] = sha256_file(root / "doc-a-fr.txt")
            (parsed / "official_doc_text.jsonl").write_text("\n".join(json.dumps(row) for row in docs) + "\n", encoding="utf-8")
            (parsed / "passages.jsonl").write_text("\n".join(json.dumps(row) for row in passages) + "\n", encoding="utf-8")
            (parsed / "parse_summary.json").write_text(
                json.dumps(
                    {
                        "documents_total": 2,
                        "passages_total": 1,
                        "documents_ok": 2,
                        "documents_low_text": 0,
                        "documents_failed": 0,
                    }
                ),
                encoding="utf-8",
            )

            report = qa_parsed_official_docs(parsed, output_path=parsed / "parse_qa_report.json", short_doc_chars=10)

            self.assertTrue(report["ok"])
            self.assertEqual(report["counts"]["documents"], 2)
            self.assertEqual(report["counts"]["passages"], 1)
            self.assertEqual(report["integrity"]["duplicate_passage_ids"], [])
            self.assertEqual(report["integrity"]["orphan_passages"], [])
            self.assertEqual(report["coverage"]["document_language_groups"]["doc-a"], ["en", "fr"])
            self.assertEqual(report["coverage"]["event_language_counts"]["crimea"], {"en": 1, "fr": 1})
            self.assertEqual(report["integrity"]["sha256_mismatches"], [])
            self.assertEqual(report["integrity"]["page_range_mismatches"], [])
            self.assertEqual(report["integrity"]["invalid_passage_id_formats"], [])
            self.assertTrue((parsed / "parse_qa_report.json").exists())


if __name__ == "__main__":
    unittest.main()
