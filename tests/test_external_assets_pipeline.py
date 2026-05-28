from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.external_assets import (
    acled_result_warnings,
    build_external_asset_plan,
    collect_existing_external_assets,
    fetch_wikimedia_for_event,
    run_external_asset_collection,
)
from geomosaic_hg.build import build_source_asset_links, load_external_assets
from geomosaic_hg.build import match_level_for_hyperedge_asset, select_assets_for_hyperedge
import geomosaic_hg.clients.acled as acled_module
from geomosaic_hg.clients.acled import ACLEDClient, ACLEDCredentials
from geomosaic_hg.clients.http import HTTPClientError
from geomosaic_hg.clients.wikimedia import WikimediaCommonsClient, WikimediaFile, wikimedia_file_to_asset
from geomosaic_hg.schema import EvidenceAsset, SourceRecord
from geomosaic_hg.validation import validate_core_tables
from collect_external_assets import load_env_file, load_prior_summary_issues
from fetch_official_docs import _build_asset as build_official_doc_asset


class ExternalAssetsPipelineTest(unittest.TestCase):
    def test_external_plan_covers_all_tier1_events(self) -> None:
        plan = build_external_asset_plan()
        self.assertEqual(len(plan["events"]), 8)
        self.assertEqual(plan["events"][0]["event_id"], "crimea")
        for event in plan["events"]:
            self.assertTrue(event["wikimedia_image_query"])
            self.assertTrue(event["wikimedia_map_query"])
            self.assertTrue(event["acled_country"] or event["event_id"] == "scs")
        scs = [event for event in plan["events"] if event["event_id"] == "scs"][0]
        self.assertTrue(scs["acled_skip_reason"])
        self.assertEqual(scs["wikimedia_page_title"], "South China Sea Arbitration")
        self.assertIn("Territorial disputes in the South China Sea", scs["wikimedia_fallback_page_titles"])

    def test_external_plan_can_expand_acled_date_window(self) -> None:
        plan = build_external_asset_plan(["ukraine"], acled_window_days=7)
        event = plan["events"][0]
        self.assertEqual(event["acled_start_date"], "2022-02-17")
        self.assertEqual(event["acled_end_date"], "2022-03-03")

    def test_collect_existing_external_assets_writes_merged_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_dir = root / "inputs"
            event_dir.mkdir()
            wikimedia_asset = {
                "asset_id": "asset_wikimedia_crimea_fixture",
                "event_id": "crimea",
                "modality": "image_full",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2014-03-16T00:00:00Z",
                "observed_time": "2014-03-16T00:00:00Z",
                "geo_location": "Crimea, Ukraine",
                "url_or_pointer": "https://commons.wikimedia.org/wiki/File:Fixture.png",
                "caption_or_transcript": "Fixture image.",
                "license_or_terms": "CC-BY",
                "redistribution_flag": True,
                "perceptual_hash": "fixturehash",
                "embedding_id": "emb_asset_wikimedia_crimea_fixture",
                "extracted_entities": ["Crimea"],
                "extracted_claims": [],
                "evidence_role": "complementary",
                "extra": {"collection_channel": "wikipedia_page_bound", "page_bound": True},
            }
            acled_asset = {
                "asset_id": "asset_acled_crimea_fixture",
                "event_id": "crimea",
                "modality": "structured_event",
                "asset_source": "ACLED",
                "source_layer": "structured",
                "viewpoint_origin": "all",
                "publish_time": "2014-03-16T00:00:00Z",
                "observed_time": "2014-03-16T00:00:00Z",
                "geo_location": "Crimea, Ukraine",
                "url_or_pointer": "acled://event/fixture",
                "caption_or_transcript": "Fixture ACLED row.",
                "license_or_terms": "ACLED API terms apply",
                "redistribution_flag": False,
                "perceptual_hash": "acledhash",
                "embedding_id": "emb_asset_acled_crimea_fixture",
                "extracted_entities": ["Crimea"],
                "extracted_claims": [],
                "evidence_role": "context",
            }
            (event_dir / "wikimedia_crimea.jsonl").write_text(json.dumps(wikimedia_asset) + "\n", encoding="utf-8")
            (event_dir / "acled_crimea.jsonl").write_text(json.dumps(acled_asset) + "\n", encoding="utf-8")

            merged = root / "collect_external_assets_assets.jsonl"
            prior_errors = [{"event_id": "crimea", "source": "wikimedia", "error": "fixture"}]
            prior_warnings = [
                {"event_id": "crimea", "source": "wikimedia", "warning": "stale fixture"},
                {"event_id": "crimea", "source": "manual", "warning": "keep fixture"},
            ]
            summary = collect_existing_external_assets(event_dir, merged, {"crimea"}, prior_errors=prior_errors, prior_warnings=prior_warnings)

            self.assertEqual(summary["counts"]["merged_assets"], 2)
            summary_path = merged.with_suffix(".summary.json")
            self.assertTrue(summary_path.exists())
            summary_row = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_row["counts"]["merged_assets"], 2)
            self.assertEqual(summary_row["errors"], prior_errors)
            self.assertEqual(summary_row["warnings"], [{"event_id": "crimea", "source": "manual", "warning": "keep fixture"}])
            rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["asset_id"] for row in rows}, {"asset_wikimedia_crimea_fixture", "asset_acled_crimea_fixture"})
            wiki = [row for row in rows if row["asset_id"] == "asset_wikimedia_crimea_fixture"][0]
            acled = [row for row in rows if row["asset_id"] == "asset_acled_crimea_fixture"][0]
            self.assertEqual(wiki["extra"]["collection_channel"], "wikipedia_page_bound")
            self.assertEqual(wiki["extra"]["record_type"], "wiki_page_asset")
            self.assertEqual(wiki["extra"]["curation_level"], "community_curated")
            self.assertEqual(wiki["extra"]["active_policy"], "primary_image_evidence")
            self.assertEqual(acled["extra"]["collection_channel"], "acled_api")
            self.assertEqual(acled["extra"]["record_type"], "curated_conflict_event")
            self.assertEqual(acled["extra"]["curation_level"], "human_curated")
            self.assertEqual(acled["extra"]["active_policy"], "optional_enrichment")
            tables = {
                "source_records": [],
                "evidence_assets": rows,
                "source_asset_links": [],
                "claim_evidence_hyperedges": [],
            }
            errors = validate_core_tables(tables)
            self.assertFalse([error for error in errors if error.startswith("evidence_assets:")])

    def test_collect_existing_writes_candidate_inventory_and_active_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            merged = root / "merged.jsonl"
            active = {
                "asset_id": "asset_wikimedia_kosovo_active",
                "event_id": "kosovo",
                "modality": "image_full",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2008-02-17T00:00:00Z",
                "observed_time": "2008-02-17T00:00:00Z",
                "geo_location": "Kosovo",
                "url_or_pointer": "https://upload.wikimedia.org/example/active.jpg",
                "caption_or_transcript": "Kosovo independence celebration.",
                "license_or_terms": "CC BY-SA",
                "redistribution_flag": True,
                "perceptual_hash": "activehash",
                "embedding_id": "emb_active",
                "extracted_entities": ["Kosovo"],
                "extracted_claims": [],
                "evidence_role": "complementary",
                "extra": {"mime": "image/jpeg", "active_bench": True},
            }
            inactive = {
                **active,
                "asset_id": "asset_wikimedia_kosovo_inactive",
                "url_or_pointer": "https://upload.wikimedia.org/example/inactive.jpg",
                "perceptual_hash": "inactivehash",
                "embedding_id": "emb_inactive",
                "extra": {"mime": "image/jpeg", "active_bench": False, "active_status": "active"},
            }
            (raw / "wikimedia_kosovo.jsonl").write_text(json.dumps(active) + "\n" + json.dumps(inactive) + "\n", encoding="utf-8")

            summary = collect_existing_external_assets(raw, merged, {"kosovo"})

            merged_rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["asset_id"] for row in merged_rows], ["asset_wikimedia_kosovo_active"])
            self.assertEqual(summary["counts"]["candidate_assets"], 2)
            self.assertEqual(summary["counts"]["active_assets"], 1)
            inventory_path = root / "candidate_inventory.jsonl"
            decisions_path = root / "selection_decisions.jsonl"
            self.assertEqual(summary["candidate_inventory"], str(inventory_path))
            self.assertEqual(summary["selection_decisions"], str(decisions_path))
            inventory_rows = [json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines()]
            decision_rows = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["asset_id"] for row in inventory_rows}, {"asset_wikimedia_kosovo_active", "asset_wikimedia_kosovo_inactive"})
            self.assertEqual({row["active_status"] for row in decision_rows}, {"active", "candidate"})
            inactive_decision = [row for row in decision_rows if row["asset_id"] == "asset_wikimedia_kosovo_inactive"][0]
            inactive_inventory = [row for row in inventory_rows if row["asset_id"] == "asset_wikimedia_kosovo_inactive"][0]
            self.assertEqual(inactive_inventory["extra"]["active_status"], "candidate")
            self.assertIn("active_bench=false", inactive_decision["selection_reason"])

    def test_load_external_assets_ignores_candidate_and_decision_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = {
                "asset_id": "asset_wikimedia_kosovo_active",
                "event_id": "kosovo",
                "modality": "image_full",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2008-02-17T00:00:00Z",
                "observed_time": "2008-02-17T00:00:00Z",
                "geo_location": "Kosovo",
                "url_or_pointer": "https://upload.wikimedia.org/example/active.jpg",
                "caption_or_transcript": "Kosovo independence celebration.",
                "license_or_terms": "CC BY-SA",
                "redistribution_flag": True,
                "perceptual_hash": "activehash",
                "embedding_id": "emb_active",
                "extracted_entities": ["Kosovo"],
                "extracted_claims": [],
                "evidence_role": "complementary",
            }
            (root / "external_assets.jsonl").write_text(json.dumps(asset) + "\n", encoding="utf-8")
            (root / "candidate_inventory.jsonl").write_text(json.dumps({**asset, "asset_id": "asset_candidate"}) + "\n", encoding="utf-8")
            (root / "selection_decisions.jsonl").write_text(json.dumps({"asset_id": "asset_candidate", "event_id": "kosovo", "selected": False}) + "\n", encoding="utf-8")

            assets = load_external_assets(root, {"kosovo"})

            self.assertEqual([asset.asset_id for asset in assets], ["asset_wikimedia_kosovo_active"])

    def test_summarize_external_assets_reports_source_coverage_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "external_assets.jsonl"
            rows = [
                {
                    "asset_id": "asset_official_scs_1",
                    "event_id": "scs",
                    "asset_source": "OFFICIAL_DOC",
                    "modality": "structured_document",
                    "source_layer": "official",
                    "extra": {
                        "record_type": "arbitration_award",
                        "collection_channel": "official_registry",
                        "active_policy": "primary_official_evidence",
                    },
                },
                {
                    "asset_id": "asset_gdelt_ukraine_1",
                    "event_id": "ukraine",
                    "asset_source": "GDELT_DOC",
                    "modality": "text",
                    "source_layer": "news",
                    "extra": {
                        "record_type": "news_pointer",
                        "collection_channel": "gdelt_doc_search",
                        "active_policy": "pointer_enrichment",
                    },
                },
                {
                    "asset_id": "asset_gdelt_ukraine_2",
                    "event_id": "ukraine",
                    "asset_source": "GDELT_DOC",
                    "modality": "text",
                    "source_layer": "news",
                    "extra": {
                        "record_type": "news_pointer",
                        "collection_channel": "gdelt_doc_search",
                        "active_policy": "pointer_enrichment",
                    },
                },
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                ["python", "script/summarize_external_assets.py", "--input", str(input_path)],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["total_assets"], 3)
            self.assertEqual(summary["by_event"], {"scs": 1, "ukraine": 2})
            self.assertEqual(summary["by_source"], {"GDELT_DOC": 2, "OFFICIAL_DOC": 1})
            self.assertIn(
                {"event_id": "ukraine", "asset_source": "GDELT_DOC", "record_type": "news_pointer", "count": 2},
                summary["coverage_matrix"],
            )
            self.assertIn(
                {"event_id": "scs", "asset_source": "OFFICIAL_DOC", "record_type": "arbitration_award", "count": 1},
                summary["coverage_matrix"],
            )

    def test_summarize_external_assets_can_write_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "external_assets.jsonl"
            output_path = root / "reports" / "summary.json"
            row = {
                "asset_id": "asset_gdelt_hongkong_1",
                "event_id": "hongkong",
                "asset_source": "GDELT_DOC",
                "modality": "text",
                "source_layer": "news",
                "extra": {
                    "record_type": "news_pointer",
                    "collection_channel": "gdelt_doc_search",
                    "active_policy": "pointer_enrichment",
                },
            }
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    "script/summarize_external_assets.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            stdout_summary = json.loads(result.stdout)
            file_summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_summary, file_summary)
            self.assertEqual(file_summary["total_assets"], 1)

    def test_official_doc_asset_has_provenance_hash_and_adapter_metadata(self) -> None:
        row = build_official_doc_asset(
            "scs",
            {
                "title": "Fixture award",
                "caption": "Fixture award text.",
                "institution": "Permanent Court of Arbitration",
                "document_type": "arbitration_award",
                "publish_time": "2016-07-12T00:00:00Z",
                "url": "https://example.test/award.pdf",
                "viewpoint_origin": "legal_international",
                "entities": ["South China Sea", "China", "Philippines"],
                "geo_location": "The Hague, Netherlands",
                "license_or_terms": "public document",
            },
            "2026-05-24T00:00:00Z",
        )

        self.assertEqual(row["asset_source"], "OFFICIAL_DOC")
        self.assertEqual(row["modality"], "structured_document")
        self.assertEqual(row["extra"]["collection_channel"], "official_registry")
        self.assertEqual(row["extra"]["curation_level"], "official")
        self.assertEqual(row["extra"]["active_policy"], "primary_official_evidence")
        self.assertTrue(row["perceptual_hash"])

    def test_official_doc_catalog_only_approved_rows_become_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "official_candidates.jsonl"
            output_dir = root / "raw"
            approved = {
                "event_id": "scs",
                "review_status": "approved",
                "reviewer": "fixture-reviewer",
                "candidate_source": "semi_auto_fixture",
                "title": "China MFA Statement on South China Sea Arbitration",
                "caption": "Official statement responding to the South China Sea arbitration award.",
                "institution": "Ministry of Foreign Affairs of the People's Republic of China",
                "document_type": "official_statement",
                "publish_time": "2016-07-12T00:00:00Z",
                "url": "https://example.test/china-mfa-scs",
                "viewpoint_origin": "china",
                "entities": ["China", "South China Sea", "Permanent Court of Arbitration"],
                "geo_location": "Beijing, China",
                "license_or_terms": "Official public document",
            }
            pending = {
                **approved,
                "review_status": "pending",
                "title": "Pending fixture should not be emitted",
                "url": "https://example.test/pending",
            }
            catalog.write_text(json.dumps(approved) + "\n" + json.dumps(pending) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    "script/fetch_official_docs.py",
                    "--catalog",
                    str(catalog),
                    "--no-builtin-catalog",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in (output_dir / "official_scs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["url_or_pointer"], "https://example.test/china-mfa-scs")
            self.assertEqual(rows[0]["extra"]["review_status"], "approved")
            self.assertEqual(rows[0]["extra"]["reviewer"], "fixture-reviewer")
            self.assertEqual(rows[0]["extra"]["candidate_source"], "semi_auto_fixture")

    def test_official_doc_catalog_can_add_docs_for_any_registered_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "official_candidates.jsonl"
            output_dir = root / "raw"
            row = {
                "event_id": "crimea",
                "review_status": "approved",
                "title": "UN General Assembly Resolution 68/262",
                "caption": "Official UN General Assembly resolution on the territorial integrity of Ukraine.",
                "institution": "UN General Assembly",
                "document_type": "official_resolution",
                "publish_time": "2014-03-27T00:00:00Z",
                "url": "https://undocs.org/A/RES/68/262",
                "viewpoint_origin": "multilateral",
                "entities": ["United Nations", "Ukraine", "Crimea", "Russian Federation"],
                "geo_location": "New York, United States",
                "license_or_terms": "UN public document",
            }
            catalog.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    "script/fetch_official_docs.py",
                    "--catalog",
                    str(catalog),
                    "--no-builtin-catalog",
                    "--event",
                    "crimea",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in (output_dir / "official_crimea.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "crimea")
            self.assertEqual(rows[0]["extra"]["record_type"], "official_resolution")

    def test_official_doc_catalog_can_apply_review_state_and_preserve_language_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "official_candidates.jsonl"
            review_state = root / "review_state.json"
            output_dir = root / "raw"
            row = {
                "asset_id": "scs-fmprc-position-paper-en",
                "event_id": "scs",
                "review_status": "pending",
                "title": "Position Paper of the Government of China on Jurisdiction",
                "caption": "Official position paper on the South China Sea arbitration jurisdiction question.",
                "institution": "Ministry of Foreign Affairs of the People's Republic of China",
                "document_type": "official_statement",
                "publish_time": "2014-12-07T00:00:00Z",
                "url": "https://example.test/fmprc-position-paper",
                "viewpoint_origin": "china",
                "entities": ["China", "Philippines", "South China Sea", "Permanent Court of Arbitration"],
                "geo_location": "Beijing, China",
                "license_or_terms": "Official public document",
                "document_group_id": "scs-fmprc-position-paper",
                "language": "en",
                "canonical_language": ["zh-Hans-CN"],
                "language_role": "official_translation",
                "authoritative_status": "official_secondary",
                "available_languages_known_to_have": [
                    {
                        "language": "zh-Hans-CN",
                        "url": "https://example.test/fmprc-position-paper-zh",
                        "language_role": "canonical_original",
                    }
                ],
            }
            catalog.write_text(json.dumps(row) + "\n", encoding="utf-8")
            review_state.write_text(
                json.dumps(
                    {
                        "review_state": {
                            "records": {
                                "scs-fmprc-position-paper-en": {
                                    "status": "approved",
                                    "note": "URL verified manually.",
                                    "checked_at": "2026-05-26T12:16:32Z",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python",
                    "script/fetch_official_docs.py",
                    "--catalog",
                    str(catalog),
                    "--review-state",
                    str(review_state),
                    "--no-builtin-catalog",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in (output_dir / "official_scs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            extra = rows[0]["extra"]
            self.assertEqual(extra["review_status"], "approved")
            self.assertEqual(extra["review_state_note"], "URL verified manually.")
            self.assertEqual(extra["document_group_id"], "scs-fmprc-position-paper")
            self.assertEqual(extra["language_role"], "official_translation")
            self.assertEqual(extra["authoritative_status"], "official_secondary")
            self.assertEqual(extra["available_languages_known_to_have"][0]["language"], "zh-Hans-CN")

    def test_official_doc_catalog_skips_pending_rows_before_required_field_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "official_candidates.jsonl"
            output_dir = root / "raw"
            pending_incomplete = {
                "asset_id": "scs-fmprc-position-paper-en",
                "event_id": "scs",
                "review_status": "pending",
                "title": "Pending row with missing metadata",
                "institution": "Ministry of Foreign Affairs",
                "document_type": "official_statement",
                "publish_time": "2014-12-07T00:00:00Z",
                "url": "https://example.test/pending",
            }
            approved = {
                "asset_id": "ukraine-un-resolution-en",
                "event_id": "ukraine",
                "review_status": "approved",
                "title": "Fixture Ukraine Resolution",
                "caption": "Official fixture resolution.",
                "institution": "UN General Assembly",
                "document_type": "official_resolution",
                "publish_time": "2022-03-02T00:00:00Z",
                "url": "https://example.test/ukraine-resolution",
                "viewpoint_origin": "multilateral",
                "entities": ["Ukraine", "United Nations"],
                "geo_location": "New York, United States",
                "license_or_terms": "UN public document",
                "primary_text_language": "uk",
                "download_by_default": True,
                "active_policy": "representative_variant_active",
                "counts_as_extra_official_evidence": True,
            }
            catalog.write_text(json.dumps(pending_incomplete) + "\n" + json.dumps(approved) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    "script/fetch_official_docs.py",
                    "--catalog",
                    str(catalog),
                    "--no-builtin-catalog",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in (output_dir / "official_ukraine.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            extra = rows[0]["extra"]
            self.assertEqual(extra["primary_text_language"], "uk")
            self.assertEqual(extra["download_by_default"], True)
            self.assertEqual(extra["catalog_active_policy"], "representative_variant_active")
            self.assertEqual(extra["counts_as_extra_official_evidence"], True)

    def test_official_doc_catalog_skips_inactive_approved_rows_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "official_candidates.jsonl"
            output_dir = root / "raw"
            inactive_reference = {
                "asset_id": "hk_nsl_a302_en",
                "event_id": "hongkong",
                "review_status": "approved",
                "active_bench": False,
                "title": "English title-only page",
                "caption": "Only the title is translated into English; the body is Chinese.",
                "institution": "Hong Kong e-Legislation",
                "document_type": "legislation_text",
                "publish_time": "2020-06-30T00:00:00Z",
                "url": "https://example.test/a302-en",
                "viewpoint_origin": "hongkong",
                "entities": ["Hong Kong", "National Security Law"],
                "geo_location": "Hong Kong",
                "license_or_terms": "Hong Kong e-Legislation terms",
                "language": "en",
                "language_role": "official_translation",
                "authoritative_status": "official_secondary",
            }
            active_chinese = {
                **inactive_reference,
                "asset_id": "hk_nsl_a302_zh",
                "active_bench": True,
                "title": "Traditional Chinese authoritative page",
                "url": "https://example.test/a302-zh",
                "language": "zh-Hant-HK",
                "language_role": "canonical_original",
                "authoritative_status": "sole_authoritative",
            }
            catalog.write_text(json.dumps(inactive_reference) + "\n" + json.dumps(active_chinese) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    "script/fetch_official_docs.py",
                    "--catalog",
                    str(catalog),
                    "--no-builtin-catalog",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in (output_dir / "official_hongkong.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["extra"]["asset_id"], "hk_nsl_a302_zh")
            self.assertEqual(rows[0]["extra"]["language"], "zh-Hant-HK")

    def test_official_doc_catalog_drops_wikipedia_language_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "official_candidates.jsonl"
            output_dir = root / "raw"
            row = {
                "asset_id": "kosovo-assembly-independence-transcript-2008-02-17-sq",
                "event_id": "kosovo",
                "review_status": "approved",
                "active_bench": True,
                "title": "Kosovo Assembly transcript",
                "caption": "Official Kosovo Assembly transcript.",
                "institution": "Assembly of Kosovo",
                "document_type": "declaration_text",
                "publish_time": "2008-02-17T00:00:00Z",
                "url": "https://old.kuvendikosoves.org/common/docs/proc/trans_s_2008_02_17_al.pdf",
                "viewpoint_origin": "kosovar",
                "entities": ["Kosovo", "Assembly of Kosovo"],
                "geo_location": "Kosovo",
                "license_or_terms": "Assembly of Kosovo official archive terms",
                "language": "sq",
                "available_language_codes": ["sq", "en"],
                "available_languages_known_to_have": [
                    {
                        "language": "sq",
                        "url": "https://old.kuvendikosoves.org/common/docs/proc/trans_s_2008_02_17_al.pdf",
                        "language_role": "canonical_original",
                    },
                    {
                        "language": "en",
                        "url": "https://en.wikipedia.org/wiki/2008_Kosovo_declaration_of_independence",
                        "language_role": "unofficial_translation",
                        "authoritative_status": "reference_only",
                    },
                ],
                "language_variants": [
                    {
                        "language": "sq",
                        "url": "https://old.kuvendikosoves.org/common/docs/proc/trans_s_2008_02_17_al.pdf",
                    },
                    {
                        "language": "en",
                        "url": "https://en.wikipedia.org/wiki/2008_Kosovo_declaration_of_independence",
                    },
                ],
                "available_language_urls_by_language": {
                    "sq": "https://old.kuvendikosoves.org/common/docs/proc/trans_s_2008_02_17_al.pdf",
                    "en": "https://en.wikipedia.org/wiki/2008_Kosovo_declaration_of_independence",
                },
                "available_language_names_by_language": {"sq": "Albanian", "en": "English"},
                "available_language_native_names_by_language": {"sq": "shqip", "en": "English"},
                "translation_provenance": "wikipedia_quotation",
            }
            catalog.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    "script/fetch_official_docs.py",
                    "--catalog",
                    str(catalog),
                    "--no-builtin-catalog",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in (output_dir / "official_kosovo.jsonl").read_text(encoding="utf-8").splitlines()]
            extra = rows[0]["extra"]
            self.assertEqual(extra["available_language_codes"], ["sq"])
            self.assertNotIn("en", extra["available_language_urls_by_language"])
            self.assertEqual(len(extra["available_languages_known_to_have"]), 1)
            self.assertEqual(len(extra["language_variants"]), 1)
            self.assertNotIn("translation_provenance", extra)

    def test_structured_document_is_selected_for_official_hyperedge(self) -> None:
        text_asset = EvidenceAsset(
            asset_id="asset_text_src_scs",
            event_id="scs",
            modality="text",
            asset_source="local",
            source_layer="official",
            viewpoint_origin="US-Anglo",
            publish_time="2016-07-12T00:00:00Z",
            observed_time="2016-07-12T00:00:00Z",
            geo_location="South China Sea",
            url_or_pointer="data/0_raw/scs/source.txt",
            caption_or_transcript="Source text.",
            license_or_terms="local",
            redistribution_flag=True,
            perceptual_hash="hash",
            embedding_id="emb_text",
            extracted_entities=["South China Sea"],
            extracted_claims=[],
            evidence_role="substantive",
            extra={"source_id": "src_scs-official"},
        )
        official_doc = EvidenceAsset(
            asset_id="asset_official_scs_award",
            event_id="scs",
            modality="structured_document",
            asset_source="OFFICIAL_DOC",
            source_layer="official",
            viewpoint_origin="legal_international",
            publish_time="2016-07-12T00:00:00Z",
            observed_time="2016-07-12T00:00:00Z",
            geo_location="The Hague, Netherlands",
            url_or_pointer="https://example.test/award.pdf",
            caption_or_transcript="Final award.",
            license_or_terms="public",
            redistribution_flag=False,
            perceptual_hash="hash",
            embedding_id="emb_doc",
            extracted_entities=["South China Sea"],
            extracted_claims=[],
            evidence_role="substantive",
            extra={"external_file": "data/0_external/external_asset_raw/official_scs.jsonl"},
        )
        structured_event = EvidenceAsset(
            asset_id="asset_structured_scs",
            event_id="scs",
            modality="structured_event",
            asset_source="GeoMosaic event registry",
            source_layer="structured",
            viewpoint_origin="all",
            publish_time="2016-07-12T00:00:00Z",
            observed_time="2016-07-12T00:00:00Z",
            geo_location="South China Sea",
            url_or_pointer="geomosaic://event/scs",
            caption_or_transcript="Structured event.",
            license_or_terms="local",
            redistribution_flag=True,
            perceptual_hash="hash",
            embedding_id="emb_structured",
            extracted_entities=["South China Sea"],
            extracted_claims=[],
            evidence_role="context",
        )
        selected_source = SourceRecord(
            source_id="src_scs-official",
            event_id="scs",
            source_layer="official",
            viewpoint_origin="US-Anglo",
            document_type="official_statement",
            institution_or_outlet="PCA",
            publish_time="2016-07-12T00:00:00Z",
            retrieval_time="2016-07-12T00:00:00Z",
            url="https://example.test/source",
            language="en",
            license_or_terms="public",
            redistribution_flag=True,
            normalized_text_hash="hash",
        )
        group = {
            "event_id": "scs",
            "source_layer": "official",
            "source_vp": "US-Anglo",
            "outlet": None,
            "article_idx": None,
        }

        selected = select_assets_for_hyperedge(group, [selected_source], {"scs": [text_asset, structured_event, official_doc]})

        self.assertIn("asset_official_scs_award", {asset.asset_id for asset in selected})
        self.assertEqual(match_level_for_hyperedge_asset(group, {"src_scs-official"}, official_doc), "L2")

    def test_acled_oauth_password_request_matches_official_docs(self) -> None:
        calls = []

        def fake_post_form_json(url, data, timeout=30):
            calls.append((url, data, timeout))
            return SimpleNamespace(data={"access_token": "token-1", "refresh_token": "refresh-1", "expires_in": 86400})

        client = ACLEDClient(credentials=ACLEDCredentials(username="user@example.com", password="secret"))
        with patch.object(acled_module, "post_form_json", fake_post_form_json):
            self.assertEqual(client.access_token(), "token-1")

        _, payload, _ = calls[0]
        self.assertEqual(payload["username"], "user@example.com")
        self.assertEqual(payload["password"], "secret")
        self.assertEqual(payload["grant_type"], "password")
        self.assertEqual(payload["client_id"], "acled")
        self.assertEqual(payload["scope"], "authenticated")

    def test_acled_credentials_accept_local_pwd_alias(self) -> None:
        with patch.dict(os.environ, {"ACLED_EMAIL": "user@example.com", "ACLED_PWD": "secret"}, clear=True):
            credentials = ACLEDCredentials.from_env()
        self.assertEqual(credentials.username, "user@example.com")
        self.assertEqual(credentials.password, "secret")

    def test_external_cli_can_load_env_like_credentials_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ACLED_api"
            path.write_text("ACLED_EMAIL=user@example.com\nACLED_PWD=secret\n", encoding="utf-8")
            with patch.dict(os.environ, {"ACLED_EMAIL": "old@example.com"}, clear=True):
                load_env_file(path)
                self.assertEqual(os.environ["ACLED_EMAIL"], "user@example.com")
                self.assertEqual(os.environ["ACLED_PWD"], "secret")

    def test_external_cli_loads_prior_summary_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            merged = Path(tmp) / "assets.jsonl"
            summary = {
                "errors": [{"source": "wikimedia", "error": "fixture"}],
                "warnings": [{"source": "acled", "warning": "fixture"}],
            }
            merged.with_suffix(".summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(load_prior_summary_issues(merged), (summary["errors"], summary["warnings"]))

    def test_acled_oauth_refreshes_expired_access_token(self) -> None:
        calls = []

        def fake_post_form_json(url, data, timeout=30):
            calls.append(data)
            return SimpleNamespace(data={"access_token": "token-2", "refresh_token": "refresh-2", "expires_in": 86400})

        client = ACLEDClient(credentials=ACLEDCredentials(username="user@example.com", password="secret"))
        client._token = "old-token"
        client._refresh_token = "refresh-1"
        client._token_expires_at = 0.0
        with patch.object(acled_module, "post_form_json", fake_post_form_json):
            self.assertEqual(client.access_token(), "token-2")

        self.assertEqual(calls[0]["refresh_token"], "refresh-1")
        self.assertEqual(calls[0]["grant_type"], "refresh_token")
        self.assertEqual(calls[0]["client_id"], "acled")

    def test_acled_client_retries_unauthorized_by_status_code(self) -> None:
        get_calls = []

        def fake_post_form_json(url, data, timeout=30):
            return SimpleNamespace(data={"access_token": "fresh-token", "refresh_token": "refresh-2", "expires_in": 86400})

        def fake_get_json(url, params, headers=None, timeout=30):
            get_calls.append(headers["Authorization"])
            if len(get_calls) == 1:
                raise HTTPClientError("fixture unauthorized", status=401)
            return SimpleNamespace(data={"status": 200, "data": []})

        client = ACLEDClient(credentials=ACLEDCredentials(username="user@example.com", password="secret"))
        client._token = "expired-token"
        client._token_expires_at = time.time() + 3600
        with patch.object(acled_module, "post_form_json", fake_post_form_json), patch.object(acled_module, "get_json", fake_get_json):
            self.assertEqual(client.read(), [])
        self.assertEqual(get_calls, ["Bearer expired-token", "Bearer fresh-token"])

    def test_acled_client_paginates_until_short_page(self) -> None:
        pages = {
            1: [{"id": "a"}, {"id": "b"}],
            2: [{"id": "c"}, {"id": "d"}],
            3: [{"id": "e"}],
        }
        seen_pages = []

        def fake_read(params=None, limit=None, page=None):
            seen_pages.append(page)
            return pages[page]

        client = ACLEDClient(credentials=ACLEDCredentials(username="user@example.com", password="secret"))
        with patch.object(client, "read", fake_read):
            rows = client.read_paginated({"country": "Ukraine"}, page_size=2, max_pages=5)

        self.assertEqual([row["id"] for row in rows], ["a", "b", "c", "d", "e"])
        self.assertEqual(seen_pages, [1, 2, 3])

    def test_acled_events_for_window_can_request_paginated_rows(self) -> None:
        calls = []

        def fake_read_paginated(params=None, page_size=5000, max_pages=20):
            calls.append((params, page_size, max_pages))
            return [{"event_id_cnty": "UKR1"}]

        client = ACLEDClient(credentials=ACLEDCredentials(username="user@example.com", password="secret"))
        with patch.object(client, "read_paginated", fake_read_paginated):
            rows = client.events_for_window(country="Ukraine", start_date="2022-02-17", end_date="2022-03-03", limit=1000, paginate=True, max_pages=7)

        self.assertEqual(rows, [{"event_id_cnty": "UKR1"}])
        params, page_size, max_pages = calls[0]
        self.assertEqual(params["country"], "Ukraine")
        self.assertEqual(params["event_date"], "2022-02-17|2022-03-03")
        self.assertEqual(page_size, 1000)
        self.assertEqual(max_pages, 7)

    def test_acled_result_warnings_mark_zero_rows_and_limit_hits(self) -> None:
        zero = acled_result_warnings("crimea", "Ukraine", "2014-03-09", "2014-03-23", 0, 200)
        limit = acled_result_warnings("ukraine", "Ukraine", "2022-02-17", "2022-03-03", 200, 200)
        skipped = acled_result_warnings("scs", None, "2016-07-05", "2016-07-19", 0, 200)
        self.assertIn("0 rows", zero[0]["warning"])
        self.assertIn("hitting limit=200", limit[0]["warning"])
        self.assertIn("multi-country", skipped[0]["warning"])

    def test_wikimedia_client_retries_rate_limit_response(self) -> None:
        calls = []

        def fake_get_json(url, params, timeout=30):
            calls.append(params)
            if len(calls) == 1:
                raise HTTPClientError("HTTP 429 for fixture")
            return SimpleNamespace(data={"query": {"pages": {"1": {"title": "File:Fixture.png"}}}})

        client = WikimediaCommonsClient(request_delay_seconds=0, max_retries=1, retry_backoff_seconds=0)
        import geomosaic_hg.clients.wikimedia as wikimedia_module

        with patch.object(wikimedia_module, "get_json", fake_get_json):
            self.assertEqual(client.search_files("fixture", limit=1), ["File:Fixture.png"])
        self.assertEqual(len(calls), 2)

    def test_wikimedia_client_uses_exponential_backoff_for_rate_limits(self) -> None:
        calls = []
        sleeps = []

        def fake_get_json(url, params, timeout=30):
            calls.append(params)
            if len(calls) < 3:
                raise HTTPClientError("HTTP 429 for fixture", status=429)
            return SimpleNamespace(data={"query": {"pages": {}}})

        client = WikimediaCommonsClient(request_delay_seconds=0, max_retries=2, retry_backoff_seconds=5)
        import geomosaic_hg.clients.wikimedia as wikimedia_module

        with patch.object(wikimedia_module, "get_json", fake_get_json), patch.object(wikimedia_module.time, "sleep", lambda seconds: sleeps.append(seconds)):
            self.assertEqual(client.search_files("fixture", limit=1), [])
        self.assertEqual(sleeps, [5, 10])

    def test_wikimedia_asset_id_keeps_image_and_map_modalities_distinct(self) -> None:
        file = WikimediaFile(
            title="File:Fixture.png",
            page_id=1,
            file_url="https://example.test/fixture.png",
            description_url="https://commons.wikimedia.org/wiki/File:Fixture.png",
            thumb_url="",
            mime="image/png",
            sha1="fixture",
            width=100,
            height=100,
            license_short_name="CC-BY",
            license_url="",
            object_name="Fixture",
            artist="",
            credit="",
            date_time="2014-03-16T00:00:00Z",
        )
        image = wikimedia_file_to_asset(file, "crimea", "image_full")
        map_asset = wikimedia_file_to_asset(file, "crimea", "map_pointer")
        self.assertNotEqual(image.asset_id, map_asset.asset_id)
        self.assertEqual(map_asset.evidence_role, "map_like")

    def test_fetch_wikimedia_filters_non_image_mime(self) -> None:
        files = [
            WikimediaFile(
                title="File:Paper.pdf",
                page_id=1,
                file_url="https://example.test/paper.pdf",
                description_url="https://commons.wikimedia.org/wiki/File:Paper.pdf",
                thumb_url="",
                mime="application/pdf",
                sha1="pdf",
                width=100,
                height=100,
                license_short_name="CC-BY",
                license_url="",
                object_name="Paper",
                artist="",
                credit="",
                date_time="2014-03-16T00:00:00Z",
            ),
            WikimediaFile(
                title="File:2014 Crimean referendum ballot.jpg",
                page_id=2,
                file_url="https://example.test/referendum.jpg",
                description_url="https://commons.wikimedia.org/wiki/File:2014_Crimean_referendum_ballot.jpg",
                thumb_url="",
                mime="image/jpeg",
                sha1="jpg",
                width=100,
                height=100,
                license_short_name="CC-BY",
                license_url="",
                object_name="2014 Crimean referendum ballot",
                artist="",
                credit="",
                date_time="2014-03-16T00:00:00Z",
            ),
            WikimediaFile(
                title="File:Crimea, Ai-Petri, low clouds.jpg",
                page_id=3,
                file_url="https://example.test/clouds.jpg",
                description_url="https://commons.wikimedia.org/wiki/File:Crimea_Ai-Petri_low_clouds.jpg",
                thumb_url="",
                mime="image/jpeg",
                sha1="clouds",
                width=100,
                height=100,
                license_short_name="CC-BY",
                license_url="",
                object_name="Crimea, Ai-Petri, low clouds",
                artist="",
                credit="",
                date_time="2014-03-16T00:00:00Z",
            ),
        ]

        class FakeClient:
            def page_imageinfo(self, page_title, limit=50):
                self.page_title = page_title
                return files

            def search_imageinfo(self, query, limit=10):
                raise AssertionError("event-page Wikimedia fetch should not use generic Commons search")

        client = FakeClient()
        rows = fetch_wikimedia_for_event(client, "crimea", image_limit=1, map_limit=0)
        self.assertEqual(client.page_title, "Annexation of Crimea by the Russian Federation")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["extra"]["mime"], "image/jpeg")
        self.assertIn("referendum", rows[0]["extra"]["title"].lower())
        self.assertEqual(rows[0]["extra"]["collection_channel"], "wikipedia_page_bound")
        self.assertIs(rows[0]["extra"]["page_bound"], True)
        self.assertEqual(rows[0]["extra"]["source_page_title"], "Annexation of Crimea by the Russian Federation")
        self.assertEqual(rows[0]["extra"]["file_title"], "File:2014 Crimean referendum ballot.jpg")
        self.assertEqual(rows[0]["extra"]["proposed_role"], "substantive_event_image")
        self.assertEqual(rows[0]["extra"]["temporal_status"], "near_event_window")
        self.assertIs(rows[0]["extra"]["active_bench"], True)
        self.assertEqual(rows[0]["evidence_role"], "complementary")

    def test_page_bound_wikimedia_asset_links_to_wiki_source_as_l1(self) -> None:
        source = SourceRecord(
            source_id="src_crimea-all-wiki",
            event_id="crimea",
            source_layer="wiki",
            viewpoint_origin="all",
            document_type="wiki_entry",
            institution_or_outlet="Wikipedia",
            publish_time="2014-03-16T00:00:00Z",
            retrieval_time="2014-03-16T00:00:00Z",
            url="https://en.wikipedia.org/wiki/Annexation_of_Crimea_by_the_Russian_Federation",
            language="en",
            license_or_terms="wikipedia-derived-local-text",
            redistribution_flag=True,
            normalized_text_hash="hash",
            extra={"source_key": "wiki", "source_page_title": "Annexation of Crimea by the Russian Federation"},
        )
        asset = EvidenceAsset(
            asset_id="asset_wikimedia_crimea_image",
            event_id="crimea",
            modality="image_full",
            asset_source="Wikimedia Commons",
            source_layer="wiki",
            viewpoint_origin="all",
            publish_time="2014-03-16T00:00:00Z",
            observed_time="2014-03-16T00:00:00Z",
            geo_location="Crimea, Ukraine",
            url_or_pointer="data/0_external/event_images/crimea/image.jpg",
            caption_or_transcript="2014 Crimean referendum ballot.",
            license_or_terms="CC BY",
            redistribution_flag=True,
            perceptual_hash="hash",
            embedding_id="emb",
            extracted_entities=["Crimea"],
            extracted_claims=[],
            evidence_role="complementary",
            extra={
                "collection_channel": "wikipedia_page_bound",
                "page_bound": True,
                "source_page_title": "Annexation of Crimea by the Russian Federation",
                "source_page_url": "https://en.wikipedia.org/wiki/Annexation_of_Crimea_by_the_Russian_Federation",
            },
        )

        links = build_source_asset_links([source], [asset])
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].match_level, "L1")
        self.assertIn("page-bound", links[0].match_reason)

    def test_fallback_wikimedia_page_does_not_link_to_primary_wiki_source_as_l1(self) -> None:
        source = SourceRecord(
            source_id="src_scs-all-wiki",
            event_id="scs",
            source_layer="wiki",
            viewpoint_origin="all",
            document_type="wiki_entry",
            institution_or_outlet="Wikipedia",
            publish_time="2016-07-12T00:00:00Z",
            retrieval_time="2016-07-12T00:00:00Z",
            url="data/0_raw/scs/scs___all__wiki.txt",
            language="en",
            license_or_terms="wikipedia-derived-local-text",
            redistribution_flag=True,
            normalized_text_hash="hash",
            extra={"source_key": "wiki", "source_page_title": "South China Sea Arbitration"},
        )
        asset = EvidenceAsset(
            asset_id="asset_wikimedia_scs_fallback_map",
            event_id="scs",
            modality="map_pointer",
            asset_source="Wikimedia Commons",
            source_layer="wiki",
            viewpoint_origin="all",
            publish_time="2016-07-12T00:00:00Z",
            observed_time="2016-07-12T00:00:00Z",
            geo_location="South China Sea",
            url_or_pointer="data/0_external/event_images/scs/map.svg",
            caption_or_transcript="South China Sea territorial claims map.",
            license_or_terms="CC BY",
            redistribution_flag=True,
            perceptual_hash="hash",
            embedding_id="emb",
            extracted_entities=["South China Sea"],
            extracted_claims=[],
            evidence_role="map_like",
            extra={
                "collection_channel": "wikipedia_page_bound",
                "page_bound": True,
                "source_page_title": "Territorial disputes in the South China Sea",
                "source_page_url": "https://en.wikipedia.org/wiki/Territorial_disputes_in_the_South_China_Sea",
                "proposed_role": "map_like",
            },
        )

        links = build_source_asset_links([source], [asset])
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].match_level, "L4")

    def test_subset_network_run_merges_all_existing_raw_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            merged = root / "merged.jsonl"
            crimea = {
                "asset_id": "asset_wikimedia_crimea_fixture",
                "event_id": "crimea",
                "modality": "image_full",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2014-03-16T00:00:00Z",
                "observed_time": "2014-03-16T00:00:00Z",
                "geo_location": "Crimea, Ukraine",
                "url_or_pointer": "https://commons.wikimedia.org/wiki/File:Fixture.png",
                "caption_or_transcript": "Fixture image.",
                "license_or_terms": "CC-BY",
                "redistribution_flag": True,
                "perceptual_hash": "fixturehash",
                "embedding_id": "emb_asset_wikimedia_crimea_fixture",
                "extracted_entities": ["Crimea"],
                "extracted_claims": [],
                "evidence_role": "complementary",
                "extra": {"active_bench": True},
            }
            ukraine = {**crimea, "asset_id": "asset_wikimedia_ukraine_fixture", "event_id": "ukraine"}
            (raw / "wikimedia_crimea.jsonl").write_text(json.dumps(crimea) + "\n", encoding="utf-8")
            (raw / "wikimedia_ukraine.jsonl").write_text(json.dumps(ukraine) + "\n", encoding="utf-8")

            summary = run_external_asset_collection(raw, merged, event_ids={"ukraine"}, skip_wikimedia=True, skip_acled=True)
            self.assertEqual(summary["by_event"], {"crimea": 1, "ukraine": 1})

    def test_collect_existing_filters_pdf_and_downloads_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            merged = root / "merged.jsonl"
            image_dir = root / "images"

            base = {
                "event_id": "crimea",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2014-03-16T00:00:00Z",
                "observed_time": "2014-03-16T00:00:00Z",
                "geo_location": "Crimea, Ukraine",
                "caption_or_transcript": "Fixture.",
                "license_or_terms": "CC-BY",
                "redistribution_flag": True,
                "perceptual_hash": "fixturehash",
                "extracted_entities": ["Crimea"],
                "extracted_claims": [],
            }
            pdf = {
                **base,
                "asset_id": "asset_wikimedia_crimea_pdf",
                "modality": "image_full",
                "url_or_pointer": "https://example.test/paper.pdf",
                "embedding_id": "emb_pdf",
                "evidence_role": "complementary",
                "extra": {"mime": "application/pdf"},
            }
            image = {
                **base,
                "asset_id": "asset_wikimedia_crimea_image",
                "modality": "map_pointer",
                "url_or_pointer": "https://example.test/map.jpg",
                "embedding_id": "emb_image",
                "evidence_role": "map_like",
                "extra": {"mime": "image/jpeg", "thumb_url": "https://example.test/thumb.jpg", "active_bench": True},
            }
            (raw / "wikimedia_crimea.jsonl").write_text(json.dumps(pdf) + "\n" + json.dumps(image) + "\n", encoding="utf-8")

            import geomosaic_hg.external_assets as external_assets_module
            downloaded_urls = []

            def fake_download(url, output_path, timeout=60, max_retries=2, retry_backoff_seconds=30.0):
                downloaded_urls.append(url)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake-image")

            with patch.object(external_assets_module, "download_url", fake_download):
                summary = collect_existing_external_assets(raw, merged, {"crimea"}, download_wikimedia_images=True, image_dir=image_dir)

            rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["asset_id"], "asset_wikimedia_crimea_image")
            self.assertTrue(rows[0]["url_or_pointer"].endswith(".jpg"))
            self.assertEqual(downloaded_urls, ["https://example.test/thumb.jpg"])
            self.assertEqual(rows[0]["extra"]["original_url"], "https://example.test/map.jpg")
            self.assertEqual(rows[0]["extra"]["download_url"], "https://example.test/thumb.jpg")
            self.assertTrue((image_dir / "crimea" / "asset_wikimedia_crimea_image.jpg").exists())
            self.assertTrue(any("Filtered non-image Wikimedia asset" in row["warning"] for row in summary["warnings"]))

    def test_materialize_wikimedia_image_renames_svg_thumbnail_by_magic_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            row = {
                "asset_id": "asset_wikimedia_scs_map",
                "event_id": "scs",
                "modality": "map_pointer",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2016-07-12T00:00:00Z",
                "observed_time": "2016-07-12T00:00:00Z",
                "geo_location": "South China Sea",
                "url_or_pointer": "https://upload.wikimedia.org/example/claims.svg",
                "caption_or_transcript": "South China Sea claims map.",
                "license_or_terms": "CC BY",
                "redistribution_flag": True,
                "perceptual_hash": "hash",
                "embedding_id": "emb",
                "extracted_entities": ["South China Sea"],
                "extracted_claims": [],
                "evidence_role": "map_like",
                "extra": {
                    "mime": "image/svg+xml",
                    "thumb_url": "https://upload.wikimedia.org/thumb/example/claims.svg/960px-claims.svg.png",
                },
            }

            import geomosaic_hg.external_assets as external_assets_module

            def fake_download(url, output_path, timeout=60, max_retries=2, retry_backoff_seconds=30.0):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

            with patch.object(external_assets_module, "download_url", fake_download):
                clean, warning = external_assets_module.materialize_wikimedia_image(row, image_dir)

            self.assertIsNone(warning)
            self.assertTrue(clean["url_or_pointer"].endswith(".png"))
            self.assertEqual(clean["extra"]["detected_file_extension"], ".png")
            self.assertTrue((image_dir / "scs" / "asset_wikimedia_scs_map.png").exists())
            self.assertFalse((image_dir / "scs" / "asset_wikimedia_scs_map.svg").exists())

    def test_collect_existing_external_assets_writes_download_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            merged = root / "merged.jsonl"
            image_dir = root / "images"
            asset = {
                "asset_id": "asset_wikimedia_kosovo_fixture",
                "event_id": "kosovo",
                "modality": "image_full",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2008-02-17T00:00:00Z",
                "observed_time": "2008-02-17T00:00:00Z",
                "geo_location": "Kosovo",
                "url_or_pointer": "https://upload.wikimedia.org/example/kosovo.jpg",
                "caption_or_transcript": "Kosovo independence celebration.",
                "license_or_terms": "CC BY-SA",
                "redistribution_flag": True,
                "perceptual_hash": "hash",
                "embedding_id": "emb",
                "extracted_entities": ["Kosovo"],
                "extracted_claims": [],
                "evidence_role": "complementary",
                "extra": {
                    "mime": "image/jpeg",
                    "collection_channel": "wikipedia_page_bound",
                    "page_bound": True,
                    "source_page_title": "2008 Kosovo declaration of independence",
                    "source_page_url": "https://en.wikipedia.org/wiki/2008_Kosovo_declaration_of_independence",
                    "file_title": "File:Kosovo fixture.jpg",
                    "caption": "Kosovo independence celebration.",
                    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "proposed_role": "substantive_event_image",
                    "temporal_status": "near_event_window",
                    "active_bench": True,
                },
            }
            (raw / "wikimedia_kosovo.jsonl").write_text(json.dumps(asset) + "\n", encoding="utf-8")

            import geomosaic_hg.external_assets as external_assets_module

            def fake_download(url, output_path, timeout=60, max_retries=2, retry_backoff_seconds=30.0):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"\xff\xd8\xfffixture")

            with patch.object(external_assets_module, "download_url", fake_download):
                collect_existing_external_assets(raw, merged, {"kosovo"}, download_wikimedia_images=True, image_dir=image_dir)

            manifest = image_dir / "manifest.jsonl"
            self.assertTrue(manifest.exists())
            manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(manifest_rows[0]["asset_id"], "asset_wikimedia_kosovo_fixture")
            self.assertEqual(manifest_rows[0]["source_page_title"], "2008 Kosovo declaration of independence")
            self.assertEqual(manifest_rows[0]["file_title"], "File:Kosovo fixture.jpg")
            self.assertEqual(manifest_rows[0]["caption"], "Kosovo independence celebration.")
            self.assertEqual(manifest_rows[0]["proposed_role"], "substantive_event_image")

    def test_download_url_retries_rate_limited_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "image.jpg"
            attempts = []

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def read(self):
                    return b"fake-image"

            def fake_urlopen(request, timeout=60):
                attempts.append(request.full_url)
                if len(attempts) == 1:
                    raise HTTPError(request.full_url, 429, "Too Many Requests", hdrs=None, fp=None)
                return FakeResponse()

            sleeps = []

            import geomosaic_hg.external_assets as external_assets_module

            with patch.object(external_assets_module, "urlopen", fake_urlopen), patch.object(external_assets_module.time, "sleep", lambda seconds: sleeps.append(seconds)):
                external_assets_module.download_url("https://example.test/image.jpg", output, timeout=1, max_retries=2, retry_backoff_seconds=5)

            self.assertEqual(len(attempts), 2)
            self.assertEqual(sleeps, [5])
            self.assertEqual(output.read_bytes(), b"fake-image")


if __name__ == "__main__":
    unittest.main()
