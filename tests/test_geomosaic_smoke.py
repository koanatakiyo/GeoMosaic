from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.bpe import BPEConfig, retrieve
from geomosaic_hg.baselines import run_baselines
from geomosaic_hg.build import build_all, load_tables
from geomosaic_hg.paths import RAW_DIR, SCORE_DIR
from geomosaic_hg.smpi import SMPI
from geomosaic_hg.validation import validate_core_tables


class GeoMosaicSmokeTest(unittest.TestCase):
    def test_build_retrieve_and_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bench"
            external_dir = Path(tmp) / "external"
            external_dir.mkdir()
            external_asset = {
                "asset_id": "asset_wikimedia_crimea_test",
                "event_id": "crimea",
                "modality": "image_full",
                "asset_source": "Wikimedia Commons",
                "source_layer": "wiki",
                "viewpoint_origin": "all",
                "publish_time": "2014-03-16T00:00:00Z",
                "observed_time": "2014-03-16T00:00:00Z",
                "geo_location": "Crimea, Ukraine",
                "url_or_pointer": "https://commons.wikimedia.org/wiki/File:Crimea_test.png",
                "caption_or_transcript": "Crimea test map image.",
                "license_or_terms": "test-license",
                "redistribution_flag": True,
                "perceptual_hash": "externalhash",
                "embedding_id": "emb_asset_wikimedia_crimea_test",
                "extracted_entities": ["Crimea"],
                "extracted_claims": [],
                "evidence_role": "complementary",
            }
            (external_dir / "wikimedia_crimea.jsonl").write_text(
                json.dumps(external_asset) + "\n",
                encoding="utf-8",
            )
            summary = build_all(
                RAW_DIR,
                out_dir,
                [SCORE_DIR / "combined_official", SCORE_DIR / "combined_zh_news"],
                {"crimea", "ukraine"},
                external_dir,
            )
            self.assertGreater(summary["counts"]["source_records"], 0)
            self.assertGreater(summary["counts"]["claim_evidence_hyperedges"], 0)
            self.assertEqual(summary["external_assets"], 1)

            tables = load_tables(out_dir)
            self.assertEqual(validate_core_tables(tables), [])
            self.assertIn("asset_wikimedia_crimea_test", {row["asset_id"] for row in tables["evidence_assets"]})
            source_keys = [
                (row["event_id"], row["source_layer"], row.get("extra", {}).get("source_key"))
                for row in tables["source_records"]
                if row.get("extra", {}).get("source_key")
            ]
            self.assertEqual(len(source_keys), len(set(source_keys)))
            guardian_one = next(
                row
                for row in tables["claim_evidence_hyperedges"]
                if row["claim_id"].startswith("crimea:news:news-guardian-1:")
            )
            self.assertEqual(guardian_one["source_record_set"], ["src_crimea-all-news-guardian-1"])
            forbidden = PROJECT_ROOT.parent.as_posix() + "/"
            for rows in tables.values():
                for row in rows:
                    self.assertNotIn(forbidden, str(row))

            index = SMPI.from_dir(out_dir)
            result = retrieve(
                index,
                "Ukraine sovereignty territorial integrity",
                BPEConfig(cutoff="2026-01-01T00:00:00Z", hyperedge_budget=5),
            )
            self.assertGreater(result["candidate_count"], 0)
            self.assertEqual(len(result["selected_hyperedges"]), 5)
            self.assertTrue(result["provenance_trace"])

            ukraine_only = retrieve(
                index,
                "Ukraine sovereignty territorial integrity",
                BPEConfig(
                    event_ids={"ukraine"},
                    cutoff="2026-01-01T00:00:00Z",
                    hyperedge_budget=3,
                ),
            )
            self.assertGreater(ukraine_only["candidate_count"], 0)
            self.assertLess(ukraine_only["candidate_count"], result["candidate_count"])
            self.assertEqual({h["event_id"] for h in ukraine_only["selected_hyperedges"]}, {"ukraine"})

            for source_layer in ("official", "news"):
                constrained = retrieve(
                    index,
                    "Ukraine sovereignty territorial integrity",
                    BPEConfig(
                        source_layers={source_layer},
                        cutoff="2026-01-01T00:00:00Z",
                        hyperedge_budget=3,
                    ),
                )
                self.assertGreater(constrained["candidate_count"], 0)
                self.assertTrue(constrained["selected_hyperedges"])
                for hyperedge in constrained["selected_hyperedges"]:
                    self.assertIn(source_layer, set(hyperedge["primary_source_layer_set"]))

            wiki_primary = retrieve(
                index,
                "Ukraine sovereignty territorial integrity",
                BPEConfig(
                    source_layers={"wiki"},
                    cutoff="2026-01-01T00:00:00Z",
                    hyperedge_budget=3,
                ),
            )
            self.assertEqual(wiki_primary["candidate_count"], 0)

            strict_match = retrieve(
                index,
                "Crimea territorial integrity referendum",
                BPEConfig(
                    max_match_level="L3",
                    cutoff="2026-01-01T00:00:00Z",
                    hyperedge_budget=3,
                ),
            )
            self.assertGreater(strict_match["candidate_count"], 0)
            self.assertTrue(strict_match["selected_hyperedges"])

            image_result = retrieve(
                index,
                "Crimea map",
                BPEConfig(
                    modalities={"image_full"},
                    cutoff="2026-01-01T00:00:00Z",
                    hyperedge_budget=3,
                ),
            )
            self.assertGreater(image_result["candidate_count"], 0)
            for hyperedge in image_result["selected_hyperedges"]:
                self.assertIn("image_full", set(hyperedge["modality_set"]))
                self.assertIn("structured_event", set(hyperedge["modality_set"]))
                self.assertIn("structured", set(hyperedge["source_layer_set"]))

            baseline_config = BPEConfig(cutoff="2026-01-01T00:00:00Z", hyperedge_budget=3, ann_k=7)
            bpe_result = retrieve(index, "Crimea territorial integrity referendum", baseline_config)
            baseline_results = run_baselines(index, "Crimea territorial integrity referendum", baseline_config)
            self.assertEqual(baseline_results["Metadata++"]["candidate_count"], bpe_result["candidate_count"])
            self.assertEqual(baseline_results["Metadata++"]["seed_count"], 7)

            event_baseline_config = BPEConfig(event_ids={"ukraine"}, cutoff="2026-01-01T00:00:00Z", hyperedge_budget=3, ann_k=7)
            event_bpe_result = retrieve(index, "Ukraine sovereignty territorial integrity", event_baseline_config)
            event_baseline_results = run_baselines(index, "Ukraine sovereignty territorial integrity", event_baseline_config)
            self.assertEqual(event_baseline_results["Metadata++"]["candidate_count"], event_bpe_result["candidate_count"])
            self.assertEqual({h["event_id"] for h in event_baseline_results["Metadata++"]["selected_hyperedges"]}, {"ukraine"})

            malformed = {name: [dict(row) for row in rows] for name, rows in tables.items()}
            malformed["source_records"][0]["word_count"] = "many"
            malformed["source_records"].append(dict(malformed["source_records"][0]))
            malformed["evidence_assets"][0]["redistribution_flag"] = "yes"
            malformed["evidence_assets"].append(dict(malformed["evidence_assets"][0]))
            malformed["source_asset_links"][0]["match_score"] = 1.5
            malformed["claim_evidence_hyperedges"][0]["confidence"] = "0.8"
            malformed["claim_evidence_hyperedges"].append(dict(malformed["claim_evidence_hyperedges"][0]))
            errors = validate_core_tables(malformed)
            self.assertTrue(any("word_count expected int" in error for error in errors), errors)
            self.assertTrue(any("redistribution_flag expected bool" in error for error in errors), errors)
            self.assertTrue(any("match_score out of range" in error for error in errors), errors)
            self.assertTrue(any("confidence expected float" in error for error in errors), errors)
            self.assertTrue(any("duplicate source_id" in error for error in errors), errors)
            self.assertTrue(any("duplicate asset_id" in error for error in errors), errors)
            self.assertTrue(any("duplicate hyperedge_id" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
