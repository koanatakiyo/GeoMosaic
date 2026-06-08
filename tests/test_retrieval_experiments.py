from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.retrieval_experiments import (  # noqa: E402
    METRIC_FIELDS,
    default_e1_e3_configs,
    format_set,
    query_for_event,
    summarize_comparison_rows,
)


class RetrievalExperimentTest(unittest.TestCase):
    def test_default_configs_cover_full_source_modality_and_constraint_ablations(self) -> None:
        configs = {config.config_id: config for config in default_e1_e3_configs()}

        self.assertIn("full", configs)
        self.assertIn("source_official", configs)
        self.assertIn("source_news", configs)
        self.assertIn("modality_text", configs)
        self.assertIn("modality_visual", configs)
        self.assertIn("modality_structured", configs)
        self.assertIn("match_l3_or_better", configs)

        self.assertEqual(configs["source_official"].source_layers, frozenset({"official"}))
        self.assertEqual(configs["modality_visual"].modalities, frozenset({"image_full", "image_restricted_pointer", "map_pointer"}))
        self.assertEqual(configs["match_l3_or_better"].max_match_level, "L3")

    def test_summary_averages_numeric_metrics_by_config_and_method(self) -> None:
        rows = [
            {
                "event_id": "crimea",
                "event_group": "non_acled",
                "config_id": "full",
                "method": "GeoMosaic-HG BPE",
                "candidate_count": 10,
                "viewpoint_coverage": 0.5,
                "viewpoint_balance": 0.25,
                "source_diversity": 2,
                "temporal_leakage_rate": 0.0,
            },
            {
                "event_id": "ukraine",
                "event_group": "acled_covered",
                "config_id": "full",
                "method": "GeoMosaic-HG BPE",
                "candidate_count": 30,
                "viewpoint_coverage": 1.0,
                "viewpoint_balance": 0.75,
                "source_diversity": 4,
                "temporal_leakage_rate": 0.2,
            },
            {
                "event_id": "crimea",
                "event_group": "non_acled",
                "config_id": "full",
                "method": "NaiveRAG",
                "candidate_count": 20,
                "viewpoint_coverage": 0.25,
                "viewpoint_balance": 0.0,
                "source_diversity": 1,
                "temporal_leakage_rate": 0.0,
            },
        ]

        summary = summarize_comparison_rows(rows)
        by_key = {(row["config_id"], row["method"]): row for row in summary}

        bpe = by_key[("full", "GeoMosaic-HG BPE")]
        self.assertEqual(bpe["events_n"], 2)
        self.assertEqual(bpe["candidate_count"], 20.0)
        self.assertEqual(bpe["viewpoint_coverage"], 0.75)
        self.assertEqual(bpe["viewpoint_balance"], 0.5)
        self.assertEqual(bpe["source_diversity"], 3.0)
        self.assertEqual(bpe["temporal_leakage_rate"], 0.1)

        for field in METRIC_FIELDS:
            self.assertIn(field, bpe)

    def test_query_and_set_format_are_stable(self) -> None:
        self.assertEqual(query_for_event("ukraine"), "Ukraine sovereignty territorial integrity")
        self.assertEqual(query_for_event("scs"), "South China Sea arbitration maritime claims")
        self.assertEqual(format_set(frozenset({"wiki", "official"})), "official,wiki")
        self.assertEqual(format_set(frozenset()), "")


if __name__ == "__main__":
    unittest.main()
