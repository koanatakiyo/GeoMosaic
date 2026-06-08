from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.publication_tables import (  # noqa: E402
    compact_e2_table,
    coverage_matrix_rows,
    e4_pairwise_direction,
    filter_e1_ablation_table,
    filter_e1_method_table,
    sort_by_bench_scale,
)


class PublicationTablesTest(unittest.TestCase):
    def test_filter_e1_method_table_uses_full_config_only(self) -> None:
        rows = [
            {"config_id": "full", "method": "GeoMosaic-HG BPE", "viewpoint_coverage": "1.0"},
            {"config_id": "source_news", "method": "GeoMosaic-HG BPE", "viewpoint_coverage": "0.9"},
            {"config_id": "full", "method": "NaiveRAG", "viewpoint_coverage": "0.5"},
        ]
        out = filter_e1_method_table(rows)
        self.assertEqual([row["method"] for row in out], ["GeoMosaic-HG BPE", "NaiveRAG"])
        self.assertTrue(all(row["config_id"] == "full" for row in out))

    def test_filter_e1_ablation_table_uses_bpe_only(self) -> None:
        rows = [
            {"config_id": "full", "method": "GeoMosaic-HG BPE"},
            {"config_id": "source_news", "method": "Metadata++"},
            {"config_id": "modality_visual", "method": "GeoMosaic-HG BPE"},
        ]
        out = filter_e1_ablation_table(rows)
        self.assertEqual([row["config_id"] for row in out], ["full", "modality_visual"])
        self.assertTrue(all(row["method"] == "GeoMosaic-HG BPE" for row in out))

    def test_coverage_matrix_counts_event_source_modality_record_type(self) -> None:
        assets = [
            {"event_id": "crimea", "asset_source": "ACLED", "modality": "structured_event", "extra": {"record_type": "curated_conflict_event"}},
            {"event_id": "crimea", "asset_source": "Wikimedia Commons", "modality": "image_full", "extra": {"record_type": "wiki_page_asset"}},
            {"event_id": "ukraine", "asset_source": "ACLED", "modality": "structured_event", "extra": {"record_type": "curated_conflict_event"}},
        ]
        rows = coverage_matrix_rows(assets)
        self.assertIn(
            {
                "event_id": "crimea",
                "asset_source": "ACLED",
                "modality": "structured_event",
                "record_type": "curated_conflict_event",
                "count": 1,
            },
            rows,
        )

    def test_sort_by_bench_scale_orders_tiers(self) -> None:
        rows = [{"bench_label": "synthetic10x"}, {"bench_label": "tier1"}, {"bench_label": "synthetic2x"}]
        self.assertEqual([row["bench_label"] for row in sort_by_bench_scale(rows)], ["tier1", "synthetic2x", "synthetic10x"])

    def test_compact_e2_table_omits_pruning_metric_by_default(self) -> None:
        rows = [
            {
                "bench_label": "tier1",
                "method": "GeoMosaic-HG BPE",
                "p50_latency_ms_mean": "10",
                "candidate_reduction_rate_mean": "0.0",
                "candidate_count_mean": "100",
            }
        ]
        out = compact_e2_table(rows)
        self.assertIn("p50_latency_ms_mean", out[0])
        self.assertIn("candidate_count_mean", out[0])
        self.assertNotIn("candidate_reduction_rate_mean", out[0])

    def test_e4_pairwise_direction_reads_actual_summary_key(self) -> None:
        summary = {
            "pairwise_direction": {
                "same_direction_pairs": 107,
                "comparable_pairs": 116,
                "changed_direction_pairs": 9,
                "direction_consistency": 0.9224137931034483,
            }
        }
        out = e4_pairwise_direction(summary)
        self.assertEqual(out["same_direction_pairs"], 107)
        self.assertEqual(out["comparable_pairs"], 116)
        self.assertEqual(out["changed_direction_pairs"], 9)


if __name__ == "__main__":
    unittest.main()
