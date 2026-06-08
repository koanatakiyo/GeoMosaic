from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.relevance_audit import (  # noqa: E402
    dcg,
    ndcg_at_k,
    select_audit_ids,
    summarize_relevance,
)


class RelevanceAuditTest(unittest.TestCase):
    def test_select_audit_ids_round_robins_by_rank_method_and_event(self) -> None:
        memberships = []
        for event_id in ["crimea", "ukraine"]:
            for method in ["GeoMosaic-HG BPE", "Metadata++ + MMR", "Metadata++"]:
                for rank in range(1, 7):
                    memberships.append(
                        {
                            "event_id": event_id,
                            "method": method,
                            "rank": rank,
                            "audit_id": f"{event_id}_{method}_{rank}",
                        }
                    )

        selected = select_audit_ids(memberships, event_ids=["crimea", "ukraine"], methods=["GeoMosaic-HG BPE", "Metadata++ + MMR", "Metadata++"], max_pairs=30)

        self.assertEqual(len(selected), 30)
        self.assertIn("crimea_GeoMosaic-HG BPE_1", selected)
        self.assertIn("ukraine_Metadata++_5", selected)
        self.assertNotIn("crimea_GeoMosaic-HG BPE_6", selected)

    def test_dcg_and_ndcg_use_graded_relevance(self) -> None:
        self.assertAlmostEqual(dcg([2, 1, 0], 3), 2 + 1 / 1.5849625007211563)
        self.assertAlmostEqual(ndcg_at_k([2, 1, 0], 3), 1.0)
        self.assertEqual(ndcg_at_k([0, 0, 0], 3), 0.0)

    def test_summarize_relevance_requires_complete_top_k_labels(self) -> None:
        labels = {
            "a": {"human_relevance": "2"},
            "b": {"human_relevance": "1"},
            "c": {"human_relevance": "0"},
            "d": {"human_relevance": "2"},
            "e": {"human_relevance": "1"},
        }
        memberships = [
            {"event_id": "crimea", "method": "GeoMosaic-HG BPE", "rank": str(rank), "audit_id": audit_id}
            for rank, audit_id in enumerate(["a", "b", "c", "d", "e"], start=1)
        ]

        rows = summarize_relevance(labels, memberships, k=5, relevance_threshold=1)
        row = rows[0]

        self.assertEqual(row["event_id"], "crimea")
        self.assertEqual(row["method"], "GeoMosaic-HG BPE")
        self.assertEqual(row["labeled_at_k"], 5)
        self.assertAlmostEqual(row["p_at_k"], 0.8)
        self.assertAlmostEqual(row["mean_relevance"], 1.2)
        self.assertGreater(row["ndcg_at_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
