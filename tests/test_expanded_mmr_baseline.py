from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.baselines import smpi_expanded_mmr  # noqa: E402
from geomosaic_hg.bpe import BPEConfig  # noqa: E402


class ExpandedMmrBaselineTest(unittest.TestCase):
    def test_expanded_mmr_uses_same_expanded_pool_as_bpe(self) -> None:
        class FakeIndex:
            evidence_assets = {}
            source_records = {}

            def prune_candidates(self, source_layers, modalities, cutoff, constraints, event_ids):
                return [
                    {
                        "hyperedge_id": "h_seed",
                        "source_layer_set": ["official"],
                        "viewpoint_origin_set": ["US"],
                        "modality_set": ["text"],
                    },
                    {
                        "hyperedge_id": "h_expanded",
                        "source_layer_set": ["news"],
                        "viewpoint_origin_set": ["EU"],
                        "modality_set": ["image_full"],
                    },
                ]

            def ann_seeds(self, query, candidates, k):
                row = dict(candidates[0])
                row["_rel"] = 0.9
                return [row]

            def expand_seeds(self, seeds, candidates, depth):
                expanded_seed = dict(seeds[0])
                expanded_neighbor = dict(candidates[1])
                expanded_neighbor["_rel"] = 0.2
                return [expanded_seed, expanded_neighbor]

        config = BPEConfig(hyperedge_budget=2, ann_k=1, expansion_depth=1)
        result = smpi_expanded_mmr(FakeIndex(), "query", config)

        self.assertEqual(result["method"], "SMPI-Expanded + MMR")
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["seed_count"], 1)
        self.assertEqual(result["expanded_count"], 2)
        self.assertEqual([h["hyperedge_id"] for h in result["selected_hyperedges"]], ["h_seed", "h_expanded"])


if __name__ == "__main__":
    unittest.main()
