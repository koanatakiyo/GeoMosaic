from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.efficiency_experiments import (  # noqa: E402
    METHOD_RUNNERS,
    BenchSpec,
    candidate_reduction_rate,
    csv_set,
    event_scope_size,
    parse_bench_dirs,
    percentile,
    run_efficiency_for_index,
    summarize_efficiency_rows,
)


class EfficiencyExperimentTest(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(percentile([5, 1, 3], 50), 3)
        self.assertEqual(percentile([5, 1, 3], 95), 5)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 80), 4)
        self.assertEqual(percentile([], 95), 0.0)

    def test_candidate_reduction_rate_handles_empty_scope(self) -> None:
        self.assertEqual(candidate_reduction_rate(candidate_count=25, scope_hyperedges=100), 0.75)
        self.assertEqual(candidate_reduction_rate(candidate_count=100, scope_hyperedges=100), 0.0)
        self.assertEqual(candidate_reduction_rate(candidate_count=0, scope_hyperedges=0), 0.0)

    def test_parse_bench_dirs_supports_labels(self) -> None:
        specs = parse_bench_dirs("tier1=data/enriched_full_bench,data/geomosaic_bench_synthetic_2x")
        self.assertEqual(specs[0].label, "tier1")
        self.assertEqual(specs[0].path.as_posix(), "data/enriched_full_bench")
        self.assertEqual(specs[1].label, "geomosaic_bench_synthetic_2x")

    def test_csv_set_trims_empty_values(self) -> None:
        self.assertEqual(csv_set(" official,news, ,wiki "), {"official", "news", "wiki"})
        self.assertEqual(csv_set(""), set())

    def test_event_scope_size_uses_global_scope_for_all(self) -> None:
        class FakeIndex:
            hyperedges = {"h1": {}, "h2": {}, "h3": {}}
            hyperedges_by_event = {"crimea": {"h1", "h2"}}

        self.assertEqual(event_scope_size(FakeIndex(), "crimea"), 2)
        self.assertEqual(event_scope_size(FakeIndex(), "all"), 3)

    def test_summarize_efficiency_rows_groups_by_bench_and_method(self) -> None:
        rows = [
            {"bench_label": "tier1", "method": "GeoMosaic-HG BPE", "p50_latency_ms": 10, "p95_latency_ms": 20, "candidate_reduction_rate": 0.5},
            {"bench_label": "tier1", "method": "GeoMosaic-HG BPE", "p50_latency_ms": 30, "p95_latency_ms": 40, "candidate_reduction_rate": 0.75},
            {"bench_label": "tier1", "method": "NaiveRAG", "p50_latency_ms": 5, "p95_latency_ms": 8, "candidate_reduction_rate": 0.25},
        ]

        summary = summarize_efficiency_rows(rows)
        by_key = {(row["bench_label"], row["method"]): row for row in summary}
        bpe = by_key[("tier1", "GeoMosaic-HG BPE")]

        self.assertEqual(bpe["rows_n"], 2)
        self.assertEqual(bpe["p50_latency_ms_mean"], 20.0)
        self.assertEqual(bpe["p95_latency_ms_mean"], 30.0)
        self.assertEqual(bpe["candidate_reduction_rate_mean"], 0.625)

    def test_run_efficiency_passes_constraint_filters_to_method(self) -> None:
        class FakeIndex:
            hyperedges = {"h1": {}, "h2": {}, "h3": {}, "h4": {}}
            hyperedges_by_event = {}

        seen = {}

        def fake_runner(index, query, config):
            seen["source_layers"] = set(config.source_layers)
            seen["modalities"] = set(config.modalities)
            seen["evidence_roles"] = set(config.evidence_roles)
            seen["max_match_level"] = config.max_match_level
            return {
                "candidate_count": 2,
                "seed_count": 1,
                "expanded_count": 1,
                "selected_hyperedges": [],
            }

        original = METHOD_RUNNERS["NaiveRAG"]
        METHOD_RUNNERS["NaiveRAG"] = fake_runner
        try:
            rows = run_efficiency_for_index(
                FakeIndex(),
                spec=BenchSpec(label="fake", path=Path("fake")),
                scope="global",
                event_ids=[],
                query="test",
                cutoff=None,
                budget=1,
                ann_k=1,
                expansion_depth=0,
                source_layers={"official"},
                modalities={"text"},
                evidence_roles={"substantive"},
                max_match_level="L2",
                warmup=0,
                repeats=1,
                methods=["NaiveRAG"],
            )
        finally:
            METHOD_RUNNERS["NaiveRAG"] = original

        self.assertEqual(seen["source_layers"], {"official"})
        self.assertEqual(seen["modalities"], {"text"})
        self.assertEqual(seen["evidence_roles"], {"substantive"})
        self.assertEqual(seen["max_match_level"], "L2")
        self.assertEqual(rows[0]["candidate_reduction_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
