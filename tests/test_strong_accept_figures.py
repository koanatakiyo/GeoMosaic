from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.strong_accept_figures import (  # noqa: E402
    build_strong_accept_summary,
    comparison_delta,
    load_constraint_points,
)


METHODS = [
    "GeoMosaic-HG BPE",
    "Metadata++ + MMR",
    "Metadata++",
    "NaiveRAG",
    "Random-SM",
]


class StrongAcceptFiguresTest(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def make_reports_dir(self) -> tempfile.TemporaryDirectory[str]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)

        e1_rows = []
        for config in ["source_official", "source_news", "modality_visual", "modality_structured"]:
            for method in METHODS:
                vc = 1.0 if method == "GeoMosaic-HG BPE" else 0.9
                bal = 0.9 if method == "GeoMosaic-HG BPE" else 0.75
                e1_rows.append(
                    {
                        "config_id": config,
                        "method": method,
                        "viewpoint_coverage": vc,
                        "viewpoint_balance": bal,
                        "source_diversity": 5.0,
                        "modality_coverage": 4.0,
                    }
                )
        self.write_csv(root / "e1_e3_summary_by_config.csv", e1_rows)

        for filename, cands, reduction in [
            ("e2_latency_source_official_summary.csv", 125, 0.642857),
            ("e2_latency_source_news_summary.csv", 225, 0.357143),
            ("e2_latency_modality_visual_summary.csv", 303, 0.133929),
            ("e2_latency_modality_structured_summary.csv", 209, 0.401786),
        ]:
            rows = []
            for i, method in enumerate(METHODS):
                rows.append(
                    {
                        "bench_label": "tier1",
                        "method": method,
                        "p50_latency_ms_mean": 20 + i,
                        "p95_latency_ms_mean": 21 + i,
                        "candidate_reduction_rate_mean": 0.0 if method == "NaiveRAG" else reduction,
                        "candidate_keep_rate_mean": 1.0 if method == "NaiveRAG" else 1.0 - reduction,
                        "candidate_count_mean": 350 if method == "NaiveRAG" else cands,
                        "expanded_count_mean": cands if method == "GeoMosaic-HG BPE" else 100,
                    }
                )
            self.write_csv(root / filename, rows)

        scaling_rows = []
        pruning_rows = []
        for label, scale in [("tier1", 1), ("synthetic10x", 10)]:
            for method in METHODS:
                base = 100 * scale
                scaling_rows.append(
                    {
                        "bench_label": label,
                        "method": method,
                        "p50_latency_ms_mean": base,
                        "p95_latency_ms_mean": base + 1,
                        "candidate_count_mean": 2800 * scale,
                        "expanded_count_mean": 100,
                    }
                )
                pruning_rows.append(
                    {
                        "bench_label": label,
                        "method": method,
                        "p50_latency_ms_mean": base / 2 if method == "GeoMosaic-HG BPE" else base,
                        "p95_latency_ms_mean": base + 1,
                        "candidate_reduction_rate_mean": 0.0 if method == "NaiveRAG" else 0.642857,
                        "candidate_keep_rate_mean": 1.0 if method == "NaiveRAG" else 0.357143,
                        "candidate_count_mean": 2800 * scale if method == "NaiveRAG" else 1000 * scale,
                        "expanded_count_mean": 100,
                    }
                )
        self.write_csv(root / "e2_scaling_summary_by_method.csv", scaling_rows)
        self.write_csv(root / "e2_scaling_official_pruning_summary_by_method.csv", pruning_rows)
        self.addCleanup(tmp.cleanup)
        return tmp

    def test_constraint_points_join_e1_metrics_with_e2_latency(self) -> None:
        tmp = self.make_reports_dir()
        points = load_constraint_points(Path(tmp.name))
        self.assertEqual(len(points), 20)
        bpe_official = next(row for row in points if row["config_id"] == "source_official" and row["method"] == "GeoMosaic-HG BPE")
        self.assertEqual(bpe_official["viewpoint_coverage"], 1.0)
        self.assertEqual(bpe_official["candidate_count"], 125.0)

    def test_comparison_delta_uses_bpe_minus_baseline(self) -> None:
        tmp = self.make_reports_dir()
        points = load_constraint_points(Path(tmp.name))
        deltas = comparison_delta(points, baseline="Metadata++ + MMR")
        self.assertEqual(len(deltas), 4)
        self.assertAlmostEqual(deltas[0]["viewpoint_coverage_delta"], 0.1)
        self.assertAlmostEqual(deltas[0]["viewpoint_balance_delta"], 0.15)

    def test_summary_reports_pruning_and_speedup(self) -> None:
        tmp = self.make_reports_dir()
        summary = build_strong_accept_summary(Path(tmp.name))
        scaling = summary["scaling_pruning"]
        self.assertAlmostEqual(scaling["official_source_filter_aware_reduction_by_scale"]["10"], 0.642857)
        self.assertAlmostEqual(scaling["bpe_10x_official_vs_unconstrained_speedup"], 2.0)


if __name__ == "__main__":
    unittest.main()
