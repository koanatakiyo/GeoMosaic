from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.visual_skeleton import (  # noqa: E402
    build_visual_manifest,
    coverage_rows,
    detect_mime_magic,
    resolve_local_image_path,
    summarize_manifest,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mP8z8AABQMBgAZyE1YAAAAASUVORK5CYII="
)


def write_test_png(path: Path) -> None:
    path.write_bytes(PNG_1X1)


class VisualSkeletonTest(unittest.TestCase):
    def test_resolve_local_image_path_prefers_existing_data_pointer_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct.png"
            write_test_png(direct)
            fallback = root / "images" / "crimea" / "asset_visual.png"
            fallback.parent.mkdir(parents=True)
            write_test_png(fallback)

            direct_asset = {"event_id": "crimea", "asset_id": "asset_visual", "url_or_pointer": direct.as_posix(), "extra": {}}
            fallback_asset = {"event_id": "crimea", "asset_id": "asset_visual", "url_or_pointer": "https://example.test/file.png", "extra": {}}

            self.assertEqual(resolve_local_image_path(direct_asset, root / "images"), direct)
            self.assertEqual(resolve_local_image_path(fallback_asset, root / "images"), fallback)

    def test_build_manifest_marks_full_image_pointer_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "bench"
            image_root = root / "event_images"
            output = root / "visual"
            external = root / "external"
            (bench).mkdir()
            local_image = image_root / "crimea" / "asset_wikimedia_crimea_image_full_demo.png"
            local_image.parent.mkdir(parents=True)
            write_test_png(local_image)

            assets = [
                {
                    "asset_id": "asset_wikimedia_crimea_image_full_demo",
                    "event_id": "crimea",
                    "modality": "image_full",
                    "asset_source": "Wikimedia Commons",
                    "source_layer": "wiki",
                    "viewpoint_origin": "all",
                    "url_or_pointer": "https://upload.wikimedia.org/demo.png",
                    "caption_or_transcript": "Crimea image",
                    "license_or_terms": "CC0",
                    "redistribution_flag": True,
                    "perceptual_hash": "abc",
                    "extra": {"collection_channel": "wikipedia_page_bound", "page_bound": True},
                },
                {
                    "asset_id": "asset_news_pointer",
                    "event_id": "crimea",
                    "modality": "image_restricted_pointer",
                    "asset_source": "Reuters",
                    "source_layer": "news",
                    "viewpoint_origin": "all",
                    "url_or_pointer": "https://example.test/restricted.jpg",
                    "caption_or_transcript": "restricted",
                    "license_or_terms": "restricted",
                    "redistribution_flag": False,
                    "perceptual_hash": "def",
                    "extra": {"collection_channel": "gdelt_doc_visual_gkg"},
                },
                {
                    "asset_id": "asset_wikimedia_crimea_image_full_missing",
                    "event_id": "crimea",
                    "modality": "image_full",
                    "asset_source": "Wikimedia Commons",
                    "source_layer": "wiki",
                    "viewpoint_origin": "all",
                    "url_or_pointer": "https://upload.wikimedia.org/missing.png",
                    "caption_or_transcript": "missing",
                    "license_or_terms": "CC0",
                    "redistribution_flag": True,
                    "perceptual_hash": "ghi",
                    "extra": {},
                },
                {
                    "asset_id": "asset_map_pointer_remote",
                    "event_id": "crimea",
                    "modality": "map_pointer",
                    "asset_source": "Wikipedia/Wikimedia pointer",
                    "source_layer": "wiki",
                    "viewpoint_origin": "all",
                    "url_or_pointer": "https://en.wikipedia.org/wiki/Crimea",
                    "caption_or_transcript": "remote pointer",
                    "license_or_terms": "pointer",
                    "redistribution_flag": False,
                    "perceptual_hash": "jkl",
                    "extra": {"generated_pointer": True},
                },
            ]
            with (bench / "evidence_assets.jsonl").open("w", encoding="utf-8") as f:
                for row in assets:
                    f.write(json.dumps(row) + "\n")

            rows, summary = build_visual_manifest(
                bench_dir=bench,
                image_root=image_root,
                output_dir=output,
                external_output_dir=external,
                backend="siglip",
                model_path=Path("/models/siglip"),
                metadata_only=True,
            )

            by_id = {row["asset_id"]: row for row in rows}
            self.assertEqual(by_id["asset_wikimedia_crimea_image_full_demo"]["visual_status"], "full_image")
            self.assertEqual(by_id["asset_wikimedia_crimea_image_full_demo"]["width"], 1)
            self.assertEqual(by_id["asset_wikimedia_crimea_image_full_demo"]["height"], 1)
            self.assertEqual(by_id["asset_news_pointer"]["visual_status"], "pointer_only")
            self.assertEqual(by_id["asset_wikimedia_crimea_image_full_missing"]["visual_status"], "missing_file")
            self.assertEqual(by_id["asset_map_pointer_remote"]["visual_status"], "pointer_only")
            self.assertEqual(summary["embedding_rows"], 0)

    def test_summary_and_coverage_rows_count_statuses(self) -> None:
        rows = [
            {"event_id": "crimea", "modality": "image_full", "visual_status": "full_image"},
            {"event_id": "crimea", "modality": "map_pointer", "visual_status": "full_image"},
            {"event_id": "ukraine", "modality": "image_restricted_pointer", "visual_status": "pointer_only"},
        ]

        summary = summarize_manifest(rows, embedding_rows=2, backend="clip", model_path=Path("/models/clip"), embedding_path=Path("/tmp/e.npy"))
        self.assertEqual(summary["visual_assets_total"], 3)
        self.assertEqual(summary["by_status"], {"full_image": 2, "pointer_only": 1})
        self.assertEqual(summary["embedding_rows"], 2)

        coverage = coverage_rows(rows)
        self.assertIn({"event_id": "crimea", "modality": "image_full", "visual_status": "full_image", "count": 1}, coverage)

    def test_detect_mime_magic_uses_bytes_not_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.svg"
            write_test_png(path)
            self.assertEqual(detect_mime_magic(path), "image/png")


if __name__ == "__main__":
    unittest.main()
