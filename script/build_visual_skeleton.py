#!/usr/bin/env python3
"""Build a visual asset manifest and optional CLIP/SigLIP embedding matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.visual_skeleton import build_visual_manifest


DEFAULT_MODEL_PATHS = {
    "siglip": Path("/data/yandan/models/siglip-base-patch16-224"),
    "clip": Path("/data/yandan/models/clip-vit-large-patch14"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=Path("data/enriched_full_bench"))
    parser.add_argument("--image-root", type=Path, default=Path("data/0_external/event_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/1_intermediate/visual_index"))
    parser.add_argument("--external-output-dir", type=Path, default=Path("/data/yandan/geomosaic_visual_index"))
    parser.add_argument("--backend", choices=sorted(DEFAULT_MODEL_PATHS), default="siglip")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--metadata-only", action="store_true", help="Build manifest/coverage without loading an embedding model.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", help="Embedding device, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path or DEFAULT_MODEL_PATHS[args.backend]
    _, summary = build_visual_manifest(
        bench_dir=args.bench_dir,
        image_root=args.image_root,
        output_dir=args.output_dir,
        external_output_dir=args.external_output_dir,
        backend=args.backend,
        model_path=model_path,
        metadata_only=args.metadata_only,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
