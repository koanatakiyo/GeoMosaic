#!/usr/bin/env python3
"""Fetch Wikimedia Commons metadata and write EvidenceAsset JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from geomosaic_hg.clients.wikimedia import WikimediaCommonsClient, wikimedia_file_to_asset
from geomosaic_hg.events import EVENTS
from geomosaic_hg.io import write_jsonl
from geomosaic_hg.paths import DATA_DIR
from geomosaic_hg.schema import as_clean_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, choices=sorted(EVENTS))
    parser.add_argument("--query", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--modality", default="image_full", choices=["image_full", "map_pointer"])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = args.query or f"{EVENTS[args.event].subject} map"
    output = args.output or DATA_DIR / "0_external" / f"wikimedia_{args.event}.jsonl"
    client = WikimediaCommonsClient()
    files = client.search_imageinfo(query, limit=args.limit)
    assets = [as_clean_dict(wikimedia_file_to_asset(file, args.event, args.modality)) for file in files]
    count = write_jsonl(output, assets)
    print(f"wrote {count} assets to {output}")


if __name__ == "__main__":
    main()
