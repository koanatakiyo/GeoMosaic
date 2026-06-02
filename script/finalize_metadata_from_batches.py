#!/usr/bin/env python3
"""Finalize failed Stage B map-reduce tasks from completed batch metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.metadata_extraction import finalize_failed_map_reduce_from_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/1_intermediate/metadata_extraction/metadata_extraction_tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/1_intermediate/metadata_extraction/official_doc_metadata_enrichment.jsonl"))
    parser.add_argument("--batch-output", type=Path, default=Path("data/1_intermediate/metadata_extraction/official_doc_metadata_batch_outputs.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/1_intermediate/metadata_extraction/official_doc_metadata_finalize_summary.json"))
    parser.add_argument("--max-items", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = finalize_failed_map_reduce_from_batches(
        tasks_path=args.tasks,
        output_path=args.output,
        batch_output_path=args.batch_output,
        summary_path=args.summary,
        max_items=args.max_items,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
