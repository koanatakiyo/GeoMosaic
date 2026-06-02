#!/usr/bin/env python3
"""Plan non-load-bearing metadata extraction tasks without calling an LLM."""

from __future__ import annotations

import argparse
import json

from geomosaic_hg.metadata_extraction_plan import add_plan_args, plan_metadata_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plan_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = plan_metadata_extraction(
        parsed_dir=args.parsed_dir,
        external_assets_path=args.external_assets,
        output_dir=args.output_dir,
        dry_run=True,
        model_id=args.model_id,
        max_full_text_chars=args.max_full_text_chars,
        batch_passages=args.batch_passages,
        prompt_version=args.prompt_version,
        schema_version=args.schema_version,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
