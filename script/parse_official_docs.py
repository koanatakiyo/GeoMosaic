#!/usr/bin/env python3
"""Parse manually materialized official documents into text and passages."""

from __future__ import annotations

import argparse
import json

from geomosaic_hg.official_doc_parsing import add_parse_args, parse_materialized_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return add_parse_args(parser).parse_args()


def main() -> None:
    args = parse_args()
    summary = parse_materialized_documents(
        args.manifest,
        args.output_dir,
        max_passage_chars=args.max_passage_chars,
        overlap_chars=args.overlap_chars,
        min_ok_chars=args.min_ok_chars,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
