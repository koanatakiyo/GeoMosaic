#!/usr/bin/env python3
"""Run deterministic QA checks over parsed official documents."""

from __future__ import annotations

import argparse
import json

from geomosaic_hg.official_doc_qa import add_qa_args, qa_parsed_official_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_qa_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (args.parsed_dir / "parse_qa_report.json")
    report = qa_parsed_official_docs(
        args.parsed_dir,
        output_path=output,
        short_doc_chars=args.short_doc_chars,
        max_replacement_ratio=args.max_replacement_ratio,
        max_control_ratio=args.max_control_ratio,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
