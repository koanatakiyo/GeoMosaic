#!/usr/bin/env python3
"""Export a Stage C human-audit queue with model-disagreement and QA flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomosaic_hg.claim_grounding import export_claim_grounding_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/claim_grounding_tasks.jsonl"),
    )
    parser.add_argument(
        "--judgments",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/claim_grounding_judgments.jsonl"),
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/claim_grounding_adjudication.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/1_intermediate/claim_grounding/human_audit"),
    )
    parser.add_argument("--hk-insufficient-sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_claim_grounding_audit(
        tasks_path=args.tasks,
        judgments_path=args.judgments,
        adjudication_path=args.adjudication,
        output_dir=args.output_dir,
        hk_insufficient_sample_size=args.hk_insufficient_sample_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
