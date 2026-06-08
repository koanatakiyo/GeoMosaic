#!/usr/bin/env python3
"""Remove selected E4 scorer rows from the shared result file after backing them up."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from geomosaic_hg.direct_scoring import reset_direct_scoring_model_rows  # noqa: E402
from geomosaic_hg.io import read_jsonl  # noqa: E402


def csv_models(value: str) -> set[str]:
    models = {item.strip() for item in value.split(",") if item.strip()}
    if not models:
        raise SystemExit("--models must include at least one model")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/1_intermediate/direct_scoring/direct_scoring_results_six_scorers.jsonl")
    parser.add_argument("--models", required=True, help="Comma-separated models to remove, e.g. deepseek,qwen")
    parser.add_argument("--backup", default=None, help="Backup JSONL for removed rows. Default is timestamped beside output.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = csv_models(args.models)
    rows = list(read_jsonl(args.output))
    counts = Counter(str(row.get("model", "")) for row in rows if str(row.get("model", "")) in models)
    backup = args.backup or f"{args.output}.reset_{'_'.join(sorted(models))}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
    summary = {
        "output": args.output,
        "models": sorted(models),
        "rows_matching": dict(sorted(counts.items())),
        "backup": backup,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return
    result = reset_direct_scoring_model_rows(args.output, models=models, backup_path=backup)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
