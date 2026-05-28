#!/usr/bin/env python3
"""Collect Tier 1 Wikimedia and ACLED EvidenceAsset JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from geomosaic_hg.external_assets import build_external_asset_plan, collect_existing_external_assets, normalize_event_ids, run_external_asset_collection
from geomosaic_hg.paths import DATA_DIR


def csv_events(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


def load_prior_summary_issues(merged_output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = merged_output.with_suffix(".summary.json")
    if not summary_path.exists():
        return [], []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []
    return list(summary.get("errors") or []), list(summary.get("warnings") or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="", help="Comma-separated event ids. Empty means all 8 Tier 1 events.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "0_external" / "external_asset_raw")
    parser.add_argument("--merged-output", type=Path, default=DATA_DIR / "0_external" / "external_assets.jsonl")
    parser.add_argument("--image-limit", type=int, default=3)
    parser.add_argument("--map-limit", type=int, default=1)
    parser.add_argument("--acled-limit", type=int, default=50)
    parser.add_argument("--acled-window-days", type=int, default=0, help="Days before/after each anchor date to include in ACLED queries.")
    parser.add_argument("--acled-max-pages", type=int, default=20, help="Maximum ACLED pages to fetch when paginating.")
    parser.add_argument("--wikimedia-delay-seconds", type=float, default=1.0, help="Minimum delay between Wikimedia API requests.")
    parser.add_argument("--wikimedia-retries", type=int, default=2, help="Retries for Wikimedia HTTP 429 responses.")
    parser.add_argument("--wikimedia-retry-backoff-seconds", type=float, default=30.0, help="Base backoff for Wikimedia HTTP 429 retries.")
    parser.add_argument("--download-wikimedia-images", action="store_true", help="Download Wikimedia image/* assets and point merged rows to local files.")
    parser.add_argument("--image-dir", type=Path, default=DATA_DIR / "0_external" / "event_images")
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--download-retries", type=int, default=2, help="Retries for Wikimedia image file downloads.")
    parser.add_argument("--download-retry-backoff-seconds", type=float, default=30.0, help="Base exponential backoff for Wikimedia image downloads.")
    parser.add_argument("--skip-wikimedia", action="store_true")
    parser.add_argument("--skip-acled", action="store_true")
    parser.add_argument("--acled-env-file", type=Path, default=None, help="Env-like file containing ACLED_EMAIL and ACLED_PASSWORD/ACLED_PWD.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first fetch error.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Tier 1 fetch plan without making network calls.")
    parser.add_argument("--collect-existing", type=Path, default=None, help="Merge existing EvidenceAsset JSONL from this directory without network calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.acled_env_file:
        load_env_file(args.acled_env_file)
    event_ids = normalize_event_ids(csv_events(args.events))
    if args.dry_run:
        plan = build_external_asset_plan(
            event_ids,
            image_limit=args.image_limit,
            map_limit=args.map_limit,
            acled_limit=args.acled_limit,
            acled_window_days=args.acled_window_days,
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.collect_existing:
        prior_errors, prior_warnings = load_prior_summary_issues(args.merged_output)
        summary = collect_existing_external_assets(
            args.collect_existing,
            args.merged_output,
            event_ids,
            prior_errors=prior_errors,
            prior_warnings=prior_warnings,
            acled_limit_hint=args.acled_limit,
            acled_window_days_hint=args.acled_window_days,
            download_wikimedia_images=args.download_wikimedia_images,
            image_dir=args.image_dir,
            download_timeout=args.download_timeout,
            download_retries=args.download_retries,
            download_retry_backoff_seconds=args.download_retry_backoff_seconds,
        )
    else:
        summary = run_external_asset_collection(
            output_dir=args.output_dir,
            merged_output=args.merged_output,
            event_ids=event_ids,
            image_limit=args.image_limit,
            map_limit=args.map_limit,
            acled_limit=args.acled_limit,
            acled_window_days=args.acled_window_days,
            acled_max_pages=args.acled_max_pages,
            wikimedia_delay_seconds=args.wikimedia_delay_seconds,
            wikimedia_retries=args.wikimedia_retries,
            wikimedia_retry_backoff_seconds=args.wikimedia_retry_backoff_seconds,
            download_wikimedia_images=args.download_wikimedia_images,
            image_dir=args.image_dir,
            download_timeout=args.download_timeout,
            download_retries=args.download_retries,
            download_retry_backoff_seconds=args.download_retry_backoff_seconds,
            skip_wikimedia=args.skip_wikimedia,
            skip_acled=args.skip_acled,
            continue_on_error=not args.fail_fast,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
