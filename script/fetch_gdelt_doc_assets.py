#!/usr/bin/env python3
"""Fetch GDELT DOC article pointers and write EvidenceAsset JSONL."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from geomosaic_hg.clients.gdelt import GDELTDOCClient, gdelt_doc_article_to_asset, gdelt_doc_article_to_image_asset
from geomosaic_hg.events import EVENTS
from geomosaic_hg.io import write_jsonl
from geomosaic_hg.paths import DATA_DIR
from geomosaic_hg.schema import as_clean_dict


GDELT_DOC_EVENT_QUERIES = {
    "crimea": '"Crimea" ("annexation" OR "referendum")',
    "iraq": '"2003 invasion of Iraq"',
    "libya": '"Libya" ("NATO intervention" OR "Operation Unified Protector")',
    "kosovo": '"Kosovo declaration of independence"',
    "scs": '"South China Sea Arbitration"',
    "jcpoa": '"JCPOA"',
    "ukraine": '"Ukraine" ("invasion" OR "sovereignty" OR "territorial integrity")',
    "hongkong": '"Hong Kong national security law"',
}


def gdelt_datetime(day: date, end_of_day: bool = False) -> str:
    suffix = "235959" if end_of_day else "000000"
    return f"{day.strftime('%Y%m%d')}{suffix}"


def default_doc_window(event_id: str, window_days: int) -> tuple[str | None, str | None, str | None, str]:
    event_date = date.fromisoformat(EVENTS[event_id].publish_time[:10])
    delta = timedelta(days=max(0, window_days))
    return gdelt_datetime(event_date - delta), gdelt_datetime(event_date + delta, end_of_day=True), None, "event_window"


def gdelt_doc_articles_to_assets(articles: list[dict], event_id: str, query: str, temporal_relation: str) -> list[dict]:
    assets = []
    for article in articles:
        assets.append(as_clean_dict(gdelt_doc_article_to_asset(article, event_id, query=query, temporal_relation=temporal_relation)))
        image_asset = gdelt_doc_article_to_image_asset(article, event_id, query=query, temporal_relation=temporal_relation)
        if image_asset is not None:
            assets.append(as_clean_dict(image_asset))
    return assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, choices=sorted(EVENTS))
    parser.add_argument("--query", default=None)
    parser.add_argument("--max-records", type=int, default=25)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--start-datetime", default=None, help="GDELT DOC startdatetime in YYYYMMDDHHMMSS.")
    parser.add_argument("--end-datetime", default=None, help="GDELT DOC enddatetime in YYYYMMDDHHMMSS.")
    parser.add_argument("--timespan", default=None, help="Optional GDELT DOC timespan such as 3months. Overrides the default event-window query.")
    parser.add_argument("--temporal-relation", default=None, help="Override temporal relation metadata.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gdelt-delay-seconds", type=float, default=5.0, help="Minimum delay between GDELT requests inside one process.")
    parser.add_argument("--gdelt-retries", type=int, default=3, help="Retries for transient GDELT request or response errors.")
    parser.add_argument("--gdelt-retry-backoff-seconds", type=float, default=15.0, help="Initial retry backoff for transient GDELT errors.")
    parser.add_argument("--gdelt-timeout-seconds", type=int, default=30, help="Per-request GDELT timeout in seconds.")
    parser.add_argument("--sort", default="hybridrel", help="GDELT DOC sort mode. Use an empty string to omit sort.")
    parser.add_argument("--strict-json", action="store_true", help="Fail if GDELT returns non-JSON after retries instead of writing an empty JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_start, default_end, default_timespan, default_relation = default_doc_window(args.event, args.window_days)
    start_datetime = args.start_datetime or default_start
    end_datetime = args.end_datetime or default_end
    timespan = args.timespan if args.timespan is not None else default_timespan
    temporal_relation = args.temporal_relation or ("custom_window" if args.start_datetime or args.end_datetime else default_relation)
    query = args.query or GDELT_DOC_EVENT_QUERIES.get(args.event) or EVENTS[args.event].subject
    client = GDELTDOCClient(
        timeout=args.gdelt_timeout_seconds,
        rate_limit_seconds=args.gdelt_delay_seconds,
        max_retries=args.gdelt_retries,
        retry_backoff_seconds=args.gdelt_retry_backoff_seconds,
        invalid_json_as_empty=not args.strict_json,
    )
    articles = client.search_articles(
        query,
        max_records=args.max_records,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        timespan=timespan,
        sort=args.sort or None,
    )
    if client.last_empty_reason:
        print(f"warning: GDELT returned non-JSON after retries; treating as empty result: {client.last_empty_reason}", file=sys.stderr)
    assets = gdelt_doc_articles_to_assets(articles, args.event, query=query, temporal_relation=temporal_relation)
    output = args.output or DATA_DIR / "0_external" / "external_asset_raw" / f"gdelt_doc_{args.event}.jsonl"
    count = write_jsonl(output, assets)
    print(f"wrote {count} GDELT DOC pointer assets to {output}")


if __name__ == "__main__":
    main()
