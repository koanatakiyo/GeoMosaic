#!/usr/bin/env python3
"""Fetch GDELT DOC article pointers and write EvidenceAsset JSONL."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from geomosaic_hg.clients.gdelt import GDELTDOCClient, gdelt_doc_article_to_asset
from geomosaic_hg.events import EVENTS
from geomosaic_hg.io import write_jsonl
from geomosaic_hg.paths import DATA_DIR
from geomosaic_hg.schema import as_clean_dict


GDELT_DOC_START_DATE = date(2017, 1, 1)

GDELT_DOC_EVENT_QUERIES = {
    "crimea": '"Crimea" ("annexation" OR "referendum")',
    "iraq": '"2003 invasion of Iraq" OR "Iraq War"',
    "libya": '"Libya" ("NATO intervention" OR "Operation Unified Protector")',
    "kosovo": '"Kosovo declaration of independence"',
    "scs": '"South China Sea Arbitration" OR "South China Sea territorial disputes"',
    "jcpoa": '"JCPOA" OR "Iran nuclear deal"',
    "ukraine": '"Ukraine" ("invasion" OR "sovereignty" OR "territorial integrity")',
    "hongkong": '"Hong Kong national security law"',
}


def gdelt_datetime(day: date, end_of_day: bool = False) -> str:
    suffix = "235959" if end_of_day else "000000"
    return f"{day.strftime('%Y%m%d')}{suffix}"


def default_doc_window(event_id: str, window_days: int) -> tuple[str | None, str | None, str | None, str]:
    event_date = date.fromisoformat(EVENTS[event_id].publish_time[:10])
    if event_date >= GDELT_DOC_START_DATE:
        delta = timedelta(days=max(0, window_days))
        return gdelt_datetime(event_date - delta), gdelt_datetime(event_date + delta, end_of_day=True), None, "event_window"
    return None, None, "3months", "retrospective_recent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, choices=sorted(EVENTS))
    parser.add_argument("--query", default=None)
    parser.add_argument("--max-records", type=int, default=25)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--start-datetime", default=None, help="GDELT DOC startdatetime in YYYYMMDDHHMMSS.")
    parser.add_argument("--end-datetime", default=None, help="GDELT DOC enddatetime in YYYYMMDDHHMMSS.")
    parser.add_argument("--timespan", default=None, help="GDELT DOC timespan such as 3months. Used for retrospective old-event pointers.")
    parser.add_argument("--temporal-relation", default=None, help="Override temporal relation metadata.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gdelt-delay-seconds", type=float, default=5.0, help="Minimum delay between GDELT requests inside one process.")
    parser.add_argument("--gdelt-retries", type=int, default=3, help="Retries for GDELT HTTP 429 rate-limit responses.")
    parser.add_argument("--gdelt-retry-backoff-seconds", type=float, default=15.0, help="Initial retry backoff for GDELT HTTP 429 responses.")
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
        rate_limit_seconds=args.gdelt_delay_seconds,
        max_retries=args.gdelt_retries,
        retry_backoff_seconds=args.gdelt_retry_backoff_seconds,
    )
    articles = client.search_articles(
        query,
        max_records=args.max_records,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        timespan=timespan,
        sort="hybridrel",
    )
    assets = [
        as_clean_dict(gdelt_doc_article_to_asset(article, args.event, query=query, temporal_relation=temporal_relation))
        for article in articles
    ]
    output = args.output or DATA_DIR / "0_external" / "external_asset_raw" / f"gdelt_doc_{args.event}.jsonl"
    count = write_jsonl(output, assets)
    print(f"wrote {count} GDELT DOC pointer assets to {output}")


if __name__ == "__main__":
    main()
