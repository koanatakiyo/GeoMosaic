#!/usr/bin/env python3
"""Fetch ACLED rows and write structured EvidenceAsset JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from geomosaic_hg.clients.acled import ACLEDClient, acled_row_to_asset
from geomosaic_hg.events import EVENTS
from geomosaic_hg.io import write_jsonl
from geomosaic_hg.paths import DATA_DIR
from geomosaic_hg.schema import as_clean_dict


EVENT_COUNTRY_HINTS = {
    "crimea": "Ukraine",
    "iraq": "Iraq",
    "libya": "Libya",
    "kosovo": "Kosovo",
    "jcpoa": "Iran",
    "ukraine": "Ukraine",
    "hongkong": "China",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, choices=sorted(EVENTS))
    parser.add_argument("--country", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--event-type", action="append", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = EVENTS[args.event]
    event_date = info.publish_time[:10]
    country = args.country or EVENT_COUNTRY_HINTS.get(args.event)
    start_date = args.start_date or event_date
    end_date = args.end_date or event_date
    fields = [
        "event_id_cnty",
        "event_date",
        "event_type",
        "sub_event_type",
        "actor1",
        "actor2",
        "location",
        "admin1",
        "country",
        "fatalities",
        "latitude",
        "longitude",
        "source",
        "notes",
    ]
    client = ACLEDClient()
    rows = client.events_for_window(
        country=country,
        start_date=start_date,
        end_date=end_date,
        event_types=args.event_type,
        fields=fields,
        limit=args.limit,
    )
    assets = [as_clean_dict(acled_row_to_asset(row, args.event)) for row in rows]
    output = args.output or DATA_DIR / "0_external" / f"acled_{args.event}.jsonl"
    count = write_jsonl(output, assets)
    print(f"wrote {count} assets to {output}")


if __name__ == "__main__":
    main()
