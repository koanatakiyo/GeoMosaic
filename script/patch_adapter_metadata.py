#!/usr/bin/env python3
"""Backfill unified adapter metadata into existing EvidenceAsset JSONL files.

Adds to extra{} (without overwriting existing keys):
  collection_channel       - already present in wikimedia; set for ACLED
  record_type              - "wiki_page_asset" | "curated_conflict_event" | ...
  curation_level           - "community_curated" | "human_curated" | "official"
  source_temporal_coverage - copied from extra.temporal_status if present
  active_policy            - "primary_image_evidence" for wikimedia; "optional_enrichment" for ACLED

ACLED is marked optional_enrichment because 5/8 events return 0 rows;
evaluation reports coverage per source adapter, not a pooled total.

Targets:
  data/0_external/external_asset_raw/wikimedia_*.jsonl
  data/0_external/external_asset_raw/acled_*.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RAW_DIR = Path(__file__).parent.parent / "data" / "0_external" / "external_asset_raw"

# Per-source adapter metadata defaults.
# Keys that already exist in extra{} are NOT overwritten.
_WIKIMEDIA_DEFAULTS = {
    "collection_channel": "wikipedia_page_bound",  # usually already present
    "record_type": "wiki_page_asset",
    "curation_level": "community_curated",
    "active_policy": "primary_image_evidence",
}

_ACLED_DEFAULTS = {
    "collection_channel": "acled_api",
    "record_type": "curated_conflict_event",
    "curation_level": "human_curated",
    "active_policy": "optional_enrichment",
    "source_temporal_coverage": "event_window",
}

_OFFICIAL_DEFAULTS = {
    "collection_channel": "official_registry",
    "record_type": "official_document",
    "curation_level": "official",
    "active_policy": "primary_official_evidence",
}


def _patch_extra(extra: dict, defaults: dict) -> bool:
    changed = False
    for key, val in defaults.items():
        if key not in extra:
            extra[key] = val
            changed = True
    return changed


def _source_temporal_coverage_from_extra(extra: dict) -> str | None:
    # wikimedia entries already have extra.temporal_status; map to our field name.
    ts = extra.get("temporal_status")
    if ts:
        return ts
    return None


def patch_file(path: Path, source_type: str, dry_run: bool = False) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    patched_lines: list[str] = []
    changed_count = 0

    if source_type == "wikimedia":
        defaults = _WIKIMEDIA_DEFAULTS
    elif source_type == "acled":
        defaults = _ACLED_DEFAULTS
    else:
        defaults = _OFFICIAL_DEFAULTS

    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        extra = obj.setdefault("extra", {})

        changed = _patch_extra(extra, defaults)
        if source_type == "wikimedia" and extra.get("active_policy") == "primary":
            extra["active_policy"] = "primary_image_evidence"
            changed = True
        if source_type == "official" and extra.get("active_policy") == "primary":
            extra["active_policy"] = "primary_official_evidence"
            changed = True

        # For wikimedia: mirror temporal_status → source_temporal_coverage if missing
        if source_type == "wikimedia" and "source_temporal_coverage" not in extra:
            cov = _source_temporal_coverage_from_extra(extra)
            if cov:
                extra["source_temporal_coverage"] = cov
                changed = True

        if changed:
            changed_count += 1

        patched_lines.append(json.dumps(obj, ensure_ascii=False))

    if not dry_run:
        path.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")
    return changed_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="Print counts only, don't write.")
    p.add_argument("--dir", type=Path, default=RAW_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.dir

    total = 0
    for pattern, source_type in [("wikimedia_*.jsonl", "wikimedia"),
                                  ("acled_*.jsonl", "acled"),
                                  ("official_*.jsonl", "official")]:
        for fpath in sorted(raw_dir.glob(pattern)):
            n = patch_file(fpath, source_type, dry_run=args.dry_run)
            label = "(dry-run) " if args.dry_run else ""
            print(f"{label}{fpath.name}: {n} entries patched")
            total += n

    print(f"\ntotal patched: {total} entries")


if __name__ == "__main__":
    main()
