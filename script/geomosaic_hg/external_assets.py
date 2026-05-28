"""Tier 1 external asset collection helpers."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from .clients.acled import ACLEDClient, acled_row_to_asset
from .clients.http import DEFAULT_USER_AGENT
from .clients.wikimedia import WikimediaCommonsClient, wikimedia_file_to_asset
from .events import EVENTS
from .io import read_jsonl, sha256_file, stable_hash, write_json, write_jsonl
from .paths import DATA_DIR, relative_to_project
from .schema import EvidenceAsset, as_clean_dict, dataclass_from_dict
from .wiki_pages import WIKIPEDIA_EVENT_FALLBACK_PAGES, WIKIPEDIA_EVENT_PAGES, wikipedia_page_url


TIER1_EVENT_IDS = tuple(EVENTS)


def progress(message: str) -> None:
    print(f"[external-assets] {message}", file=sys.stderr, flush=True)

EVENT_COUNTRY_HINTS = {
    "crimea": "Ukraine",
    "iraq": "Iraq",
    "libya": "Libya",
    "kosovo": "Kosovo",
    "jcpoa": "Iran",
    "ukraine": "Ukraine",
    "hongkong": "China",
}

ACLED_COVERAGE_NOTES = {
    "crimea": "Ukraine ACLED coverage begins in 1/2018, so the 2014 Crimea anchor date is outside country coverage.",
    "iraq": "Iraq ACLED coverage begins in 1/2016, so the 2003 Iraq anchor date is outside country coverage.",
    "kosovo": "Kosovo ACLED coverage begins in 1/2018, so the 2008 Kosovo anchor date is outside country coverage.",
    "jcpoa": "Iran ACLED coverage begins in 1/2016, so the 2015 JCPOA anchor date is outside country coverage; JCPOA is also a diplomatic agreement rather than a conflict-event anchor.",
    "scs": "South China Sea is multi-country and has no single ACLED country filter; China ACLED coverage begins in 1/2018, after the 2016 arbitration anchor date.",
}

ACLED_FIELDS = [
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

EVENT_IMAGE_TERMS = {
    "crimea": ("annexation", "referendum", "2014", "russian", "ukrainian", "military", "soldier", "sevastopol"),
    "iraq": ("invasion", "iraq war", "2003", "baghdad", "military", "soldier", "marine"),
    "libya": ("intervention", "civil war", "2011", "nato", "military", "no-fly", "operation unified protector", "odyssey dawn", "libyan"),
    "kosovo": ("declaration", "independence", "2008", "newborn", "pavaresise", "recognition", "recognizing"),
    "scs": ("arbitration", "claims", "claim", "territorial", "dispute", "nine-dash", "south china sea"),
    "jcpoa": ("jcpoa", "joint comprehensive", "iran nuclear", "agreement", "deal", "zarif", "kerry", "2015"),
    "ukraine": ("invasion", "war", "2022", "russian", "military", "kyiv", "donbas", "oblast"),
    "hongkong": ("national security", "protest", "police", "2020", "demonstration", "hong kong"),
}

MAP_TERMS = ("map", "territorial", "claims", "claim", "recognition", "recognizing", "control", "phase", "belligerents", "countries supplying")

GENERIC_IMAGE_TERMS = (
    "clouds",
    "waterfront",
    "railway station",
    "locomotive",
    "mountain",
    "beach",
    "landscape",
    "sunset",
    "flag of",
    "coat of arms",
    "ensign",
    "logo",
    "signature",
    "location dot",
    "symbol",
    "naval jack",
)

IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/tiff": ".tif",
}

INACTIVE_ASSET_STATUSES = {"candidate", "inactive", "dropped", "obsolete", "decorative_dropped"}


def normalize_event_ids(event_ids: Iterable[str] | None = None) -> list[str]:
    if event_ids is None:
        return list(TIER1_EVENT_IDS)
    out = []
    for event_id in event_ids:
        if event_id not in EVENTS:
            raise ValueError(f"Unknown event_id={event_id}")
        out.append(event_id)
    return out


def event_window(event_id: str, window_days: int = 0) -> tuple[str, str]:
    center = date.fromisoformat(EVENTS[event_id].publish_time[:10])
    delta = timedelta(days=max(0, window_days))
    return (center - delta).isoformat(), (center + delta).isoformat()


def acled_skip_reason(event_id: str) -> str:
    if EVENT_COUNTRY_HINTS.get(event_id):
        return ""
    return ACLED_COVERAGE_NOTES.get(event_id, "No ACLED country filter is configured.")


def acled_result_warnings(event_id: str, country: str | None, start_date: str, end_date: str, row_count: int, limit: int) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if not country:
        warnings.append(
            {
                "event_id": event_id,
                "source": "acled",
                "warning": acled_skip_reason(event_id),
            }
        )
        return warnings
    if row_count == 0:
        warnings.append(
            {
                "event_id": event_id,
                "source": "acled",
                "warning": (
                    f"ACLED returned 0 rows for country={country} window={start_date}/{end_date}. "
                    f"{ACLED_COVERAGE_NOTES.get(event_id, 'Check the country/date query strategy.')}"
                ),
            }
        )
    if limit > 0 and row_count == limit:
        warnings.append(
            {
                "event_id": event_id,
                "source": "acled",
                "warning": f"ACLED returned {row_count} rows, hitting limit={limit}; data may be truncated.",
            }
        )
    return warnings


def dedupe_issue_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("event_id", "")),
            str(row.get("source", "")),
            str(row.get("error", "")),
            str(row.get("warning", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def infer_acled_raw_warnings(
    input_dir: Path,
    event_ids: Iterable[str] | None = None,
    limit_hint: int | None = None,
    window_days_hint: int = 0,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for event_id in normalize_event_ids(event_ids):
        path = input_dir / f"acled_{event_id}.jsonl"
        if not path.exists():
            continue
        row_count = sum(1 for _ in read_jsonl(path))
        country = EVENT_COUNTRY_HINTS.get(event_id)
        start_date, end_date = event_window(event_id, window_days_hint)
        warnings.extend(acled_result_warnings(event_id, country, start_date, end_date, row_count, limit_hint or 0))
    return warnings


def build_external_asset_plan(
    event_ids: Iterable[str] | None = None,
    image_limit: int = 3,
    map_limit: int = 1,
    acled_limit: int = 50,
    acled_window_days: int = 0,
) -> dict[str, Any]:
    events = []
    for event_id in normalize_event_ids(event_ids):
        info = EVENTS[event_id]
        start_date, end_date = event_window(event_id, acled_window_days)
        country = EVENT_COUNTRY_HINTS.get(event_id)
        events.append(
            {
                "event_id": event_id,
                "event_name": info.name,
                "subject": info.subject,
                "wikimedia_image_query": f"{info.subject} geopolitical event",
                "wikimedia_map_query": f"{info.subject} map",
                "wikimedia_page_title": WIKIPEDIA_EVENT_PAGES.get(event_id, ""),
                "wikimedia_fallback_page_titles": list(WIKIPEDIA_EVENT_FALLBACK_PAGES.get(event_id, ())),
                "wikimedia_image_limit": image_limit,
                "wikimedia_map_limit": map_limit if info.map_eligible else 0,
                "acled_country": country,
                "acled_start_date": start_date,
                "acled_end_date": end_date,
                "acled_limit": acled_limit,
                "acled_skip_reason": acled_skip_reason(event_id) if not country else "",
                "acled_coverage_note": ACLED_COVERAGE_NOTES.get(event_id, ""),
            }
        )
    return {"events": events}


def fetch_wikimedia_for_event(client: WikimediaCommonsClient, event_id: str, image_limit: int = 3, map_limit: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_title = WIKIPEDIA_EVENT_PAGES.get(event_id)
    if not page_title:
        return rows

    candidate_limit = max(50, (image_limit + map_limit) * 10)
    files_by_title: dict[str, tuple[Any, dict[str, Any]]] = {}
    for title in (page_title, *WIKIPEDIA_EVENT_FALLBACK_PAGES.get(event_id, ())):
        progress(f"wikimedia {event_id}: reading Wikipedia page images for {title!r}")
        page_context = wikipedia_page_context(client, title)
        for file in client.page_imageinfo(title, limit=candidate_limit):
            files_by_title.setdefault(file.title, (file, page_context))
    files = rank_wikimedia_files([file for file, _ in files_by_title.values()], event_id)
    selected_titles: set[str] = set()

    for file in files:
        if len([row for row in rows if row.get("modality") == "image_full"]) >= image_limit:
            break
        if is_map_like_wikimedia_file(file):
            continue
        rows.append(wikipedia_page_bound_asset_row(file, event_id, "image_full", files_by_title[file.title][1]))
        selected_titles.add(file.title)

    if EVENTS[event_id].map_eligible and map_limit > 0:
        for file in files:
            if len([row for row in rows if row.get("modality") == "map_pointer"]) >= map_limit:
                break
            if file.title in selected_titles or not is_map_like_wikimedia_file(file):
                continue
            rows.append(wikipedia_page_bound_asset_row(file, event_id, "map_pointer", files_by_title[file.title][1]))
            selected_titles.add(file.title)
    rows = dedupe_assets(rows)
    progress(f"wikimedia {event_id}: selected {len(rows)} page-bound assets")
    return rows


def wikipedia_page_context(client: WikimediaCommonsClient, page_title: str) -> dict[str, Any]:
    if hasattr(client, "page_metadata"):
        try:
            metadata = client.page_metadata(page_title)
            if metadata:
                return {
                    "source_page_title": metadata.get("title") or page_title,
                    "source_page_url": metadata.get("url") or wikipedia_page_url(page_title),
                    "source_page_id": metadata.get("page_id"),
                    "source_page_revision_id": metadata.get("revision_id"),
                    "source_page_revision_time": metadata.get("revision_time", ""),
                }
        except Exception:
            pass
    return {
        "source_page_title": page_title,
        "source_page_url": wikipedia_page_url(page_title),
        "source_page_id": None,
        "source_page_revision_id": None,
        "source_page_revision_time": "",
    }


def wikipedia_page_bound_asset_row(file: Any, event_id: str, modality: str, page_context: dict[str, Any]) -> dict[str, Any]:
    row = as_clean_dict(wikimedia_file_to_asset(file, event_id, modality))
    extra = dict(row.get("extra") or {})
    extra.update(page_context)
    extra.update(
        {
            "collection_channel": "wikipedia_page_bound",
            "page_bound": True,
            "file_title": getattr(file, "title", ""),
            "caption": getattr(file, "object_name", "") or getattr(file, "title", ""),
            "section_anchor": "",
            "asset_revision_time": getattr(file, "date_time", ""),
            "proposed_role": proposed_wikimedia_role(file, event_id, modality),
            "temporal_status": temporal_status_for_wikimedia_file(file, event_id, modality),
            "active_bench": True,
            "active_status": "active",
            "selection_reason": "selected from event Wikipedia page embedded images after MIME/decorative/relevance filters",
        }
    )
    row["extra"] = extra
    return annotate_external_asset_metadata(row)


def backfill_wikipedia_page_bound_metadata(row: dict[str, Any]) -> dict[str, Any]:
    if not is_wikimedia_asset(row):
        return row
    event_id = str(row.get("event_id", ""))
    page_title = WIKIPEDIA_EVENT_PAGES.get(event_id)
    clean = dict(row)
    extra = dict(clean.get("extra") or {})
    is_page_bound = bool(extra.get("page_bound") or extra.get("source_page_title") or extra.get("source_page_url") or extra.get("collection_channel") == "wikipedia_page_bound")
    if not is_page_bound:
        extra.setdefault("collection_channel", "commons_fallback_search")
        extra.setdefault("page_bound", False)
        extra.setdefault("active_bench", False)
        if extra.get("active_bench") is True:
            extra.setdefault("active_status", "active")
            extra.setdefault("selection_reason", "selected by explicit active_bench=true override")
        else:
            extra.setdefault("active_status", "candidate")
            extra.setdefault("selection_reason", "kept in candidate inventory; no Wikipedia source-page binding was recorded")
        clean["extra"] = extra
        return annotate_external_asset_metadata(clean)
    if not page_title:
        clean["extra"] = extra
        return annotate_external_asset_metadata(clean)
    title = str(extra.get("file_title") or extra.get("title") or "")
    file_like = SimpleNamespace(
        title=title,
        object_name=title.replace("File:", ""),
        credit=extra.get("credit", ""),
        artist=extra.get("artist", ""),
        date_time=clean.get("observed_time", ""),
    )
    extra.setdefault("collection_channel", "wikipedia_page_bound")
    extra.setdefault("page_bound", True)
    extra.setdefault("source_page_title", page_title)
    extra.setdefault("source_page_url", wikipedia_page_url(page_title))
    extra.setdefault("source_page_id", None)
    extra.setdefault("source_page_revision_id", None)
    extra.setdefault("source_page_revision_time", "")
    extra.setdefault("file_title", title)
    extra.setdefault("caption", title.replace("File:", ""))
    extra.setdefault("section_anchor", "")
    extra.setdefault("asset_revision_time", clean.get("observed_time", ""))
    extra.setdefault("proposed_role", proposed_wikimedia_role(file_like, event_id, str(clean.get("modality", ""))))
    extra.setdefault("temporal_status", temporal_status_for_wikimedia_file(file_like, event_id, str(clean.get("modality", ""))))
    extra.setdefault("active_bench", True)
    extra.setdefault("active_status", "active")
    extra.setdefault("selection_reason", "selected from event Wikipedia page embedded images after MIME/decorative/relevance filters")
    clean["extra"] = extra
    return annotate_external_asset_metadata(clean)


def annotate_external_asset_metadata(row: dict[str, Any]) -> dict[str, Any]:
    clean = dict(row)
    extra = dict(clean.get("extra") or {})
    asset_source = str(clean.get("asset_source", ""))
    modality = str(clean.get("modality", ""))
    if asset_source == "ACLED":
        extra.setdefault("collection_channel", "acled_api")
        extra.setdefault("record_type", "curated_conflict_event")
        extra.setdefault("curation_level", "human_curated")
        extra.setdefault("source_temporal_coverage", "event_window")
        extra.setdefault("active_policy", "optional_enrichment")
        extra.setdefault("active_bench", True)
        extra.setdefault("active_status", "active")
    elif asset_source == "GDELT_DOC":
        extra.setdefault("collection_channel", "gdelt_doc_search")
        extra.setdefault("record_type", "news_pointer")
        extra.setdefault("curation_level", "machine_indexed_news_pointer")
        extra.setdefault("source_temporal_coverage", str(extra.get("temporal_relation") or "unknown"))
        extra.setdefault("active_policy", "pointer_enrichment")
        extra.setdefault("active_bench", True)
        extra.setdefault("active_status", "active")
    elif asset_source == "OFFICIAL_DOC":
        extra.setdefault("collection_channel", "official_registry")
        extra.setdefault("record_type", "official_document")
        extra.setdefault("curation_level", "official")
        extra.setdefault("source_temporal_coverage", "unknown")
        if extra.get("active_policy") == "primary":
            extra["active_policy"] = "primary_official_evidence"
        else:
            extra.setdefault("active_policy", "primary_official_evidence")
        extra.setdefault("active_bench", True)
        extra.setdefault("active_status", "active")
    elif asset_source == "Wikimedia Commons":
        page_bound = bool(extra.get("page_bound") or extra.get("collection_channel") == "wikipedia_page_bound")
        if page_bound:
            extra.setdefault("collection_channel", "wikipedia_page_bound")
            extra.setdefault("record_type", "wiki_page_asset")
            if extra.get("active_policy") == "primary":
                extra["active_policy"] = "primary_image_evidence"
            else:
                extra.setdefault("active_policy", "primary_image_evidence")
        else:
            extra.setdefault("collection_channel", "commons_fallback_search")
            extra.setdefault("record_type", "commons_media_candidate")
            extra.setdefault("active_policy", "fallback_candidate")
            extra.setdefault("active_bench", False)
            extra.setdefault("active_status", "candidate")
        extra.setdefault("curation_level", "community_curated")
        extra.setdefault("source_temporal_coverage", "page_context" if page_bound else "search_context")
    else:
        extra.setdefault("collection_channel", "external_asset_import")
        extra.setdefault("record_type", "external_asset")
        extra.setdefault("curation_level", "unknown")
        extra.setdefault("source_temporal_coverage", "unknown")
        extra.setdefault("active_policy", "active_unless_flagged")
        extra.setdefault("active_bench", True)
        extra.setdefault("active_status", "active")
    if modality == "structured_event" and "record_type" not in extra:
        extra["record_type"] = "structured_event_candidate"
    active_status = str(extra.get("active_status", "active")).lower()
    if extra.get("active_bench") is False and active_status == "active":
        extra["active_status"] = "candidate"
    if extra.get("active_bench") is True and active_status == "candidate":
        extra["active_status"] = "active"
    clean["extra"] = extra
    return clean


def is_active_benchmark_asset(row: dict[str, Any]) -> bool:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    if extra.get("active_bench") is False:
        return False
    active_status = str(extra.get("active_status", "active")).lower()
    return active_status not in INACTIVE_ASSET_STATUSES


def asset_selection_decision(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    active = is_active_benchmark_asset(row)
    active_status = str(extra.get("active_status", "active" if active else "candidate"))
    reason = str(extra.get("selection_reason") or "")
    if not active:
        if extra.get("active_bench") is False:
            reason = "excluded from benchmark because active_bench=false"
        elif active_status.lower() in INACTIVE_ASSET_STATUSES:
            reason = f"excluded from benchmark because active_status={active_status}"
        elif not reason:
            reason = "excluded from benchmark by active asset policy"
    elif not reason:
        reason = "selected for benchmark assets"
    return {
        "asset_id": row.get("asset_id", ""),
        "event_id": row.get("event_id", ""),
        "asset_source": row.get("asset_source", ""),
        "modality": row.get("modality", ""),
        "collection_channel": extra.get("collection_channel", ""),
        "record_type": extra.get("record_type", ""),
        "active_bench": extra.get("active_bench", True),
        "active_status": active_status,
        "selected": active,
        "selection_reason": reason,
    }


def proposed_wikimedia_role(file: Any, event_id: str, modality: str) -> str:
    text = wikimedia_file_text(file)
    if modality == "map_pointer" or is_map_like_wikimedia_file(file):
        return "map_like"
    if event_id == "scs":
        return "background"
    if event_id == "jcpoa":
        return "official_context"
    if event_id == "hongkong" and any(term in text for term in ("protest", "demonstration", "extradition")):
        return "protest_context"
    if event_id == "iraq" and any(term in text for term in ("bush", "blair", "chirac", "berlusconi")):
        return "official_context"
    if event_id == "kosovo" and any(term in text for term in ("biden", "sejdiu")):
        return "official_context"
    return "substantive_event_image"


def temporal_status_for_wikimedia_file(file: Any, event_id: str, modality: str) -> str:
    text = wikimedia_file_text(file)
    event_date = date.fromisoformat(EVENTS[event_id].publish_time[:10])
    event_year = str(event_date.year)
    if modality == "map_pointer" and any(term in text for term in ("recognizing", "recognition", "countries", "phase", "control")):
        return "dynamic_updated_map"
    if event_year in text:
        return "near_event_window"
    raw_time = str(getattr(file, "date_time", "") or "")
    try:
        asset_date = date.fromisoformat(raw_time[:10])
    except ValueError:
        return "unknown"
    delta = (asset_date - event_date).days
    if delta == 0:
        return "contemporaneous"
    if -14 <= delta <= 45:
        return "near_event_window"
    if delta > 45:
        return "later_context"
    return "historical_background"


def wikimedia_file_text(file: Any) -> str:
    return " ".join(
        str(value or "")
        for value in (
            getattr(file, "title", ""),
            getattr(file, "object_name", ""),
            getattr(file, "credit", ""),
            getattr(file, "artist", ""),
        )
    ).lower()


def wikimedia_event_relevance_score(file: Any, event_id: str) -> int:
    text = wikimedia_file_text(file)
    if any(term in text for term in GENERIC_IMAGE_TERMS):
        return -10
    terms = EVENT_IMAGE_TERMS.get(event_id, ())
    score = sum(1 for term in terms if term in text)
    if is_map_like_wikimedia_file(file):
        score += 1
    return score


def is_map_like_wikimedia_file(file: Any) -> bool:
    text = wikimedia_file_text(file)
    return any(term in text for term in MAP_TERMS)


def rank_wikimedia_files(files: Iterable[Any], event_id: str) -> list[Any]:
    candidates = [
        file
        for file in files
        if is_wikimedia_image_mime(getattr(file, "mime", None)) and wikimedia_event_relevance_score(file, event_id) > 0
    ]
    return sorted(candidates, key=lambda file: (-wikimedia_event_relevance_score(file, event_id), getattr(file, "title", "")))


def fetch_acled_for_event(client: ACLEDClient, event_id: str, limit: int = 50, window_days: int = 0, max_pages: int = 20) -> list[dict[str, Any]]:
    country = EVENT_COUNTRY_HINTS.get(event_id)
    if not country:
        return []
    start_date, end_date = event_window(event_id, window_days)
    rows = client.events_for_window(
        country=country,
        start_date=start_date,
        end_date=end_date,
        fields=ACLED_FIELDS,
        limit=limit,
        paginate=True,
        max_pages=max_pages,
    )
    return [as_clean_dict(acled_row_to_asset(row, event_id)) for row in rows]


def dedupe_assets(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        asset_id = str(row.get("asset_id", ""))
        if not asset_id or asset_id in by_id:
            continue
        by_id[asset_id] = row
    return [by_id[asset_id] for asset_id in sorted(by_id)]


def is_wikimedia_asset(row: dict[str, Any]) -> bool:
    return row.get("asset_source") == "Wikimedia Commons"


def is_wikimedia_image_mime(mime: str | None) -> bool:
    return bool(mime and mime.startswith("image/"))


def wikimedia_mime(row: dict[str, Any]) -> str:
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    return str(extra.get("mime") or "")


def filter_external_asset_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not is_wikimedia_asset(row):
        return row, None
    mime = wikimedia_mime(row)
    if mime and not is_wikimedia_image_mime(mime):
        return None, {
            "event_id": row.get("event_id", ""),
            "source": "wikimedia",
            "warning": f"Filtered non-image Wikimedia asset {row.get('asset_id', '')} with mime={mime}.",
        }
    return row, None


def image_extension(mime: str, url: str) -> str:
    suffix = Path(unquote(urlsplit(url).path)).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".tif", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    if mime in IMAGE_EXTENSIONS:
        return IMAGE_EXTENSIONS[mime]
    return ".img"


def detected_image_extension(path: Path) -> str:
    try:
        header = path.read_bytes()[:64]
    except OSError:
        return ""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return ".gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    stripped = header.lstrip().lower()
    if stripped.startswith(b"<svg") or stripped.startswith(b"<?xml"):
        return ".svg"
    return ""


def download_url(url: str, output_path: Path, timeout: int = 60, max_retries: int = 2, retry_backoff_seconds: float = 30.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    attempts = max(0, max_retries) + 1
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                output_path.write_bytes(response.read())
            return
        except HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        except URLError as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        time.sleep(retry_backoff_seconds * (2**attempt))


def materialize_wikimedia_image(
    row: dict[str, Any],
    image_dir: Path,
    timeout: int = 60,
    download_retries: int = 2,
    download_retry_backoff_seconds: float = 30.0,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not is_wikimedia_asset(row):
        return row, None
    mime = wikimedia_mime(row)
    if not is_wikimedia_image_mime(mime):
        return row, None
    extra = dict(row.get("extra") or {})
    original_url = str(extra.get("original_url") or row.get("url_or_pointer") or "")
    if not original_url.startswith(("http://", "https://")):
        return row, None
    remote_url = str(extra.get("thumb_url") or original_url)
    extension = image_extension(mime, remote_url)
    local_path = image_dir / str(row.get("event_id", "unknown")) / f"{row['asset_id']}{extension}"
    legacy_extension = IMAGE_EXTENSIONS.get(mime, "")
    legacy_path = image_dir / str(row.get("event_id", "unknown")) / f"{row['asset_id']}{legacy_extension}" if legacy_extension else local_path
    if not local_path.exists() and legacy_path != local_path and legacy_path.exists():
        local_path = legacy_path
    try:
        if not local_path.exists():
            download_url(
                remote_url,
                local_path,
                timeout=timeout,
                max_retries=download_retries,
                retry_backoff_seconds=download_retry_backoff_seconds,
            )
        detected_extension = detected_image_extension(local_path)
        if detected_extension and detected_extension != local_path.suffix.lower():
            detected_path = local_path.with_suffix(detected_extension)
            if not detected_path.exists():
                local_path.replace(detected_path)
            local_path = detected_path
        clean = dict(row)
        extra["original_url"] = original_url
        extra["download_url"] = remote_url
        extra["local_image_path"] = relative_to_project(local_path)
        if detected_extension:
            extra["detected_file_extension"] = detected_extension
        extra["downloaded"] = True
        clean["extra"] = extra
        clean["url_or_pointer"] = relative_to_project(local_path)
        clean["perceptual_hash"] = sha256_file(local_path)
        clean["embedding_id"] = f"emb_{row['asset_id']}_{stable_hash(extra['local_image_path'])}"
        return clean, None
    except RuntimeError as exc:
        return row, {
            "event_id": row.get("event_id", ""),
            "source": "wikimedia",
            "warning": str(exc),
        }


def wikimedia_image_manifest_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if not is_wikimedia_asset(row):
        return None
    extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
    local_image_path = str(extra.get("local_image_path") or "")
    if not local_image_path:
        return None
    return {
        "asset_id": row.get("asset_id", ""),
        "event_id": row.get("event_id", ""),
        "modality": row.get("modality", ""),
        "asset_source": row.get("asset_source", ""),
        "source_layer": row.get("source_layer", ""),
        "evidence_role": row.get("evidence_role", ""),
        "local_image_path": local_image_path,
        "original_url": extra.get("original_url", ""),
        "download_url": extra.get("download_url", ""),
        "file_title": extra.get("file_title") or extra.get("title", ""),
        "caption": extra.get("caption") or row.get("caption_or_transcript", ""),
        "source_page_title": extra.get("source_page_title", ""),
        "source_page_url": extra.get("source_page_url", ""),
        "source_page_revision_id": extra.get("source_page_revision_id"),
        "source_page_revision_time": extra.get("source_page_revision_time", ""),
        "section_anchor": extra.get("section_anchor", ""),
        "license_or_terms": row.get("license_or_terms", ""),
        "license_url": extra.get("license_url", ""),
        "retrieval_time": row.get("observed_time", ""),
        "publish_time": row.get("publish_time", ""),
        "asset_revision_time": extra.get("asset_revision_time", ""),
        "mime": extra.get("mime", ""),
        "detected_file_extension": extra.get("detected_file_extension", ""),
        "width": extra.get("width"),
        "height": extra.get("height"),
        "collection_channel": extra.get("collection_channel", ""),
        "page_bound": extra.get("page_bound"),
        "proposed_role": extra.get("proposed_role", ""),
        "temporal_status": extra.get("temporal_status", ""),
        "active_bench": extra.get("active_bench"),
        "active_status": extra.get("active_status", ""),
        "selection_reason": extra.get("selection_reason", ""),
        "perceptual_hash": row.get("perceptual_hash", ""),
    }


def write_wikimedia_image_manifest(rows: list[dict[str, Any]], image_dir: Path) -> Path:
    manifest_rows = [manifest for row in rows if (manifest := wikimedia_image_manifest_row(row))]
    manifest_path = image_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    return manifest_path


def collect_existing_external_assets(
    input_dir: Path,
    merged_output: Path,
    event_ids: Iterable[str] | None = None,
    prior_errors: list[dict[str, Any]] | None = None,
    prior_warnings: list[dict[str, Any]] | None = None,
    acled_limit_hint: int | None = None,
    acled_window_days_hint: int = 0,
    download_wikimedia_images: bool = False,
    image_dir: Path = DATA_DIR / "0_external" / "event_images",
    download_timeout: int = 60,
    download_retries: int = 2,
    download_retry_backoff_seconds: float = 30.0,
    candidate_inventory_output: Path | None = None,
    selection_decisions_output: Path | None = None,
) -> dict[str, Any]:
    wanted = set(normalize_event_ids(event_ids))
    candidate_rows: list[dict[str, Any]] = []
    input_files = []
    warnings: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.jsonl")):
        if path.resolve() == merged_output.resolve():
            continue
        file_rows = []
        for row in read_jsonl(path):
            event_id = str(row.get("event_id", ""))
            if event_id not in wanted:
                continue
            row, warning = filter_external_asset_row(row)
            if warning:
                warnings.append(warning)
            if row is None:
                continue
            row = backfill_wikipedia_page_bound_metadata(row)
            row = annotate_external_asset_metadata(row)
            asset = dataclass_from_dict(EvidenceAsset, row)
            clean = as_clean_dict(asset)
            clean_extra = dict(clean.get("extra", {}))
            clean_extra.setdefault("external_input_file", relative_to_project(path))
            clean["extra"] = clean_extra
            file_rows.append(clean)
        if file_rows:
            input_files.append(relative_to_project(path))
            candidate_rows.extend(file_rows)

    candidates = dedupe_assets(candidate_rows)
    active_candidates = [row for row in candidates if is_active_benchmark_asset(row)]
    merged_rows: list[dict[str, Any]] = []
    for row in active_candidates:
        clean = row
        if download_wikimedia_images:
            clean, warning = materialize_wikimedia_image(
                clean,
                image_dir,
                timeout=download_timeout,
                download_retries=download_retries,
                download_retry_backoff_seconds=download_retry_backoff_seconds,
            )
            if warning:
                warnings.append(warning)
        merged_rows.append(clean)
    inventory_path = candidate_inventory_output or (merged_output.parent / "candidate_inventory.jsonl")
    decisions_path = selection_decisions_output or (merged_output.parent / "selection_decisions.jsonl")
    write_jsonl(inventory_path, candidates)
    write_jsonl(decisions_path, [asset_selection_decision(row) for row in candidates])
    write_jsonl(merged_output, merged_rows)
    summary = summarize_assets(merged_rows, merged_output, input_files)
    summary["candidate_inventory"] = relative_to_project(inventory_path)
    summary["selection_decisions"] = relative_to_project(decisions_path)
    summary["counts"]["candidate_assets"] = len(candidates)
    summary["counts"]["active_assets"] = len(merged_rows)
    summary["errors"] = dedupe_issue_rows(prior_errors or [])
    carried_warnings = [row for row in (prior_warnings or []) if row.get("source") not in {"acled", "wikimedia"}]
    summary["warnings"] = dedupe_issue_rows(
        [
            *carried_warnings,
            *warnings,
            *infer_acled_raw_warnings(input_dir, event_ids, acled_limit_hint, acled_window_days_hint),
        ]
    )
    if download_wikimedia_images:
        summary["image_manifest"] = relative_to_project(write_wikimedia_image_manifest(merged_rows, image_dir))
    write_json(merged_output.with_suffix(".summary.json"), summary)
    return summary


def summarize_assets(rows: list[dict[str, Any]], merged_output: Path, input_files: list[str] | None = None) -> dict[str, Any]:
    by_event = Counter(str(row.get("event_id", "")) for row in rows)
    by_source = Counter(str(row.get("asset_source", "")) for row in rows)
    by_modality = Counter(str(row.get("modality", "")) for row in rows)
    return {
        "merged_output": relative_to_project(merged_output),
        "input_files": input_files or [],
        "counts": {
            "merged_assets": len(rows),
            "events": len(by_event),
        },
        "by_event": dict(sorted(by_event.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_modality": dict(sorted(by_modality.items())),
    }


def run_external_asset_collection(
    output_dir: Path = DATA_DIR / "0_external" / "external_asset_raw",
    merged_output: Path = DATA_DIR / "0_external" / "external_assets.jsonl",
    event_ids: Iterable[str] | None = None,
    image_limit: int = 3,
    map_limit: int = 1,
    acled_limit: int = 50,
    acled_window_days: int = 0,
    acled_max_pages: int = 20,
    wikimedia_delay_seconds: float = 1.0,
    wikimedia_retries: int = 2,
    wikimedia_retry_backoff_seconds: float = 30.0,
    download_wikimedia_images: bool = False,
    image_dir: Path = DATA_DIR / "0_external" / "event_images",
    download_timeout: int = 60,
    download_retries: int = 2,
    download_retry_backoff_seconds: float = 30.0,
    skip_wikimedia: bool = False,
    skip_acled: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wikimedia = WikimediaCommonsClient(
        request_delay_seconds=wikimedia_delay_seconds,
        max_retries=wikimedia_retries,
        retry_backoff_seconds=wikimedia_retry_backoff_seconds,
    )
    acled = ACLEDClient()
    errors = []
    collection_warnings = []

    for event_id in normalize_event_ids(event_ids):
        if not skip_wikimedia:
            try:
                progress(f"starting Wikimedia collection for {event_id}")
                rows = fetch_wikimedia_for_event(wikimedia, event_id, image_limit=image_limit, map_limit=map_limit)
                write_jsonl(output_dir / f"wikimedia_{event_id}.jsonl", rows)
                progress(f"wrote {len(rows)} Wikimedia rows for {event_id}")
            except Exception as exc:
                if not continue_on_error:
                    raise
                errors.append({"event_id": event_id, "source": "wikimedia", "error": str(exc)})
        if not skip_acled:
            try:
                progress(f"starting ACLED collection for {event_id}")
                country = EVENT_COUNTRY_HINTS.get(event_id)
                start_date, end_date = event_window(event_id, acled_window_days)
                rows = fetch_acled_for_event(acled, event_id, limit=acled_limit, window_days=acled_window_days, max_pages=acled_max_pages)
                write_jsonl(output_dir / f"acled_{event_id}.jsonl", rows)
                collection_warnings.extend(acled_result_warnings(event_id, country, start_date, end_date, len(rows), acled_limit))
                progress(f"wrote {len(rows)} ACLED rows for {event_id}")
            except Exception as exc:
                if not continue_on_error:
                    raise
                errors.append({"event_id": event_id, "source": "acled", "error": str(exc)})

    progress("merging existing external assets")
    return collect_existing_external_assets(
        output_dir,
        merged_output,
        event_ids=None,
        prior_errors=errors,
        prior_warnings=collection_warnings,
        acled_limit_hint=acled_limit,
        acled_window_days_hint=acled_window_days,
        download_wikimedia_images=download_wikimedia_images,
        image_dir=image_dir,
        download_timeout=download_timeout,
        download_retries=download_retries,
        download_retry_backoff_seconds=download_retry_backoff_seconds,
    )
