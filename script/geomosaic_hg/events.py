"""Event and filename parsing helpers for the offline GeoGround-MM core."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventInfo:
    event_id: str
    name: str
    subject: str
    publish_time: str
    geo_location: str
    map_eligible: bool


EVENTS: dict[str, EventInfo] = {
    "crimea": EventInfo("crimea", "Crimea Annexation", "Crimea", "2014-03-16T00:00:00Z", "Crimea, Ukraine", True),
    "iraq": EventInfo("iraq", "Iraq Invasion", "Iraq", "2003-03-20T00:00:00Z", "Iraq", False),
    "libya": EventInfo("libya", "Libya NATO Intervention", "Libya", "2011-03-19T00:00:00Z", "Libya", False),
    "kosovo": EventInfo("kosovo", "Kosovo Independence", "Kosovo", "2008-02-17T00:00:00Z", "Kosovo", True),
    "scs": EventInfo("scs", "South China Sea Arbitration", "South China Sea", "2016-07-12T00:00:00Z", "South China Sea", True),
    "jcpoa": EventInfo("jcpoa", "JCPOA Agreement", "Iran nuclear agreement", "2015-07-14T00:00:00Z", "Iran", False),
    "ukraine": EventInfo("ukraine", "Russia-Ukraine War", "Ukraine", "2022-02-24T00:00:00Z", "Ukraine", True),
    "hongkong": EventInfo("hongkong", "Hong Kong National Security Law", "Hong Kong", "2020-06-30T00:00:00Z", "Hong Kong", True),
}

VIEWPOINT_ALIASES = {
    "us": "us_anglo",
    "us-anglo": "us_anglo",
    "us_anglo": "us_anglo",
    "eu": "eu",
    "europe": "eu",
    "russia": "russia",
    "ru": "russia",
    "china": "china",
    "cn": "china",
    "un": "un",
    "wikipedia": "all",
    "wiki": "all",
    "all": "all",
    "news": "all",
}

DISPLAY_VIEWPOINT = {
    "us_anglo": "US-Anglo",
    "eu": "EU",
    "russia": "Russia",
    "china": "China",
    "un": "UN",
    "all": "All",
}

EVENT_ALIASES = {
    "csc": "scs",
    "southchinasea": "scs",
    "lybia": "libya",
    "hkNSL": "hongkong",
    "hknsl": "hongkong",
    "hong_kong": "hongkong",
}

NEWS_OUTLET_ALIASES = {
    "g": "Guardian",
    "guardian": "Guardian",
    "r": "Reuters",
    "reuters": "Reuters",
    "wsj": "WSJ",
}


@dataclass(frozen=True)
class ParsedRawText:
    path: Path
    event_id: str
    source_layer: str
    viewpoint_origin: str
    document_type: str
    institution_or_outlet: str
    source_key: str
    language: str


def normalize_event(value: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9_]+", "", value).lower()
    return EVENT_ALIASES.get(raw, raw)


def normalize_viewpoint(value: str | None) -> str:
    if value is None:
        return "all"
    raw = value.strip().lower().replace(" ", "_")
    raw = raw.replace("-", "_")
    raw = raw.replace("us_anglo", "us_anglo")
    if raw.startswith("news_"):
        return "all"
    if raw.startswith("news-"):
        return "all"
    return VIEWPOINT_ALIASES.get(raw, raw)


def display_viewpoint(value: str) -> str:
    return DISPLAY_VIEWPOINT.get(normalize_viewpoint(value), value)


def parse_raw_text_path(path: str | Path) -> ParsedRawText | None:
    p = Path(path)
    stem = p.stem

    m = re.match(r"^(?P<event>[^_]+)__(?P<vp>.+)__official$", stem)
    if m:
        event_id = normalize_event(m.group("event"))
        vp = normalize_viewpoint(m.group("vp"))
        return ParsedRawText(
            path=p,
            event_id=event_id,
            source_layer="official",
            viewpoint_origin=vp,
            document_type="official_statement",
            institution_or_outlet=display_viewpoint(vp),
            source_key=f"official-{vp}",
            language="en",
        )

    m = re.match(r"^(?P<event>[^_]+)___all__wiki$", stem)
    if m:
        event_id = normalize_event(m.group("event"))
        return ParsedRawText(
            path=p,
            event_id=event_id,
            source_layer="wiki",
            viewpoint_origin="all",
            document_type="wiki_entry",
            institution_or_outlet="Wikipedia",
            source_key="wiki",
            language="en",
        )

    m = re.match(r"^(?P<event>[^_]+)___all__news_(?P<outlet>[a-zA-Z]+)(?:_(?P<idx>\d+))?$", stem)
    if m:
        event_id = normalize_event(m.group("event"))
        outlet = NEWS_OUTLET_ALIASES.get(m.group("outlet").lower(), m.group("outlet").title())
        idx = m.group("idx")
        return ParsedRawText(
            path=p,
            event_id=event_id,
            source_layer="news",
            viewpoint_origin="all",
            document_type="news_article",
            institution_or_outlet=outlet,
            source_key=f"news-{outlet.lower()}-{idx}" if idx else f"news-{outlet.lower()}",
            language="en",
        )

    return None


@dataclass(frozen=True)
class ParsedNewsPdf:
    path: Path
    event_id: str
    outlet: str
    article_idx: int


def parse_news_pdf_path(path: str | Path) -> ParsedNewsPdf | None:
    p = Path(path)
    stem = p.stem.lower()
    m = re.match(r"^(?P<outlet>wsj|reuters|guardian|g|r)_(?P<event>[a-z]+)(?P<idx>\d+)$", stem)
    if not m:
        return None
    event_id = normalize_event(m.group("event"))
    outlet = NEWS_OUTLET_ALIASES.get(m.group("outlet"), m.group("outlet").title())
    return ParsedNewsPdf(path=p, event_id=event_id, outlet=outlet, article_idx=int(m.group("idx")))


def score_source_layer_from_dir(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "news" in name:
        return "news"
    if "wiki" in name:
        return "wiki"
    return "official"
