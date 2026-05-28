"""Wikimedia Commons client for redistributable image and map metadata."""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from ..events import EVENTS
from ..io import sha256_text, stable_hash
from ..paths import relative_to_project
from ..schema import EvidenceAsset
from .http import HTTPClientError, get_json


COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


@dataclass(frozen=True)
class WikimediaFile:
    title: str
    page_id: int | None
    file_url: str
    description_url: str
    thumb_url: str
    mime: str
    sha1: str
    width: int | None
    height: int | None
    license_short_name: str
    license_url: str
    object_name: str
    artist: str
    credit: str
    date_time: str


class WikimediaCommonsClient:
    """Small wrapper around MediaWiki Action API for Commons files."""

    def __init__(
        self,
        api_url: str = COMMONS_API_URL,
        wikipedia_api_url: str = WIKIPEDIA_API_URL,
        timeout: int = 30,
        request_delay_seconds: float = 1.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 30.0,
    ) -> None:
        self.api_url = api_url
        self.wikipedia_api_url = wikipedia_api_url
        self.timeout = timeout
        self.request_delay_seconds = max(0.0, request_delay_seconds)
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._last_request_at = 0.0

    def _get_json(self, params: dict[str, Any], api_url: str | None = None) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            elapsed = time.time() - self._last_request_at
            if elapsed < self.request_delay_seconds:
                time.sleep(self.request_delay_seconds - elapsed)
            try:
                response = get_json(api_url or self.api_url, params, timeout=self.timeout)
                self._last_request_at = time.time()
                return response.data
            except HTTPClientError as exc:
                self._last_request_at = time.time()
                if "HTTP 429" not in str(exc) or attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("unreachable Wikimedia retry state")

    def search_files(self, query: str, limit: int = 10) -> list[str]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": min(limit, 50),
            "prop": "info",
        }
        data = self._get_json(params)
        pages = data.get("query", {}).get("pages", {})
        return [page["title"] for page in pages.values() if page.get("title", "").startswith("File:")]

    def imageinfo(self, titles: list[str], thumb_width: int = 640, api_url: str | None = None) -> list[WikimediaFile]:
        if not titles:
            return []
        out: list[WikimediaFile] = []
        for chunk_start in range(0, len(titles), 10):
            chunk = titles[chunk_start : chunk_start + 10]
            params = {
                "action": "query",
                "format": "json",
                "titles": "|".join(chunk),
                "prop": "imageinfo",
                "iiprop": "url|sha1|mime|size|extmetadata|timestamp",
                "iiurlwidth": thumb_width,
            }
            data = self._get_json(params, api_url=api_url)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                meta = info.get("extmetadata") or {}

                def mv(key: str) -> str:
                    return strip_html(str((meta.get(key) or {}).get("value", "")))

                out.append(
                    WikimediaFile(
                        title=page.get("title", ""),
                        page_id=page.get("pageid"),
                        file_url=info.get("url", ""),
                        description_url=info.get("descriptionurl", ""),
                        thumb_url=info.get("thumburl", ""),
                        mime=info.get("mime", ""),
                        sha1=info.get("sha1", ""),
                        width=info.get("width"),
                        height=info.get("height"),
                        license_short_name=mv("LicenseShortName"),
                        license_url=mv("LicenseUrl"),
                        object_name=mv("ObjectName"),
                        artist=mv("Artist"),
                        credit=mv("Credit"),
                        date_time=mv("DateTime") or info.get("timestamp", ""),
                    )
                )
        return out

    def search_imageinfo(self, query: str, limit: int = 10, thumb_width: int = 640) -> list[WikimediaFile]:
        return self.imageinfo(self.search_files(query, limit), thumb_width=thumb_width)

    def page_file_titles(self, page_title: str, limit: int = 50) -> list[str]:
        titles: list[str] = []
        continue_params: dict[str, Any] = {}
        while len(titles) < limit:
            params: dict[str, Any] = {
                "action": "query",
                "format": "json",
                "redirects": 1,
                "titles": page_title,
                "prop": "images",
                "imlimit": min(50, limit - len(titles)),
            }
            params.update(continue_params)
            data = self._get_json(params, api_url=self.wikipedia_api_url)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for image in page.get("images") or []:
                    title = image.get("title", "")
                    if title.startswith("File:") and title not in titles:
                        titles.append(title)
                        if len(titles) >= limit:
                            break
                if len(titles) >= limit:
                    break
            if "continue" not in data or len(titles) >= limit:
                break
            continue_params = data["continue"]
        return titles

    def page_imageinfo(self, page_title: str, limit: int = 50, thumb_width: int = 640) -> list[WikimediaFile]:
        return self.imageinfo(self.page_file_titles(page_title, limit), thumb_width=thumb_width, api_url=self.wikipedia_api_url)

    def page_metadata(self, page_title: str) -> dict[str, Any]:
        params = {
            "action": "query",
            "format": "json",
            "redirects": 1,
            "titles": page_title,
            "prop": "info|revisions",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "rvlimit": 1,
        }
        data = self._get_json(params, api_url=self.wikipedia_api_url)
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        revisions = page.get("revisions") or [{}]
        title = page.get("title") or page_title
        return {
            "title": title,
            "url": page.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            "page_id": page.get("pageid"),
            "revision_id": page.get("lastrevid") or revisions[0].get("revid"),
            "revision_time": revisions[0].get("timestamp") or page.get("touched", ""),
        }


def wikimedia_file_to_asset(file: WikimediaFile, event_id: str, modality: str = "image_full") -> EvidenceAsset:
    info = EVENTS[event_id]
    title = file.object_name or file.title
    caption = f"{title}. Wikimedia Commons metadata: {file.license_short_name or 'license unavailable'}."
    license_terms = file.license_short_name or "wikimedia-commons-check-file-page"
    pointer = file.file_url or file.description_url
    return EvidenceAsset(
        asset_id=f"asset_wikimedia_{event_id}_{modality}_{stable_hash(file.title)}",
        event_id=event_id,
        modality=modality,
        asset_source="Wikimedia Commons",
        source_layer="wiki",
        viewpoint_origin="all",
        publish_time=info.publish_time,
        observed_time=file.date_time or info.publish_time,
        geo_location=info.geo_location,
        url_or_pointer=pointer,
        caption_or_transcript=caption,
        license_or_terms=license_terms,
        redistribution_flag=bool(file.file_url and file.license_short_name),
        perceptual_hash=file.sha1 or sha256_text(pointer),
        embedding_id=f"emb_asset_wikimedia_{event_id}_{modality}_{stable_hash(file.title)}",
        extracted_entities=[info.subject, title],
        extracted_claims=[],
        evidence_role="map_like" if modality == "map_pointer" else "complementary",
        extra={
            "title": file.title,
            "description_url": file.description_url,
            "thumb_url": file.thumb_url,
            "mime": file.mime,
            "width": file.width,
            "height": file.height,
            "license_url": file.license_url,
            "artist": file.artist,
            "credit": file.credit,
        },
    )
