"""GDELT DOC 2.0 client for news/article pointer metadata."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..events import EVENTS
from ..io import sha256_text, stable_hash
from ..schema import EvidenceAsset
from .http import HTTPClientError, get_json


GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GDELTDOCClient:
    """Small wrapper around the public GDELT DOC 2.0 full-text search API."""

    def __init__(
        self,
        api_url: str = GDELT_DOC_API_URL,
        timeout: int = 30,
        rate_limit_seconds: float = 5.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 10.0,
    ) -> None:
        self.api_url = api_url
        self.timeout = timeout
        self.rate_limit_seconds = max(0.0, rate_limit_seconds)
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._last_request_at = 0.0

    def _respect_rate_limit(self) -> None:
        if not self._last_request_at or self.rate_limit_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait_seconds = self.rate_limit_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _is_retryable_error(self, exc: HTTPClientError) -> bool:
        if exc.status in {408, 425, 429, 500, 502, 503, 504}:
            return True
        return exc.status is None and str(exc).startswith("Request failed")

    def _get_json_with_retry(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = get_json(self.api_url, params, timeout=self.timeout)
                self._last_request_at = time.monotonic()
                return response.data
            except HTTPClientError as exc:
                self._last_request_at = time.monotonic()
                if not self._is_retryable_error(exc) or attempt >= self.max_retries:
                    raise
                wait_seconds = self.retry_backoff_seconds * (2**attempt)
                time.sleep(wait_seconds)
                if wait_seconds >= self.rate_limit_seconds:
                    self._last_request_at = 0.0
        return {}

    def search_articles(
        self,
        query: str,
        max_records: int = 25,
        start_datetime: str | None = None,
        end_datetime: str | None = None,
        timespan: str | None = None,
        sort: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max(1, min(max_records, 250)),
            "startdatetime": start_datetime,
            "enddatetime": end_datetime,
            "timespan": timespan,
            "sort": sort,
        }
        data = self._get_json_with_retry(params)
        articles = data.get("articles", [])
        if not isinstance(articles, list):
            return []
        return [article for article in articles if isinstance(article, dict)]


def gdelt_seen_date_to_iso(value: str, fallback: str) -> str:
    raw = str(value or "")
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return fallback


def gdelt_doc_article_to_asset(
    article: dict[str, Any],
    event_id: str,
    query: str,
    temporal_relation: str = "unknown",
) -> EvidenceAsset:
    info = EVENTS[event_id]
    url = str(article.get("url") or article.get("url_mobile") or "")
    title = str(article.get("title") or url or "GDELT DOC article")
    seendate = gdelt_seen_date_to_iso(str(article.get("seendate", "")), info.publish_time)
    domain = str(article.get("domain") or "")
    language = str(article.get("language") or "")
    source_country = str(article.get("sourcecountry") or "")
    text = f"{title}. GDELT DOC article pointer from {domain or 'unknown domain'}."
    asset_id = f"asset_gdelt_doc_{event_id}_{stable_hash(url or title)}"
    return EvidenceAsset(
        asset_id=asset_id,
        event_id=event_id,
        modality="text",
        asset_source="GDELT_DOC",
        source_layer="news",
        viewpoint_origin=source_country or "news_aggregate",
        publish_time=seendate,
        observed_time=seendate,
        geo_location=info.geo_location,
        url_or_pointer=url,
        caption_or_transcript=text,
        license_or_terms="GDELT DOC pointer; original publisher terms apply",
        redistribution_flag=False,
        perceptual_hash=sha256_text("|".join([url, title, seendate])),
        embedding_id=f"emb_{asset_id}",
        extracted_entities=[info.subject, title, domain],
        extracted_claims=[],
        evidence_role="context",
        extra={
            "collection_channel": "gdelt_doc_search",
            "record_type": "news_pointer",
            "curation_level": "machine_indexed_news_pointer",
            "source_temporal_coverage": temporal_relation,
            "active_policy": "pointer_enrichment",
            "active_bench": True,
            "active_status": "active",
            "temporal_relation": temporal_relation,
            "query": query,
            "domain": domain,
            "language": language,
            "sourcecountry": source_country,
            "socialimage": article.get("socialimage", ""),
            "gdelt_doc": article,
        },
    )
