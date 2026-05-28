"""ACLED API client for structured conflict-event metadata."""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Any

from ..events import EVENTS
from ..io import sha256_text, stable_hash
from ..schema import EvidenceAsset
from .http import HTTPClientError, get_json, post_form_json


ACLED_API_URL = "https://acleddata.com/api/acled/read"
ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"


@dataclass(frozen=True)
class ACLEDCredentials:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "ACLEDCredentials":
        username = os.environ.get("ACLED_USERNAME") or os.environ.get("ACLED_EMAIL")
        password = os.environ.get("ACLED_PASSWORD") or os.environ.get("ACLED_PWD")
        if not username or not password:
            raise ValueError("Set ACLED_USERNAME or ACLED_EMAIL, and ACLED_PASSWORD or ACLED_PWD.")
        return cls(username=username, password=password)


class ACLEDClient:
    """OAuth-authenticated ACLED `/api/acled/read` client."""

    def __init__(
        self,
        credentials: ACLEDCredentials | None = None,
        api_url: str = ACLED_API_URL,
        token_url: str = ACLED_TOKEN_URL,
        timeout: int = 30,
    ) -> None:
        self.credentials = credentials
        self.api_url = api_url
        self.token_url = token_url
        self.timeout = timeout
        self._token: str | None = None
        self._refresh_token: str | None = None
        self._token_expires_at = 0.0

    def access_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        if self._refresh_token:
            try:
                return self._request_token(
                    {
                        "refresh_token": self._refresh_token,
                        "grant_type": "refresh_token",
                        "client_id": "acled",
                    }
                )
            except (HTTPClientError, RuntimeError):
                self._refresh_token = None
        credentials = self.credentials or ACLEDCredentials.from_env()
        return self._request_token(
            {
                "username": credentials.username,
                "password": credentials.password,
                "grant_type": "password",
                "client_id": "acled",
                "scope": "authenticated",
            }
        )

    def _request_token(self, data: dict[str, Any]) -> str:
        response = post_form_json(self.token_url, data, timeout=self.timeout).data
        token = response.get("access_token")
        if not token:
            raise RuntimeError(f"ACLED token response did not contain access_token: {response}")
        self._token = str(token)
        if response.get("refresh_token"):
            self._refresh_token = str(response["refresh_token"])
        self._token_expires_at = time.time() + int(response.get("expires_in", 86400))
        return self._token

    def read(self, params: dict[str, Any] | None = None, limit: int | None = None, page: int | None = None) -> list[dict[str, Any]]:
        query = {"_format": "json", **(params or {})}
        if limit is not None:
            query["limit"] = limit
        if page is not None:
            query["page"] = page
        headers = {"Authorization": f"Bearer {self.access_token()}", "Content-Type": "application/json"}
        try:
            response = get_json(self.api_url, query, headers=headers, timeout=self.timeout).data
        except HTTPClientError as exc:
            if exc.status != 401:
                raise
            self._token = None
            headers["Authorization"] = f"Bearer {self.access_token()}"
            response = get_json(self.api_url, query, headers=headers, timeout=self.timeout).data
        if int(response.get("status", 200)) >= 400:
            raise RuntimeError(f"ACLED API error: {response}")
        data = response.get("data", response)
        if isinstance(data, list):
            return data
        raise RuntimeError(f"Unexpected ACLED response payload: {response}")

    def read_paginated(self, params: dict[str, Any] | None = None, page_size: int = 5000, max_pages: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        last_page_count = 0
        for page in range(1, max(1, max_pages) + 1):
            page_rows = self.read(params, limit=page_size, page=page)
            rows.extend(page_rows)
            last_page_count = len(page_rows)
            if len(page_rows) < page_size:
                return rows
        if last_page_count >= page_size:
            warnings.warn(
                f"ACLED pagination reached max_pages={max_pages} with page_size={page_size}; data may be truncated.",
                RuntimeWarning,
                stacklevel=2,
            )
        return rows

    def events_for_window(
        self,
        country: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        event_types: list[str] | None = None,
        fields: list[str] | None = None,
        limit: int = 500,
        paginate: bool = False,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if country:
            params["country"] = country
        if start_date and end_date:
            params["event_date"] = f"{start_date}|{end_date}"
            params["event_date_where"] = "BETWEEN"
        if event_types:
            params["event_type"] = "|".join(event_types)
        if fields:
            params["fields"] = "|".join(fields)
        if paginate:
            return self.read_paginated(params, page_size=limit, max_pages=max_pages)
        rows = self.read(params, limit=limit)
        if limit is not None and limit > 0 and len(rows) >= limit:
            warnings.warn(
                f"ACLED result for {country or 'all countries'} ({start_date or '?'}~{end_date or '?'}) "
                f"hit limit={limit}; data may be truncated. Increase --acled-limit or add pagination.",
                RuntimeWarning,
                stacklevel=2,
            )
        return rows


def acled_row_to_asset(row: dict[str, Any], event_id: str) -> EvidenceAsset:
    info = EVENTS[event_id]
    acled_id = str(row.get("event_id_cnty") or row.get("event_id_no_cnty") or stable_hash(str(row)))
    event_date = str(row.get("event_date") or info.publish_time[:10])
    location = ", ".join(str(row.get(k, "")) for k in ("location", "admin1", "country") if row.get(k)) or info.geo_location
    event_type = str(row.get("event_type") or "ACLED event")
    fatalities = str(row.get("fatalities", ""))
    text = f"ACLED {event_type} event {acled_id} on {event_date} at {location}; fatalities={fatalities}."
    return EvidenceAsset(
        asset_id=f"asset_acled_{event_id}_{stable_hash(acled_id)}",
        event_id=event_id,
        modality="structured_event",
        asset_source="ACLED",
        source_layer="structured",
        viewpoint_origin="all",
        publish_time=f"{event_date}T00:00:00Z",
        observed_time=f"{event_date}T00:00:00Z",
        geo_location=location,
        url_or_pointer=f"acled://event/{acled_id}",
        caption_or_transcript=text,
        license_or_terms="ACLED API terms apply",
        redistribution_flag=False,
        perceptual_hash=sha256_text(str(row)),
        embedding_id=f"emb_asset_acled_{event_id}_{stable_hash(acled_id)}",
        extracted_entities=[str(row.get("actor1", "")), str(row.get("actor2", "")), location],
        extracted_claims=[],
        evidence_role="context",
        extra={"acled": row},
    )
