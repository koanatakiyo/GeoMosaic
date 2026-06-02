"""Tiny stdlib HTTP helpers with polite defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "GeoMosaic-HG/0.1 (research prototype; contact: local)"
JSON_ESCAPE_CHARS = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


@dataclass
class HTTPResponse:
    url: str
    status: int
    data: Any


class HTTPClientError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body_preview: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body_preview = body_preview


def _escape_invalid_json_backslashes(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not nxt or nxt not in JSON_ESCAPE_CHARS:
                out.append("\\\\")
                i += 1
                continue
            out.append(ch)
            out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_json_lenient(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_escape_invalid_json_backslashes(text))


def get_json(base_url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: int = 30) -> HTTPResponse:
    params = {k: v for k, v in (params or {}).items() if v is not None}
    query = urlencode(params, doseq=True)
    url = f"{base_url}?{query}" if query else base_url
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT, **(headers or {})})
    body = ""
    status: int | None = None
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8")
            return HTTPResponse(url=url, status=status, data=load_json_lenient(body))
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        preview = text[:500]
        raise HTTPClientError(f"HTTP {exc.code} for {url}: {preview}", status=exc.code, body_preview=preview) from exc
    except URLError as exc:
        raise HTTPClientError(f"Request failed for {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        preview = body[:500]
        raise HTTPClientError(f"Invalid JSON response from {url}: {exc}; response_preview={preview!r}", status=status, body_preview=preview) from exc


def post_form_json(base_url: str, data: dict[str, Any], headers: dict[str, str] | None = None, timeout: int = 30) -> HTTPResponse:
    encoded = urlencode(data).encode("utf-8")
    request = Request(
        base_url,
        data=encoded,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return HTTPResponse(url=base_url, status=response.status, data=json.loads(body))
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise HTTPClientError(f"HTTP {exc.code} for {base_url}: {text[:500]}", status=exc.code) from exc
    except URLError as exc:
        raise HTTPClientError(f"Request failed for {base_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPClientError(f"Invalid JSON response from {base_url}: {exc}") from exc
