"""Deterministic parsing for manually materialized official documents."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, sha256_text, stable_hash, write_json, write_jsonl


COPIED_STATUSES = {"copied", "copied_alternate_extension"}


class ParserDependencyError(RuntimeError):
    """Raised when an optional parser dependency is unavailable."""


def page_spans_for_text(text: str) -> list[dict[str, int]]:
    """Infer page spans from pdfminer form-feed separators; plain text is one page."""
    if not text:
        return []
    spans: list[dict[str, int]] = []
    start = 0
    page_number = 1
    for idx, ch in enumerate(text):
        if ch != "\f":
            continue
        spans.append({"page_number": page_number, "char_start": start, "char_end": idx + 1})
        start = idx + 1
        page_number += 1
    if start < len(text):
        spans.append({"page_number": page_number, "char_start": start, "char_end": len(text)})
    return spans or [{"page_number": 1, "char_start": 0, "char_end": len(text)}]


def page_range_for_span(page_spans: list[dict[str, int]], char_start: int, char_end: int) -> tuple[int | None, int | None]:
    """Map a character span to inclusive page_start/page_end values."""
    if not page_spans:
        return None, None
    touched = [
        span["page_number"]
        for span in page_spans
        if span["char_start"] < char_end and span["char_end"] > char_start
    ]
    if touched:
        return min(touched), max(touched)
    for span in page_spans:
        if char_start < span["char_end"]:
            return span["page_number"], span["page_number"]
    return page_spans[-1]["page_number"], page_spans[-1]["page_number"]


def split_passages(text: str, max_chars: int = 1800, overlap_chars: int = 150) -> list[dict[str, Any]]:
    """Split text into overlapping character passages while preserving offsets."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    passages: list[dict[str, Any]] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            split_at = text.rfind("\n\n", start + max_chars // 3, end)
            if split_at == -1:
                split_at = text.rfind("\n", start + max_chars // 3, end)
            if split_at == -1:
                split_at = text.rfind(" ", start + max_chars // 2, end)
            if split_at > start:
                end = split_at
        if end <= start:
            end = min(n, start + max_chars)

        passages.append(
            {
                "passage_index": len(passages),
                "char_start": start,
                "char_end": end,
                "text": text[start:end],
            }
        )
        if end >= n:
            break
        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start
    return passages


def parse_text_file(path: Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, {"parser": "manual_text", "page_count": 1, "page_spans": page_spans_for_text(text), "parser_warning": ""}


def parse_pdf_file(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.pdfpage import PDFPage
    except ModuleNotFoundError as exc:
        raise ParserDependencyError("pdfminer.six is required for PDF parsing") from exc

    text = extract_text(path.as_posix()) or ""
    page_spans = page_spans_for_text(text)
    try:
        with path.open("rb") as f:
            page_count = sum(1 for _ in PDFPage.get_pages(f))
    except Exception:
        page_count = len(page_spans) or None
    return text, {"parser": "pdfminer.six", "page_count": page_count, "page_spans": page_spans, "parser_warning": ""}


def parse_one_manifest_row(row: dict[str, Any], min_ok_chars: int) -> dict[str, Any]:
    local_path = Path(row.get("local_path") or "")
    base = {
        "document_id": row.get("document_id") or stable_hash(row.get("source_filename", "")),
        "event_id": row.get("event_id", ""),
        "language": row.get("language", ""),
        "source_filename": row.get("source_filename", ""),
        "local_path": row.get("local_path", ""),
        "extra": {
            "expected_source_filename": row.get("expected_source_filename", row.get("source_filename", "")),
            "state_source_filename": row.get("state_source_filename", ""),
            "actual_source_filename": row.get("actual_source_filename", ""),
            "source_resolution": row.get("source_resolution", ""),
            "source_url": row.get("source_url", ""),
            "output_kind": row.get("output_kind", ""),
            "materialization_method": row.get("materialization_method", ""),
            "organize_status": row.get("organize_status", ""),
            "manifest_sha256": row.get("sha256", ""),
        },
    }
    try:
        if not local_path.exists():
            raise FileNotFoundError(local_path.as_posix())
        if local_path.suffix.lower() == ".pdf":
            text, metadata = parse_pdf_file(local_path)
        else:
            text, metadata = parse_text_file(local_path)
        char_count = len(text)
        parse_quality = "ok" if char_count >= min_ok_chars else "low_text"
        text_hash = sha256_text(text)
        extra = {
            **base["extra"],
            "page_spans": metadata.pop("page_spans", []),
            "text_sha256": text_hash,
        }
        parser_warning = metadata.pop("parser_warning", "")
        if parser_warning:
            extra["parser_warning"] = parser_warning
        return {
            **base,
            **metadata,
            "extra": extra,
            "text": text,
            "sha256": sha256_file(local_path),
            "char_count": char_count,
            "parse_quality": parse_quality,
        }
    except ParserDependencyError as exc:
        extra = {
            **base["extra"],
            "page_spans": [],
            "text_sha256": sha256_text(""),
            "parser_warning": "pdf_parser_missing",
            "parse_error": str(exc),
        }
        return {
            **base,
            "parser": "pdfminer.six",
            "page_count": None,
            "extra": extra,
            "text": "",
            "sha256": sha256_file(local_path) if local_path.exists() else "",
            "char_count": 0,
            "parse_quality": "failed",
        }
    except Exception as exc:
        extra = {
            **base["extra"],
            "page_spans": [],
            "text_sha256": sha256_text(""),
            "parse_error": f"{type(exc).__name__}: {exc}",
        }
        return {
            **base,
            "parser": "unknown",
            "page_count": None,
            "extra": extra,
            "text": "",
            "sha256": sha256_file(local_path) if local_path.exists() else "",
            "char_count": 0,
            "parse_quality": "failed",
        }


def copied_manifest_rows(manifest_path: str | Path) -> tuple[list[dict[str, Any]], int]:
    rows = list(read_jsonl(manifest_path))
    copied = [row for row in rows if row.get("organize_status") in COPIED_STATUSES and row.get("local_path")]
    return copied, len(rows) - len(copied)


def parse_materialized_documents(
    manifest_path: str | Path,
    output_dir: str | Path,
    max_passage_chars: int = 1800,
    overlap_chars: int = 150,
    min_ok_chars: int = 100,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    output = Path(output_dir)
    rows, skipped = copied_manifest_rows(manifest)

    doc_rows = [parse_one_manifest_row(row, min_ok_chars=min_ok_chars) for row in rows]
    passage_rows: list[dict[str, Any]] = []
    for doc in doc_rows:
        if doc["parse_quality"] == "failed" or not doc["text"]:
            continue
        passages = split_passages(doc["text"], max_chars=max_passage_chars, overlap_chars=overlap_chars)
        for passage in passages:
            language = doc["language"] or "unknown"
            page_spans = doc.get("extra", {}).get("page_spans", [])
            page_start, page_end = page_range_for_span(page_spans, passage["char_start"], passage["char_end"])
            passage_id = f"passage_{doc['document_id']}_{language}_{passage['passage_index']:04d}"
            passage_rows.append(
                {
                    "passage_id": passage_id,
                    "document_id": doc["document_id"],
                    "event_id": doc["event_id"],
                    "language": doc["language"],
                    "source_filename": doc["source_filename"],
                    "page_start": page_start,
                    "page_end": page_end,
                    **passage,
                    "extra": {
                        "text_sha256": sha256_text(passage["text"]),
                    },
                }
            )

    text_output_rows = []
    for doc in doc_rows:
        text_output_rows.append(doc)

    write_jsonl(output / "official_doc_text.jsonl", text_output_rows)
    write_jsonl(output / "passages.jsonl", passage_rows)

    char_counts = [row["char_count"] for row in doc_rows]
    shortest_documents = sorted(
        (
            {
                "document_id": row["document_id"],
                "event_id": row["event_id"],
                "language": row["language"],
                "source_filename": row["source_filename"],
                "char_count": row["char_count"],
                "parse_quality": row["parse_quality"],
            }
            for row in doc_rows
        ),
        key=lambda row: row["char_count"],
    )[:10]
    summary = {
        "manifest_path": manifest.as_posix(),
        "output_dir": output.as_posix(),
        "manifest_rows_total": len(list(read_jsonl(manifest))),
        "manifest_rows_skipped": skipped,
        "documents_total": len(doc_rows),
        "documents_ok": sum(1 for row in doc_rows if row["parse_quality"] == "ok"),
        "documents_low_text": sum(1 for row in doc_rows if row["parse_quality"] == "low_text"),
        "documents_failed": sum(1 for row in doc_rows if row["parse_quality"] == "failed"),
        "passages_total": len(passage_rows),
        "total_char_count": sum(char_counts),
        "min_char_count": min(char_counts) if char_counts else 0,
        "max_char_count": max(char_counts) if char_counts else 0,
        "shortest_documents": shortest_documents,
        "by_event": dict(sorted(Counter(row["event_id"] for row in doc_rows).items())),
        "by_language": dict(sorted(Counter(row["language"] for row in doc_rows).items())),
        "by_parser": dict(sorted(Counter(row["parser"] for row in doc_rows).items())),
        "by_parse_quality": dict(sorted(Counter(row["parse_quality"] for row in doc_rows).items())),
        "warnings": [
            {
                "document_id": row["document_id"],
                "source_filename": row["source_filename"],
                "parser_warning": row.get("extra", {}).get("parser_warning", ""),
                "parse_error": row.get("extra", {}).get("parse_error", ""),
            }
            for row in doc_rows
            if row.get("extra", {}).get("parser_warning") or row.get("extra", {}).get("parse_error")
        ],
    }
    write_json(output / "parse_summary.json", summary)
    return summary


def add_parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--manifest", type=Path, default=Path("data/0_external/official_doc_materialized/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/0_external/official_doc_parsed"))
    parser.add_argument("--max-passage-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=150)
    parser.add_argument("--min-ok-chars", type=int, default=100)
    return parser
