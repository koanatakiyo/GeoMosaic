"""Deterministic QA checks for parsed official-document text/passages."""

from __future__ import annotations

import argparse
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, sha256_file, write_json
from .official_doc_parsing import page_range_for_span


DOC_REQUIRED_FIELDS = {
    "document_id",
    "event_id",
    "language",
    "source_filename",
    "local_path",
    "parser",
    "text",
    "char_count",
    "page_count",
    "parse_quality",
    "sha256",
    "extra",
}
DOC_FORBIDDEN_TOP_LEVEL = {
    "expected_source_filename",
    "page_spans",
    "text_sha256",
    "parser_warning",
    "parse_error",
    "source_sha256",
    "file_sha256",
}
DOC_REQUIRED_EXTRA_FIELDS = {"text_sha256", "page_spans", "expected_source_filename"}

PASSAGE_REQUIRED_FIELDS = {
    "passage_id",
    "document_id",
    "event_id",
    "language",
    "source_filename",
    "page_start",
    "page_end",
    "passage_index",
    "char_start",
    "char_end",
    "text",
    "extra",
}
PASSAGE_FORBIDDEN_TOP_LEVEL = {"text_sha256"}
PASSAGE_REQUIRED_EXTRA_FIELDS = {"text_sha256"}


def parse_source_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    parts = stem.split("__", 4)
    if len(parts) != 5:
        return {
            "event_id": "unknown",
            "viewpoint_origin": "unknown",
            "evidence_scope": "unknown",
            "language": "unknown",
            "document_id": stem,
        }
    event_id, viewpoint_origin, evidence_scope, language, document_id = parts
    return {
        "event_id": event_id,
        "viewpoint_origin": viewpoint_origin,
        "evidence_scope": evidence_scope,
        "language": language,
        "document_id": document_id,
    }


def nested_counter(rows: list[dict[str, Any]], outer_key: str, inner_key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get(outer_key, ""))][str(row.get(inner_key, ""))] += 1
    return {outer: dict(sorted(counter.items())) for outer, counter in sorted(counts.items())}


def high_replacement_ratio(text: str, threshold: float) -> float:
    if not text:
        return 0.0
    return text.count("\ufffd") / len(text)


def control_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    allowed = {"\n", "\r", "\t", "\f"}
    control_count = sum(1 for ch in text if ch not in allowed and unicodedata.category(ch) == "Cc")
    return control_count / len(text)


def compact_doc(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": row.get("document_id", ""),
        "event_id": row.get("event_id", ""),
        "language": row.get("language", ""),
        "source_filename": row.get("source_filename", ""),
    }


def compact_passage(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "passage_id": row.get("passage_id", ""),
        "document_id": row.get("document_id", ""),
        "language": row.get("language", ""),
        "source_filename": row.get("source_filename", ""),
    }


def expected_passage_id(row: dict[str, Any]) -> str:
    document_id = row.get("document_id", "")
    language = row.get("language") or "unknown"
    passage_index = row.get("passage_index")
    if not isinstance(passage_index, int):
        return ""
    return f"passage_{document_id}_{language}_{passage_index:04d}"


def qa_parsed_official_docs(
    parsed_dir: str | Path,
    output_path: str | Path | None = None,
    short_doc_chars: int = 1000,
    max_replacement_ratio: float = 0.01,
    max_control_ratio: float = 0.01,
) -> dict[str, Any]:
    parsed = Path(parsed_dir)
    doc_path = parsed / "official_doc_text.jsonl"
    passage_path = parsed / "passages.jsonl"
    summary_path = parsed / "parse_summary.json"

    docs = list(read_jsonl(doc_path))
    passages = list(read_jsonl(passage_path))
    parse_summary = read_json(summary_path, default={}) or {}

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not doc_path.exists():
        errors.append({"check": "missing_file", "path": doc_path.as_posix()})
    if not passage_path.exists():
        errors.append({"check": "missing_file", "path": passage_path.as_posix()})

    doc_missing_fields = [
        {**compact_doc(row), "missing_fields": sorted(DOC_REQUIRED_FIELDS - set(row))}
        for row in docs
        if DOC_REQUIRED_FIELDS - set(row)
    ]
    doc_forbidden_fields = [
        {**compact_doc(row), "forbidden_fields": sorted(DOC_FORBIDDEN_TOP_LEVEL & set(row))}
        for row in docs
        if DOC_FORBIDDEN_TOP_LEVEL & set(row)
    ]
    doc_missing_extra = [
        {**compact_doc(row), "missing_extra_fields": sorted(DOC_REQUIRED_EXTRA_FIELDS - set(row.get("extra", {})))}
        for row in docs
        if DOC_REQUIRED_EXTRA_FIELDS - set(row.get("extra", {}))
    ]
    if doc_missing_fields:
        errors.append({"check": "doc_missing_required_fields", "rows": doc_missing_fields[:25], "count": len(doc_missing_fields)})
    if doc_forbidden_fields:
        errors.append({"check": "doc_forbidden_top_level_fields", "rows": doc_forbidden_fields[:25], "count": len(doc_forbidden_fields)})
    if doc_missing_extra:
        errors.append({"check": "doc_missing_extra_fields", "rows": doc_missing_extra[:25], "count": len(doc_missing_extra)})

    passage_missing_fields = [
        {**compact_passage(row), "missing_fields": sorted(PASSAGE_REQUIRED_FIELDS - set(row))}
        for row in passages
        if PASSAGE_REQUIRED_FIELDS - set(row)
    ]
    passage_forbidden_fields = [
        {**compact_passage(row), "forbidden_fields": sorted(PASSAGE_FORBIDDEN_TOP_LEVEL & set(row))}
        for row in passages
        if PASSAGE_FORBIDDEN_TOP_LEVEL & set(row)
    ]
    passage_missing_extra = [
        {**compact_passage(row), "missing_extra_fields": sorted(PASSAGE_REQUIRED_EXTRA_FIELDS - set(row.get("extra", {})))}
        for row in passages
        if PASSAGE_REQUIRED_EXTRA_FIELDS - set(row.get("extra", {}))
    ]
    if passage_missing_fields:
        errors.append({"check": "passage_missing_required_fields", "rows": passage_missing_fields[:25], "count": len(passage_missing_fields)})
    if passage_forbidden_fields:
        errors.append({"check": "passage_forbidden_top_level_fields", "rows": passage_forbidden_fields[:25], "count": len(passage_forbidden_fields)})
    if passage_missing_extra:
        errors.append({"check": "passage_missing_extra_fields", "rows": passage_missing_extra[:25], "count": len(passage_missing_extra)})

    passage_id_counts = Counter(row.get("passage_id", "") for row in passages)
    duplicate_passage_ids = sorted(pid for pid, count in passage_id_counts.items() if pid and count > 1)
    if duplicate_passage_ids:
        errors.append({"check": "duplicate_passage_ids", "passage_ids": duplicate_passage_ids[:50], "count": len(duplicate_passage_ids)})

    doc_by_key = {(row.get("document_id"), row.get("language"), row.get("source_filename")): row for row in docs}
    orphan_passages: list[dict[str, Any]] = []
    invalid_char_spans: list[dict[str, Any]] = []
    text_mismatch_passages: list[dict[str, Any]] = []
    missing_page_fields: list[dict[str, Any]] = []
    invalid_page_ranges: list[dict[str, Any]] = []
    page_range_mismatches: list[dict[str, Any]] = []
    invalid_passage_id_formats: list[dict[str, Any]] = []
    for passage in passages:
        expected_id = expected_passage_id(passage)
        if expected_id and passage.get("passage_id") != expected_id:
            invalid_passage_id_formats.append({**compact_passage(passage), "expected_passage_id": expected_id})
        key = (passage.get("document_id"), passage.get("language"), passage.get("source_filename"))
        doc = doc_by_key.get(key)
        if doc is None:
            orphan_passages.append(compact_passage(passage))
            continue
        start = passage.get("char_start")
        end = passage.get("char_end")
        text = doc.get("text", "")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(text):
            invalid_char_spans.append(compact_passage(passage))
        elif passage.get("text") != text[start:end]:
            text_mismatch_passages.append(compact_passage(passage))
        elif "page_start" in passage and "page_end" in passage:
            expected_page_start, expected_page_end = page_range_for_span(doc.get("extra", {}).get("page_spans", []), start, end)
            if (passage.get("page_start"), passage.get("page_end")) != (expected_page_start, expected_page_end):
                page_range_mismatches.append(
                    {
                        **compact_passage(passage),
                        "page_start": passage.get("page_start"),
                        "page_end": passage.get("page_end"),
                        "expected_page_start": expected_page_start,
                        "expected_page_end": expected_page_end,
                    }
                )
        if "page_start" not in passage or "page_end" not in passage:
            missing_page_fields.append(compact_passage(passage))
        else:
            page_start = passage.get("page_start")
            page_end = passage.get("page_end")
            if not isinstance(page_start, int) or not isinstance(page_end, int) or page_start <= 0 or page_end < page_start:
                invalid_page_ranges.append(compact_passage(passage))

    for check, rows in (
        ("orphan_passages", orphan_passages),
        ("invalid_passage_char_spans", invalid_char_spans),
        ("passage_text_mismatches", text_mismatch_passages),
        ("passages_missing_page_fields", missing_page_fields),
        ("invalid_passage_page_ranges", invalid_page_ranges),
        ("page_range_mismatches", page_range_mismatches),
        ("invalid_passage_id_formats", invalid_passage_id_formats),
    ):
        if rows:
            errors.append({"check": check, "rows": rows[:25], "count": len(rows)})

    char_count_mismatches = [
        {**compact_doc(row), "char_count": row.get("char_count"), "actual_char_count": len(row.get("text", ""))}
        for row in docs
        if row.get("char_count") != len(row.get("text", ""))
    ]
    if char_count_mismatches:
        errors.append({"check": "doc_char_count_mismatches", "rows": char_count_mismatches[:25], "count": len(char_count_mismatches)})

    summary_mismatches = []
    if parse_summary:
        expected_actual = {
            "documents_total": len(docs),
            "passages_total": len(passages),
            "documents_ok": sum(1 for row in docs if row.get("parse_quality") == "ok"),
            "documents_low_text": sum(1 for row in docs if row.get("parse_quality") == "low_text"),
            "documents_failed": sum(1 for row in docs if row.get("parse_quality") == "failed"),
        }
        for key, actual in expected_actual.items():
            if parse_summary.get(key) != actual:
                summary_mismatches.append({"field": key, "summary": parse_summary.get(key), "actual": actual})
    if summary_mismatches:
        errors.append({"check": "parse_summary_mismatches", "rows": summary_mismatches, "count": len(summary_mismatches)})

    empty_docs = [compact_doc(row) for row in docs if not row.get("text")]
    short_docs = [
        {**compact_doc(row), "char_count": row.get("char_count", 0)}
        for row in docs
        if 0 < int(row.get("char_count") or 0) < short_doc_chars
    ]
    non_ok_docs = [
        {**compact_doc(row), "parse_quality": row.get("parse_quality", "")}
        for row in docs
        if row.get("parse_quality") != "ok"
    ]
    high_replacement_docs = [
        {**compact_doc(row), "replacement_char_ratio": round(high_replacement_ratio(row.get("text", ""), max_replacement_ratio), 6)}
        for row in docs
        if high_replacement_ratio(row.get("text", ""), max_replacement_ratio) > max_replacement_ratio
    ]
    high_control_docs = [
        {**compact_doc(row), "control_char_ratio": round(control_char_ratio(row.get("text", "")), 6)}
        for row in docs
        if control_char_ratio(row.get("text", "")) > max_control_ratio
    ]
    sha256_mismatches: list[dict[str, Any]] = []
    sha256_uncheckable: list[dict[str, Any]] = []
    for row in docs:
        local_path = Path(row.get("local_path", ""))
        if not local_path.exists():
            sha256_uncheckable.append(compact_doc(row))
            continue
        actual_sha256 = sha256_file(local_path)
        if row.get("sha256") != actual_sha256:
            sha256_mismatches.append({**compact_doc(row), "recorded_sha256": row.get("sha256", ""), "actual_sha256": actual_sha256})
    for check, rows in (
        ("empty_docs", empty_docs),
        ("short_docs", short_docs),
        ("non_ok_parse_quality_docs", non_ok_docs),
        ("high_replacement_char_ratio_docs", high_replacement_docs),
        ("high_control_char_ratio_docs", high_control_docs),
        ("sha256_uncheckable_docs", sha256_uncheckable),
    ):
        if rows:
            warnings.append({"check": check, "rows": rows[:25], "count": len(rows)})
    if sha256_mismatches:
        errors.append({"check": "sha256_mismatches", "rows": sha256_mismatches[:25], "count": len(sha256_mismatches)})

    source_metadata = [parse_source_filename(row.get("source_filename", "")) for row in docs]
    event_scope_counts: dict[str, Counter[str]] = defaultdict(Counter)
    viewpoint_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row, meta in zip(docs, source_metadata):
        event_id = str(row.get("event_id", ""))
        event_scope_counts[event_id][meta["evidence_scope"]] += 1
        viewpoint_counts[event_id][meta["viewpoint_origin"]] += 1

    language_groups: dict[str, set[str]] = defaultdict(set)
    for row in docs:
        language_groups[str(row.get("document_id", ""))].add(str(row.get("language", "")))
    document_language_groups = {doc_id: sorted(languages) for doc_id, languages in sorted(language_groups.items())}

    char_counts = [int(row.get("char_count") or 0) for row in docs]
    report = {
        "ok": not errors,
        "parsed_dir": parsed.as_posix(),
        "inputs": {
            "official_doc_text": doc_path.as_posix(),
            "passages": passage_path.as_posix(),
            "parse_summary": summary_path.as_posix(),
        },
        "counts": {
            "documents": len(docs),
            "passages": len(passages),
            "documents_ok": sum(1 for row in docs if row.get("parse_quality") == "ok"),
            "documents_low_text": sum(1 for row in docs if row.get("parse_quality") == "low_text"),
            "documents_failed": sum(1 for row in docs if row.get("parse_quality") == "failed"),
            "total_char_count": sum(char_counts),
            "min_char_count": min(char_counts) if char_counts else 0,
            "max_char_count": max(char_counts) if char_counts else 0,
        },
        "coverage": {
            "by_event": dict(sorted(Counter(row.get("event_id", "") for row in docs).items())),
            "by_language": dict(sorted(Counter(row.get("language", "") for row in docs).items())),
            "by_parser": dict(sorted(Counter(row.get("parser", "") for row in docs).items())),
            "by_parse_quality": dict(sorted(Counter(row.get("parse_quality", "") for row in docs).items())),
            "event_language_counts": nested_counter(docs, "event_id", "language"),
            "event_scope_counts": {event: dict(sorted(counter.items())) for event, counter in sorted(event_scope_counts.items())},
            "event_viewpoint_counts": {event: dict(sorted(counter.items())) for event, counter in sorted(viewpoint_counts.items())},
            "document_language_groups": document_language_groups,
            "multilingual_document_groups": sum(1 for languages in document_language_groups.values() if len(languages) > 1),
        },
        "integrity": {
            "duplicate_passage_ids": duplicate_passage_ids,
            "orphan_passages": orphan_passages[:25],
            "invalid_passage_char_spans": invalid_char_spans[:25],
            "passage_text_mismatches": text_mismatch_passages[:25],
            "passages_missing_page_fields": missing_page_fields[:25],
            "invalid_passage_page_ranges": invalid_page_ranges[:25],
            "page_range_mismatches": page_range_mismatches[:25],
            "invalid_passage_id_formats": invalid_passage_id_formats[:25],
            "doc_char_count_mismatches": char_count_mismatches[:25],
            "summary_mismatches": summary_mismatches,
            "sha256_mismatches": sha256_mismatches[:25],
        },
        "quality": {
            "empty_docs": empty_docs,
            "short_doc_threshold_chars": short_doc_chars,
            "short_docs": short_docs,
            "max_replacement_ratio": max_replacement_ratio,
            "high_replacement_char_ratio_docs": high_replacement_docs,
            "max_control_ratio": max_control_ratio,
            "high_control_char_ratio_docs": high_control_docs,
        },
        "errors": errors,
        "warnings": warnings,
    }

    if output_path:
        write_json(output_path, report)
    return report


def add_qa_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/0_external/official_doc_parsed"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--short-doc-chars", type=int, default=1000)
    parser.add_argument("--max-replacement-ratio", type=float, default=0.01)
    parser.add_argument("--max-control-ratio", type=float, default=0.01)
    return parser
