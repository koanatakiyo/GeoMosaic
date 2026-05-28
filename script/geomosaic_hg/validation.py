"""Validation helpers for GeoMosaic-HG JSONL tables."""

from __future__ import annotations

import json
from typing import Any

from .paths import PROJECT_ROOT
from .schema import EVIDENCE_ROLES, MATCH_LEVELS, MODALITIES, SOURCE_LAYERS, validate_table_rows


def _check_subset(errors: list[str], table: str, row_no: int, field: str, values: list[str], allowed: set[str]) -> None:
    for value in values:
        if value not in allowed:
            errors.append(f"{table}:{row_no} invalid {field}={value}")


def _check_unique(errors: list[str], table: str, rows: list[dict[str, Any]], field: str) -> None:
    seen: set[Any] = set()
    for i, row in enumerate(rows, 1):
        value = row.get(field)
        if value in seen:
            errors.append(f"{table}:{i} duplicate {field}={value}")
        seen.add(value)


def _check_range(errors: list[str], table: str, row_no: int, field: str, value: Any, lo: float = 0.0, hi: float = 1.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if value < lo or value > hi:
        errors.append(f"{table}:{row_no} {field} out of range [{lo}, {hi}]")


def validate_core_tables(tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    expected = ["source_records", "evidence_assets", "source_asset_links", "claim_evidence_hyperedges"]
    for name in expected:
        rows = tables.get(name, [])
        if not rows:
            errors.append(f"{name} is empty")
        errors.extend(validate_table_rows(name, rows))

    source_ids = {row.get("source_id") for row in tables.get("source_records", [])}
    asset_ids = {row.get("asset_id") for row in tables.get("evidence_assets", [])}
    forbidden = PROJECT_ROOT.parent.as_posix() + "/"
    _check_unique(errors, "source_records", tables.get("source_records", []), "source_id")
    _check_unique(errors, "evidence_assets", tables.get("evidence_assets", []), "asset_id")
    _check_unique(errors, "claim_evidence_hyperedges", tables.get("claim_evidence_hyperedges", []), "hyperedge_id")
    _check_unique(errors, "claim_evidence_hyperedges", tables.get("claim_evidence_hyperedges", []), "claim_id")

    for i, row in enumerate(tables.get("source_records", []), 1):
        if row.get("source_layer") not in SOURCE_LAYERS:
            errors.append(f"source_records:{i} invalid source_layer={row.get('source_layer')}")
        if forbidden in json.dumps(row, ensure_ascii=False):
            errors.append(f"source_records:{i} contains parent project absolute path")

    for i, row in enumerate(tables.get("evidence_assets", []), 1):
        if row.get("source_layer") not in SOURCE_LAYERS:
            errors.append(f"evidence_assets:{i} invalid source_layer={row.get('source_layer')}")
        if row.get("modality") not in MODALITIES:
            errors.append(f"evidence_assets:{i} invalid modality={row.get('modality')}")
        if row.get("evidence_role") not in EVIDENCE_ROLES:
            errors.append(f"evidence_assets:{i} invalid evidence_role={row.get('evidence_role')}")
        if not row.get("url_or_pointer") or not row.get("perceptual_hash"):
            errors.append(f"evidence_assets:{i} missing provenance pointer/hash")
        if forbidden in json.dumps(row, ensure_ascii=False):
            errors.append(f"evidence_assets:{i} contains parent project absolute path")

    for i, row in enumerate(tables.get("source_asset_links", []), 1):
        if row.get("source_id") not in source_ids:
            errors.append(f"source_asset_links:{i} unknown source_id={row.get('source_id')}")
        if row.get("asset_id") not in asset_ids:
            errors.append(f"source_asset_links:{i} unknown asset_id={row.get('asset_id')}")
        if row.get("match_level") not in MATCH_LEVELS:
            errors.append(f"source_asset_links:{i} invalid match_level={row.get('match_level')}")
        if row.get("evidence_role") not in EVIDENCE_ROLES:
            errors.append(f"source_asset_links:{i} invalid evidence_role={row.get('evidence_role')}")
        _check_range(errors, "source_asset_links", i, "match_score", row.get("match_score"))

    for i, row in enumerate(tables.get("claim_evidence_hyperedges", []), 1):
        for source_id in row.get("source_record_set", []):
            if source_id not in source_ids:
                errors.append(f"claim_evidence_hyperedges:{i} unknown source_id={source_id}")
        for asset_id in row.get("evidence_asset_set", []):
            if asset_id not in asset_ids:
                errors.append(f"claim_evidence_hyperedges:{i} unknown asset_id={asset_id}")
        _check_subset(errors, "claim_evidence_hyperedges", i, "source_layer_set", row.get("source_layer_set", []), SOURCE_LAYERS)
        _check_subset(errors, "claim_evidence_hyperedges", i, "modality_set", row.get("modality_set", []), MODALITIES)
        _check_subset(errors, "claim_evidence_hyperedges", i, "match_level_multiset", row.get("match_level_multiset", []), set(MATCH_LEVELS))
        _check_subset(errors, "claim_evidence_hyperedges", i, "evidence_role_multiset", row.get("evidence_role_multiset", []), EVIDENCE_ROLES)
        if not row.get("provenance_trace"):
            errors.append(f"claim_evidence_hyperedges:{i} missing provenance_trace")
        if forbidden in json.dumps(row, ensure_ascii=False):
            errors.append(f"claim_evidence_hyperedges:{i} contains parent project absolute path")
        _check_range(errors, "claim_evidence_hyperedges", i, "confidence", row.get("confidence"))
        _check_range(errors, "claim_evidence_hyperedges", i, "relevance", row.get("relevance"))

    return errors
