"""Dataclasses for the four GeoMosaic-HG core JSONL tables."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, get_args, get_origin, get_type_hints


SOURCE_LAYERS = {"official", "news", "wiki", "structured", "synthetic"}
MODALITIES = {"text", "image_full", "image_restricted_pointer", "map_pointer", "structured_event", "structured_document"}
MATCH_LEVELS = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
EVIDENCE_ROLES = {
    "substantive",
    "complementary",
    "highlighting",
    "map_like",
    "portrait",
    "symbolic",
    "decorative",
    "conflicting",
    "unverifiable",
    "context",
}


@dataclass
class SourceRecord:
    source_id: str
    event_id: str
    source_layer: str
    viewpoint_origin: str
    document_type: str
    institution_or_outlet: str
    publish_time: str
    retrieval_time: str
    url: str
    language: str
    license_or_terms: str
    redistribution_flag: bool
    normalized_text_hash: str
    local_path: str = ""
    word_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceAsset:
    asset_id: str
    event_id: str
    modality: str
    asset_source: str
    source_layer: str
    viewpoint_origin: str
    publish_time: str
    observed_time: str
    geo_location: str
    url_or_pointer: str
    caption_or_transcript: str
    license_or_terms: str
    redistribution_flag: bool
    perceptual_hash: str
    embedding_id: str
    extracted_entities: list[str]
    extracted_claims: list[str]
    evidence_role: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceAssetLink:
    source_id: str
    asset_id: str
    match_level: str
    match_score: float
    match_reason: str
    evidence_role: str
    verifier_status: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimEvidenceHyperedge:
    hyperedge_id: str
    claim_id: str
    event_id: str
    entity_set: list[str]
    source_record_set: list[str]
    evidence_asset_set: list[str]
    primary_source_layer_set: list[str]
    source_layer_set: list[str]
    modality_set: list[str]
    viewpoint_origin_set: list[str]
    match_level_multiset: list[str]
    evidence_role_multiset: list[str]
    time_span: dict[str, str]
    provenance_trace: list[str]
    confidence: float
    claim_text: str = ""
    relation: str = "context"
    relevance: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


TABLE_CLASSES = {
    "source_records": SourceRecord,
    "evidence_assets": EvidenceAsset,
    "source_asset_links": SourceAssetLink,
    "claim_evidence_hyperedges": ClaimEvidenceHyperedge,
}


def _type_name(expected: Any) -> str:
    origin = get_origin(expected)
    args = get_args(expected)
    if expected is Any:
        return "Any"
    if origin is list and args:
        return f"list[{_type_name(args[0])}]"
    if origin is dict and args:
        return f"dict[{_type_name(args[0])}, {_type_name(args[1])}]"
    if isinstance(expected, type):
        return expected.__name__
    return str(expected)


def _matches_type(value: Any, expected: Any) -> bool:
    if expected is Any:
        return True
    origin = get_origin(expected)
    args = get_args(expected)
    if origin is list:
        if not isinstance(value, list):
            return False
        item_type = args[0] if args else Any
        return all(_matches_type(item, item_type) for item in value)
    if origin is dict:
        if not isinstance(value, dict):
            return False
        key_type, value_type = args if args else (Any, Any)
        return all(_matches_type(k, key_type) and _matches_type(v, value_type) for k, v in value.items())
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    if isinstance(expected, type):
        return isinstance(value, expected)
    return True


def dataclass_from_dict(cls: type, data: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    values = {k: v for k, v in data.items() if k in names}
    extra = {k: v for k, v in data.items() if k not in names}
    if "extra" in names:
        values["extra"] = {**extra, **values.get("extra", {})}
    return cls(**values)


def as_clean_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        data = asdict(obj)
    else:
        data = dict(obj)
    if isinstance(data.get("extra"), dict) and not data["extra"]:
        data.pop("extra")
    return data


def match_level_value(level: str) -> int:
    if level not in MATCH_LEVELS:
        raise ValueError(f"Unknown match level: {level}")
    return MATCH_LEVELS[level]


def max_match_level(levels: list[str]) -> str:
    if not levels:
        return "L4"
    return max(levels, key=match_level_value)


def validate_table_rows(table_name: str, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    cls = TABLE_CLASSES[table_name]
    type_hints = get_type_hints(cls)
    required = [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING]
    field_names = {f.name for f in fields(cls)}
    for idx, row in enumerate(rows, 1):
        for name in required:
            if name not in row:
                errors.append(f"{table_name}:{idx} missing {name}")
        unknown = sorted(set(row) - field_names)
        if unknown:
            errors.append(f"{table_name}:{idx} unknown fields {unknown}")
        for name, expected in type_hints.items():
            if name in row and not _matches_type(row[name], expected):
                errors.append(f"{table_name}:{idx} {name} expected {_type_name(expected)}")
    return errors
