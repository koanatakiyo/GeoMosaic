"""Offline construction pipeline for GeoMosaic-Bench core tables."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .events import (
    EVENTS,
    ParsedRawText,
    display_viewpoint,
    normalize_viewpoint,
    parse_news_pdf_path,
    parse_raw_text_path,
    score_source_layer_from_dir,
)
from .io import read_jsonl, sha256_file, sha256_text, slug, stable_hash, write_json, write_jsonl
from .paths import BENCH_DIR, DATA_DIR, RAW_DIR, SCORE_DIR, ensure_dir, relative_to_project
from .schema import (
    ClaimEvidenceHyperedge,
    EvidenceAsset,
    SourceAssetLink,
    SourceRecord,
    as_clean_dict,
    dataclass_from_dict,
    match_level_value,
)
from .wiki_pages import WIKIPEDIA_EVENT_PAGE_ALIASES, WIKIPEDIA_EVENT_PAGES, wikipedia_page_url


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def short_text(text: str, limit: int = 800) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut or text[:limit]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def iter_raw_texts(raw_dir: Path, event_ids: set[str] | None = None) -> Iterable[tuple[ParsedRawText, str]]:
    for path in sorted(raw_dir.glob("*/*.txt")):
        parsed = parse_raw_text_path(path)
        if not parsed:
            continue
        if event_ids and parsed.event_id not in event_ids:
            continue
        yield parsed, path.read_text(encoding="utf-8", errors="replace")


def source_id_for(path: Path) -> str:
    return f"src_{slug(path.stem)}"


def asset_id_for_source(source_id: str) -> str:
    return f"asset_text_{source_id}"


def build_source_records(raw_dir: Path = RAW_DIR, event_ids: set[str] | None = None) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for parsed, text in iter_raw_texts(raw_dir, event_ids):
        info = EVENTS.get(parsed.event_id)
        if info is None:
            continue
        source_id = source_id_for(parsed.path)
        layer = parsed.source_layer
        redistributable = layer in {"official", "wiki"}
        if layer == "official":
            terms = "official-public-source-or-local-research-copy"
        elif layer == "wiki":
            terms = "wikipedia-derived-local-text"
        else:
            terms = "restricted-news-local-research-copy"
        extra = {
            "source_key": parsed.source_key,
            "event_name": info.name,
            "subject": info.subject,
            "text_preview": short_text(text, 360),
        }
        if layer == "wiki":
            source_page_title = WIKIPEDIA_EVENT_PAGES.get(parsed.event_id, "")
            if source_page_title:
                aliases = [source_page_title, *WIKIPEDIA_EVENT_PAGE_ALIASES.get(parsed.event_id, ())]
                extra["source_page_title"] = source_page_title
                extra["source_page_aliases"] = aliases
                extra["source_page_url"] = wikipedia_page_url(source_page_title)
        records.append(
            SourceRecord(
                source_id=source_id,
                event_id=parsed.event_id,
                source_layer=layer,
                viewpoint_origin=parsed.viewpoint_origin,
                document_type=parsed.document_type,
                institution_or_outlet=parsed.institution_or_outlet,
                publish_time=info.publish_time,
                retrieval_time=info.publish_time,
                url=relative_to_project(parsed.path),
                language=parsed.language,
                license_or_terms=terms,
                redistribution_flag=redistributable,
                normalized_text_hash=sha256_text(normalize_text(text)),
                local_path=relative_to_project(parsed.path),
                word_count=word_count(text),
                extra=extra,
            )
        )
    return records


def build_text_assets(source_records: list[SourceRecord]) -> list[EvidenceAsset]:
    assets: list[EvidenceAsset] = []
    for rec in source_records:
        info = EVENTS[rec.event_id]
        preview = rec.extra.get("text_preview", "")
        entities = sorted({info.subject, display_viewpoint(rec.viewpoint_origin), rec.institution_or_outlet})
        if rec.source_layer == "wiki":
            role = "context"
        elif rec.source_layer == "news":
            role = "complementary"
        else:
            role = "substantive"
        asset_id = asset_id_for_source(rec.source_id)
        assets.append(
            EvidenceAsset(
                asset_id=asset_id,
                event_id=rec.event_id,
                modality="text",
                asset_source=rec.institution_or_outlet,
                source_layer=rec.source_layer,
                viewpoint_origin=rec.viewpoint_origin,
                publish_time=rec.publish_time,
                observed_time=rec.retrieval_time,
                geo_location=info.geo_location,
                url_or_pointer=rec.local_path or rec.url,
                caption_or_transcript=preview,
                license_or_terms=rec.license_or_terms,
                redistribution_flag=rec.redistribution_flag,
                perceptual_hash=rec.normalized_text_hash,
                embedding_id=f"emb_{asset_id}",
                extracted_entities=entities,
                extracted_claims=[],
                evidence_role=role,
                extra={"source_id": rec.source_id, "source_key": rec.extra.get("source_key", "")},
            )
        )
    return assets


def build_pdf_pointer_assets(raw_dir: Path = RAW_DIR, event_ids: set[str] | None = None) -> list[EvidenceAsset]:
    assets: list[EvidenceAsset] = []
    pdf_dir = raw_dir / "raw_news"
    if not pdf_dir.exists():
        return assets
    for path in sorted(pdf_dir.glob("*.pdf")):
        parsed = parse_news_pdf_path(path)
        if not parsed or parsed.event_id not in EVENTS:
            continue
        if event_ids and parsed.event_id not in event_ids:
            continue
        info = EVENTS[parsed.event_id]
        file_hash = sha256_file(path)
        asset_id = f"asset_newspdf_{parsed.event_id}_{slug(parsed.outlet)}_{parsed.article_idx}"
        assets.append(
            EvidenceAsset(
                asset_id=asset_id,
                event_id=parsed.event_id,
                modality="image_restricted_pointer",
                asset_source=parsed.outlet,
                source_layer="news",
                viewpoint_origin="all",
                publish_time=info.publish_time,
                observed_time=info.publish_time,
                geo_location=info.geo_location,
                url_or_pointer=relative_to_project(path),
                caption_or_transcript=f"Restricted local news PDF pointer for {info.name}, {parsed.outlet} article {parsed.article_idx}.",
                license_or_terms="restricted-news-pointer-only",
                redistribution_flag=False,
                perceptual_hash=file_hash,
                embedding_id=f"emb_{asset_id}",
                extracted_entities=[info.subject, parsed.outlet],
                extracted_claims=[],
                evidence_role="complementary",
                extra={"outlet": parsed.outlet, "article_idx": parsed.article_idx},
            )
        )
    return assets


def build_map_pointer_assets(event_ids: set[str] | None = None) -> list[EvidenceAsset]:
    assets: list[EvidenceAsset] = []
    for event_id, info in EVENTS.items():
        if event_ids and event_id not in event_ids:
            continue
        if not info.map_eligible:
            continue
        asset_id = f"asset_map_{event_id}"
        wiki_slug = info.subject.replace(" ", "_")
        assets.append(
            EvidenceAsset(
                asset_id=asset_id,
                event_id=event_id,
                modality="map_pointer",
                asset_source="Wikipedia/Wikimedia pointer",
                source_layer="wiki",
                viewpoint_origin="all",
                publish_time=info.publish_time,
                observed_time=info.publish_time,
                geo_location=info.geo_location,
                url_or_pointer=f"https://en.wikipedia.org/wiki/{wiki_slug}",
                caption_or_transcript=f"Map-oriented provenance pointer for {info.name}.",
                license_or_terms="pointer-only-check-license-before-redistribution",
                redistribution_flag=False,
                perceptual_hash=sha256_text(f"map:{event_id}:{wiki_slug}"),
                embedding_id=f"emb_{asset_id}",
                extracted_entities=[info.subject],
                extracted_claims=[],
                evidence_role="map_like",
                extra={"generated_pointer": True},
            )
        )
    return assets


def build_structured_event_assets(event_ids: set[str] | None = None) -> list[EvidenceAsset]:
    assets: list[EvidenceAsset] = []
    for event_id, info in EVENTS.items():
        if event_ids and event_id not in event_ids:
            continue
        asset_id = f"asset_event_{event_id}"
        text = f"Structured event row for {info.name}: subject={info.subject}; date={info.publish_time}; location={info.geo_location}."
        assets.append(
            EvidenceAsset(
                asset_id=asset_id,
                event_id=event_id,
                modality="structured_event",
                asset_source="GeoMosaic event registry",
                source_layer="structured",
                viewpoint_origin="all",
                publish_time=info.publish_time,
                observed_time=info.publish_time,
                geo_location=info.geo_location,
                url_or_pointer=f"geomosaic://event/{event_id}",
                caption_or_transcript=text,
                license_or_terms="local-derived-metadata",
                redistribution_flag=True,
                perceptual_hash=sha256_text(text),
                embedding_id=f"emb_{asset_id}",
                extracted_entities=[info.subject, info.geo_location],
                extracted_claims=[],
                evidence_role="context",
                extra={"event_name": info.name},
            )
        )
    return assets


def load_external_assets(external_dir: Path | None = None, event_ids: set[str] | None = None) -> list[EvidenceAsset]:
    external_dir = external_dir or DATA_DIR / "0_external"
    if not external_dir.exists():
        return []

    assets: list[EvidenceAsset] = []
    preferred = external_dir / "external_assets.jsonl"
    auxiliary_names = {"candidate_inventory.jsonl", "selection_decisions.jsonl", "manifest.jsonl"}
    paths = [preferred] if preferred.exists() else [path for path in sorted(external_dir.glob("*.jsonl")) if path.name not in auxiliary_names]
    for path in paths:
        for row in read_jsonl(path):
            event_id = str(row.get("event_id", ""))
            if event_id not in EVENTS:
                continue
            if event_ids and event_id not in event_ids:
                continue
            try:
                asset = dataclass_from_dict(EvidenceAsset, row)
            except TypeError as exc:
                raise ValueError(f"Invalid external EvidenceAsset row in {path}: {exc}") from exc
            asset.extra = {**asset.extra, "external_file": relative_to_project(path)}
            assets.append(asset)
    return assets


def build_evidence_assets(
    source_records: list[SourceRecord],
    raw_dir: Path = RAW_DIR,
    event_ids: set[str] | None = None,
    external_dir: Path | None = None,
) -> list[EvidenceAsset]:
    return [
        *build_text_assets(source_records),
        *build_pdf_pointer_assets(raw_dir, event_ids),
        *build_map_pointer_assets(event_ids),
        *build_structured_event_assets(event_ids),
        *load_external_assets(external_dir, event_ids),
    ]


def _is_page_bound_wiki_asset(source: SourceRecord, asset: EvidenceAsset) -> bool:
    if not (
        source.source_layer == "wiki"
        and source.extra.get("source_key") == "wiki"
        and asset.source_layer == "wiki"
        and asset.extra.get("collection_channel") == "wikipedia_page_bound"
        and asset.extra.get("page_bound") is True
    ):
        return False
    source_title = str(source.extra.get("source_page_title") or "").strip().lower()
    asset_title = str(asset.extra.get("source_page_title") or "").strip().lower()
    if source_title and asset_title:
        aliases = source.extra.get("source_page_aliases") or [source_title]
        normalized_aliases = {str(alias).strip().lower() for alias in aliases if str(alias).strip()}
        return asset_title in normalized_aliases
    source_url = str(source.extra.get("source_page_url") or "").strip().lower()
    asset_url = str(asset.extra.get("source_page_url") or "").strip().lower()
    return bool(source_url and asset_url and source_url == asset_url)


def build_source_asset_links(source_records: list[SourceRecord], assets: list[EvidenceAsset]) -> list[SourceAssetLink]:
    links: list[SourceAssetLink] = []
    text_asset_by_source = {a.extra.get("source_id"): a for a in assets if a.modality == "text"}
    assets_by_event: dict[str, list[EvidenceAsset]] = defaultdict(list)
    for asset in assets:
        assets_by_event[asset.event_id].append(asset)

    seen: set[tuple[str, str]] = set()

    def add(source: SourceRecord, asset: EvidenceAsset, level: str, score: float, reason: str, role: str | None = None) -> None:
        key = (source.source_id, asset.asset_id)
        if key in seen:
            return
        seen.add(key)
        links.append(
            SourceAssetLink(
                source_id=source.source_id,
                asset_id=asset.asset_id,
                match_level=level,
                match_score=round(score, 4),
                match_reason=reason,
                evidence_role=role or asset.evidence_role,
                verifier_status="auto_heuristic",
            )
        )

    for source in source_records:
        text_asset = text_asset_by_source.get(source.source_id)
        if text_asset:
            add(source, text_asset, "L0", 1.0, "asset is the text representation of the source record")

        for asset in assets_by_event[source.event_id]:
            if asset.modality == "text":
                continue
            if asset.modality == "image_restricted_pointer" and source.source_layer == "news":
                source_key = str(source.extra.get("source_key", ""))
                if _is_exact_news_pointer({"source_key": source_key}, asset):
                    add(source, asset, "L1", 0.78, "same news article pointer")
                elif asset.source_layer == "news":
                    add(source, asset, "L2", 0.55, "same event and news source layer")
            elif _is_page_bound_wiki_asset(source, asset):
                add(source, asset, "L1", 0.82, "Wikipedia page-bound media from the same source article")
            elif asset.source_layer == source.source_layer and asset.asset_source == source.institution_or_outlet:
                add(source, asset, "L1", 0.78, "same event and same publisher/outlet pointer")
            elif asset.modality in {"structured_event", "structured_document", "map_pointer"}:
                if asset.modality == "map_pointer":
                    add(source, asset, "L4", 0.45, "event-level contextual provenance pointer")
                elif asset.modality == "structured_document" and asset.source_layer == source.source_layer:
                    add(source, asset, "L2", 0.65, "same source layer official structured document")
                else:
                    add(source, asset, "L3", 0.45, "event-level contextual provenance pointer")
            elif asset.source_layer == "news" and source.source_layer == "news":
                add(source, asset, "L2", 0.55, "same event and news source layer")
            else:
                add(source, asset, "L3", 0.35, "same event context across source layers")

    return links


def load_claim_audit_groups(score_dirs: list[Path]) -> dict[tuple[Any, ...], dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for score_dir in score_dirs:
        if not score_dir.exists():
            continue
        layer = score_source_layer_from_dir(score_dir)
        for path in sorted(score_dir.glob("*_claim_audit.jsonl")):
            for row in read_jsonl(path):
                event_id = str(row.get("event", "")).lower()
                if event_id not in EVENTS:
                    continue
                source_vp = str(row.get("source_vp", "all"))
                scored_vp = normalize_viewpoint(str(row.get("scored_vp", "all")))
                claim_id = str(row.get("claim_id", "unknown"))
                outlet = row.get("outlet")
                article_idx = row.get("article_idx")
                key = (event_id, layer, source_vp, outlet, article_idx, scored_vp, claim_id)
                if key not in groups:
                    groups[key] = {
                        "event_id": event_id,
                        "source_layer": layer,
                        "source_vp": source_vp,
                        "outlet": outlet,
                        "article_idx": article_idx,
                        "scored_vp": scored_vp,
                        "claim_id": claim_id,
                        "score_sum": 0.0,
                        "max_sum": 0.0,
                        "n": 0,
                        "justifications": [],
                        "models": set(),
                    }
                g = groups[key]
                score = row.get("score")
                max_score = row.get("max", 2)
                if score is not None:
                    g["score_sum"] += float(score)
                    g["max_sum"] += float(max_score or 0)
                g["n"] += 1
                if row.get("justification"):
                    g["justifications"].append(str(row["justification"]))
                if row.get("model"):
                    g["models"].add(str(row["model"]))
    return groups


def _news_source_vp_parts(source_vp: str) -> tuple[str | None, int | None]:
    m = re.match(r"^news[-_ ](?P<outlet>guardian|reuters|wsj)[-_ ](?P<idx>\d+)$", source_vp.strip(), re.I)
    if not m:
        return None, None
    return m.group("outlet").title() if m.group("outlet").lower() != "wsj" else "WSJ", int(m.group("idx"))


def _news_source_key_parts(source_key: str) -> tuple[str | None, int | None]:
    m = re.match(r"^news-(?P<outlet>guardian|reuters|wsj)(?:-(?P<idx>\d+))?$", source_key.strip(), re.I)
    if not m:
        return None, None
    outlet = m.group("outlet").title() if m.group("outlet").lower() != "wsj" else "WSJ"
    idx = int(m.group("idx")) if m.group("idx") else None
    return outlet, idx


def _is_exact_news_pointer(group_or_source: dict[str, Any], asset: EvidenceAsset) -> bool:
    outlet = group_or_source.get("outlet")
    article_idx = group_or_source.get("article_idx")
    if outlet is None and group_or_source.get("source_key"):
        outlet, article_idx = _news_source_key_parts(str(group_or_source.get("source_key", "")))
    if not outlet or article_idx is None:
        return False
    return (
        asset.modality == "image_restricted_pointer"
        and asset.extra.get("outlet") == outlet
        and asset.extra.get("article_idx") == int(article_idx)
    )


def select_source_records(group: dict[str, Any], records_by_event: dict[str, list[SourceRecord]]) -> list[SourceRecord]:
    event_records = records_by_event.get(group["event_id"], [])
    layer = group["source_layer"]
    if layer == "official":
        vp = normalize_viewpoint(group["source_vp"])
        matched = [r for r in event_records if r.source_layer == "official" and r.viewpoint_origin == vp]
        return matched[:1] or [r for r in event_records if r.source_layer == "official"][:1]
    if layer == "news":
        outlet = group.get("outlet")
        article_idx = group.get("article_idx")
        if outlet is None or article_idx is None:
            outlet, article_idx = _news_source_vp_parts(str(group.get("source_vp", "")))
        if outlet and article_idx:
            key = f"news-{str(outlet).lower()}-{article_idx}"
            matched = [r for r in event_records if r.source_layer == "news" and r.extra.get("source_key") == key]
            if matched:
                return matched[:1]
        return [r for r in event_records if r.source_layer == "news"][:1]
    if layer == "wiki":
        return [r for r in event_records if r.source_layer == "wiki"][:1]
    return event_records[:1]


def _asset_priority(group: dict[str, Any], selected_sources: set[str], asset: EvidenceAsset) -> tuple[int, str]:
    if asset.extra.get("source_id") in selected_sources:
        return (0, asset.asset_id)
    outlet = group.get("outlet")
    article_idx = group.get("article_idx")
    if asset.modality == "image_restricted_pointer" and outlet and article_idx:
        if asset.extra.get("outlet") == outlet and asset.extra.get("article_idx") == int(article_idx):
            return (1, asset.asset_id)
    if asset.source_layer == "wiki" and asset.modality == "text":
        return (2, asset.asset_id)
    if asset.extra.get("external_file") and asset.modality == "image_full":
        return (3, asset.asset_id)
    if asset.modality == "map_pointer":
        return (3, asset.asset_id)
    if asset.modality == "structured_document":
        return (3, asset.asset_id)
    if asset.modality == "structured_event":
        return (4, asset.asset_id)
    return (5, asset.asset_id)


def select_assets_for_hyperedge(group: dict[str, Any], selected_records: list[SourceRecord], assets_by_event: dict[str, list[EvidenceAsset]]) -> list[EvidenceAsset]:
    selected_sources = {r.source_id for r in selected_records}
    assets = sorted(assets_by_event.get(group["event_id"], []), key=lambda a: _asset_priority(group, selected_sources, a))
    out: list[EvidenceAsset] = []
    seen: set[str] = set()

    def add(asset: EvidenceAsset) -> None:
        if asset.asset_id in seen:
            return
        out.append(asset)
        seen.add(asset.asset_id)

    for asset in assets:
        priority = _asset_priority(group, selected_sources, asset)[0]
        if priority <= 4:
            add(asset)
        if len(out) >= 4 and any(a.extra.get("source_id") in selected_sources for a in out):
            break
    if any(a.extra.get("external_file") for a in out) and not any(a.modality == "structured_event" for a in out):
        structured = sorted(
            (a for a in assets if a.modality == "structured_event"),
            key=lambda a: (a.asset_source != "GeoMosaic event registry", a.asset_id),
        )
        if structured:
            add(structured[0])
    if len(out) < 2:
        for asset in assets:
            add(asset)
            if len(out) >= 2:
                break
    return out


def match_level_for_hyperedge_asset(group: dict[str, Any], selected_sources: set[str], asset: EvidenceAsset) -> str:
    if asset.extra.get("source_id") in selected_sources:
        return "L0"
    if _is_exact_news_pointer(group, asset):
        return "L1"
    if asset.source_layer == group["source_layer"] and asset.modality not in {"map_pointer", "structured_event"}:
        return "L2"
    if asset.modality == "map_pointer":
        return "L4"
    return "L3"


def relation_from_confidence(confidence: float) -> str:
    if confidence >= 0.67:
        return "support"
    if confidence <= 0.05:
        return "conflicting"
    return "context"


def claim_text_for(group: dict[str, Any], info_name: str) -> str:
    if group["justifications"]:
        return short_text(" ".join(group["justifications"]), 420)
    vp = display_viewpoint(group["scored_vp"])
    return f"{info_name} claim {group['claim_id']} evaluated for {vp} viewpoint."


def build_claim_hyperedges(
    source_records: list[SourceRecord],
    assets: list[EvidenceAsset],
    score_dirs: list[Path] | None = None,
) -> list[ClaimEvidenceHyperedge]:
    if score_dirs is None:
        score_dirs = [SCORE_DIR / "combined_official", SCORE_DIR / "combined_zh_news"]
    groups = load_claim_audit_groups(score_dirs)
    records_by_event: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in source_records:
        records_by_event[record.event_id].append(record)
    assets_by_event: dict[str, list[EvidenceAsset]] = defaultdict(list)
    for asset in assets:
        assets_by_event[asset.event_id].append(asset)

    hyperedges: list[ClaimEvidenceHyperedge] = []
    for group in groups.values():
        event_id = group["event_id"]
        info = EVENTS[event_id]
        selected_records = select_source_records(group, records_by_event)
        selected_assets = select_assets_for_hyperedge(group, selected_records, assets_by_event)
        if not selected_records or not selected_assets:
            continue
        confidence = group["score_sum"] / group["max_sum"] if group["max_sum"] else 0.0
        claim_id = f"{event_id}:{group['source_layer']}:{slug(str(group['source_vp']))}:{display_viewpoint(group['scored_vp'])}:{group['claim_id']}"
        hyperedge_id = f"he_{stable_hash(claim_id)}"
        source_layers = sorted({r.source_layer for r in selected_records} | {a.source_layer for a in selected_assets})
        modalities = sorted({a.modality for a in selected_assets})
        viewpoints = sorted(
            {r.viewpoint_origin for r in selected_records}
            | {a.viewpoint_origin for a in selected_assets}
            | {group["scored_vp"]}
        )
        selected_source_ids = {r.source_id for r in selected_records}
        match_levels = []
        context_match_levels = []
        roles = []
        for asset in selected_assets:
            level = match_level_for_hyperedge_asset(group, selected_source_ids, asset)
            if match_level_value(level) <= match_level_value("L2"):
                match_levels.append(level)
            else:
                context_match_levels.append(level)
            roles.append(asset.evidence_role)
        times = [a.publish_time for a in selected_assets if a.publish_time]
        source_ids = sorted({r.source_id for r in selected_records})
        asset_ids = sorted({a.asset_id for a in selected_assets})
        provenance = [
            f"{a.asset_id}:{a.perceptual_hash[:16]}:{a.url_or_pointer}"
            for a in selected_assets
            if a.url_or_pointer and a.perceptual_hash
        ]
        hyperedges.append(
            ClaimEvidenceHyperedge(
                hyperedge_id=hyperedge_id,
                claim_id=claim_id,
                event_id=event_id,
                entity_set=sorted({info.subject, display_viewpoint(group["scored_vp"])}),
                source_record_set=source_ids,
                evidence_asset_set=asset_ids,
                primary_source_layer_set=sorted({r.source_layer for r in selected_records}),
                source_layer_set=source_layers,
                modality_set=modalities,
                viewpoint_origin_set=viewpoints,
                match_level_multiset=match_levels or context_match_levels,
                evidence_role_multiset=roles,
                time_span={"start": min(times), "end": max(times)} if times else {"start": info.publish_time, "end": info.publish_time},
                provenance_trace=provenance,
                confidence=round(confidence, 4),
                claim_text=claim_text_for(group, info.name),
                relation=relation_from_confidence(confidence),
                relevance=round(confidence, 4),
                extra={
                    "audit_rows": group["n"],
                    "source_vp": group["source_vp"],
                    "scored_vp": group["scored_vp"],
                    "models": sorted(group["models"]),
                    "context_match_level_multiset": context_match_levels,
                },
            )
        )
    return sorted(hyperedges, key=lambda h: h.hyperedge_id)


def build_all(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = BENCH_DIR,
    score_dirs: list[Path] | None = None,
    event_ids: set[str] | None = None,
    external_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    if score_dirs is None:
        score_dirs = [SCORE_DIR / "combined_official", SCORE_DIR / "combined_zh_news"]
    source_records = build_source_records(raw_dir, event_ids)
    assets = build_evidence_assets(source_records, raw_dir, event_ids, external_dir)
    links = build_source_asset_links(source_records, assets)
    hyperedges = build_claim_hyperedges(source_records, assets, score_dirs)

    if event_ids:
        hyperedges = [h for h in hyperedges if h.event_id in event_ids]

    counts = {
        "source_records": write_jsonl(output_dir / "source_records.jsonl", [as_clean_dict(r) for r in source_records]),
        "evidence_assets": write_jsonl(output_dir / "evidence_assets.jsonl", [as_clean_dict(a) for a in assets]),
        "source_asset_links": write_jsonl(output_dir / "source_asset_links.jsonl", [as_clean_dict(l) for l in links]),
        "claim_evidence_hyperedges": write_jsonl(output_dir / "claim_evidence_hyperedges.jsonl", [as_clean_dict(h) for h in hyperedges]),
    }
    summary = {
        "output_dir": relative_to_project(output_dir),
        "counts": counts,
        "events": sorted({r.event_id for r in source_records}),
        "score_dirs": [relative_to_project(p) for p in score_dirs if p.exists()],
        "external_assets": len([a for a in assets if a.extra.get("external_file")]),
        "external_asset_files": sorted({a.extra["external_file"] for a in assets if a.extra.get("external_file")}),
        "notes": [
            "Generated offline from project-local raw text, news PDF pointers, and claim-audit outputs.",
            "Optional external assets are loaded from data/0_external/external_assets.jsonl when present.",
            "No path in generated records requires the parent geopo directory.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def table_paths(output_dir: Path = BENCH_DIR) -> dict[str, Path]:
    return {
        "source_records": output_dir / "source_records.jsonl",
        "evidence_assets": output_dir / "evidence_assets.jsonl",
        "source_asset_links": output_dir / "source_asset_links.jsonl",
        "claim_evidence_hyperedges": output_dir / "claim_evidence_hyperedges.jsonl",
    }


def load_tables(output_dir: Path = BENCH_DIR) -> dict[str, list[dict[str, Any]]]:
    paths = table_paths(output_dir)
    return {name: list(read_jsonl(path)) for name, path in paths.items()}


def dataclass_counts(rows: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[type(row).__name__] += 1
    return dict(counts)
