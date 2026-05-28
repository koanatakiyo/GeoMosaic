#!/usr/bin/env python3
"""Generate EvidenceAsset JSONL records for curated official structured documents.

Covers the four events where ACLED returns 0 rows:
  jcpoa, scs, kosovo, libya

Each record is hand-curated here (document URL, institution, date, entities).
The script serialises them into EvidenceAsset schema with:
  modality        = "structured_document"
  source_layer    = "official"
  asset_source    = "OFFICIAL_DOC"
  record_type     = "official_resolution" | "agreement_text" | "arbitration_award" |
                    "legal_advisory" | "declaration_text" | "official_statement"
  curation_level  = "official"
  active_policy   = "primary_official_evidence"

Output: data/0_external/external_asset_raw/official_{event_id}.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from geomosaic_hg.events import EVENTS


# ── Curated document catalogue ─────────────────────────────────────────────────
# Fields per entry:
#   title, caption, institution, document_type, publish_time (ISO),
#   url, viewpoint_origin, entities, geo_location, license_or_terms,
#   redistribution_flag (bool, default False)
OFFICIAL_DOCS: dict[str, list[dict]] = {
    "jcpoa": [
        {
            "title": "Joint Comprehensive Plan of Action (JCPOA)",
            "caption": (
                "Full text of the Joint Comprehensive Plan of Action agreed by Iran "
                "and the P5+1 (US, UK, France, Russia, China, Germany plus EU) on "
                "14 July 2015 in Vienna, setting verifiable limits on Iran's nuclear "
                "programme in exchange for comprehensive sanctions relief."
            ),
            "institution": "EU External Action Service / P5+1",
            "document_type": "agreement_text",
            "publish_time": "2015-07-14T00:00:00Z",
            "url": "https://eeas.europa.eu/archives/docs/statements-eeas/docs/iran_agreement/iran_joint-comprehensive-plan-of-action_en.pdf",
            "viewpoint_origin": "multilateral",
            "entities": ["Iran", "United States", "European Union", "Russia", "China",
                         "United Kingdom", "France", "Germany", "P5+1", "JCPOA", "Vienna"],
            "geo_location": "Vienna, Austria",
            "license_or_terms": "Public official document",
        },
        {
            "title": "UN Security Council Resolution 2231 (2015)",
            "caption": (
                "UNSC Resolution 2231 (20 July 2015) endorsed the JCPOA and "
                "established a revised sanctions architecture for Iran, terminating "
                "resolutions 1737, 1747, 1803, and 1929 contingent on IAEA "
                "verification of Iranian compliance. Adopted unanimously 15–0."
            ),
            "institution": "UN Security Council",
            "document_type": "official_resolution",
            "publish_time": "2015-07-20T00:00:00Z",
            "url": "https://undocs.org/S/RES/2231(2015)",
            "viewpoint_origin": "multilateral",
            "entities": ["UN Security Council", "Iran", "JCPOA", "IAEA", "P5+1"],
            "geo_location": "New York, United States",
            "license_or_terms": "UN public document",
        },
        {
            "title": "IAEA Report on Iran's JCPOA Commitments — Implementation Day",
            "caption": (
                "Report by IAEA Director General Yukiya Amano confirming that as of "
                "16 January 2016 Iran had completed all nuclear steps required under "
                "the JCPOA, triggering 'Implementation Day' and the simultaneous "
                "lifting of EU, US, and UN nuclear-related sanctions."
            ),
            "institution": "International Atomic Energy Agency",
            "document_type": "official_statement",
            "publish_time": "2016-01-16T00:00:00Z",
            "url": "https://www.iaea.org/sites/default/files/16/01/gov2016-1.pdf",
            "viewpoint_origin": "multilateral",
            "entities": ["IAEA", "Iran", "JCPOA", "Implementation Day", "Yukiya Amano"],
            "geo_location": "Vienna, Austria",
            "license_or_terms": "IAEA public document",
        },
    ],

    "scs": [
        {
            "title": "South China Sea Arbitration — Final Award (PCA Case No. 2013-19)",
            "caption": (
                "Final Award (12 July 2016) of the Arbitral Tribunal constituted under "
                "Annex VII of UNCLOS in Philippines v. China. The Tribunal ruled (a) "
                "China's nine-dash line maritime claims have no legal basis under UNCLOS, "
                "(b) no disputed feature in the Spratly Islands qualifies as an island "
                "under Art. 121, and (c) China violated the Philippines' sovereign rights "
                "in its EEZ through fishing and construction activities."
            ),
            "institution": "Permanent Court of Arbitration",
            "document_type": "arbitration_award",
            "publish_time": "2016-07-12T00:00:00Z",
            "url": "https://pca-cpa.org/wp-content/uploads/sites/175/2016/07/PH-CN-20160712-Award.pdf",
            "viewpoint_origin": "legal_international",
            "entities": ["Permanent Court of Arbitration", "Philippines", "China", "UNCLOS",
                         "South China Sea", "nine-dash line", "Spratly Islands"],
            "geo_location": "The Hague, Netherlands",
            "license_or_terms": "PCA public document",
        },
        {
            "title": "PCA Press Release — South China Sea Arbitration Final Award",
            "caption": (
                "Official PCA press release (12 July 2016) announcing the Final Award, "
                "summarising the Tribunal's key findings on jurisdiction, the nine-dash "
                "line, the status of features under UNCLOS Article 121, China's conduct "
                "in the Philippines' EEZ, and the Tribunal's unanimous decision."
            ),
            "institution": "Permanent Court of Arbitration",
            "document_type": "official_statement",
            "publish_time": "2016-07-12T00:00:00Z",
            "url": "https://pca-cpa.org/en/news/pca-press-release-the-south-china-sea-arbitration-the-republic-of-the-philippines-v-the-peoples-republic-of-china/",
            "viewpoint_origin": "legal_international",
            "entities": ["Permanent Court of Arbitration", "Philippines", "China",
                         "UNCLOS", "South China Sea", "nine-dash line"],
            "geo_location": "The Hague, Netherlands",
            "license_or_terms": "PCA public document",
        },
        {
            "title": "US Department of State Statement on South China Sea Arbitration Award",
            "caption": (
                "Statement by US Secretary of State John Kerry (12 July 2016) welcoming "
                "the PCA Award as 'an important contribution to shared goals of a peaceful "
                "resolution', calling it 'final and legally binding', and urging China and "
                "the Philippines to comply — a key Western government response to the ruling."
            ),
            "institution": "US Department of State",
            "document_type": "official_statement",
            "publish_time": "2016-07-12T00:00:00Z",
            "url": "https://2009-2017.state.gov/secretary/remarks/2016/07/259741.htm",
            "viewpoint_origin": "western",
            "entities": ["United States", "John Kerry", "Philippines", "China",
                         "South China Sea", "Permanent Court of Arbitration"],
            "geo_location": "Washington D.C., United States",
            "license_or_terms": "US Government public domain",
            "redistribution_flag": True,
        },
    ],

    "kosovo": [
        {
            "title": "Kosovo Declaration of Independence (17 February 2008)",
            "caption": (
                "Declaration adopted by the Assembly of Kosovo on 17 February 2008, "
                "proclaiming Kosovo an independent, democratic, and sovereign state. "
                "The text commits Kosovo to the Ahtisaari Plan, to protection of minority "
                "communities, and to international obligations — the founding legal act "
                "of the Republic of Kosovo."
            ),
            "institution": "Assembly of Kosovo",
            "document_type": "declaration_text",
            "publish_time": "2008-02-17T00:00:00Z",
            "url": "https://www.assembly-kosova.org/common/docs/Dek_Pav_e.pdf",
            "viewpoint_origin": "kosovar",
            "entities": ["Kosovo", "Assembly of Kosovo", "Hashim Thaçi",
                         "Serbia", "Ahtisaari Plan", "UNMIK"],
            "geo_location": "Pristina, Kosovo",
            "license_or_terms": "Public official document",
        },
        {
            "title": "ICJ Advisory Opinion — Accordance with International Law of the Unilateral Declaration of Independence in Respect of Kosovo (2010)",
            "caption": (
                "Advisory opinion of the International Court of Justice (22 July 2010) "
                "concluding by 10 votes to 4 that Kosovo's 17 February 2008 declaration "
                "of independence did not violate general international law, UNSC Resolution "
                "1244, or the Constitutional Framework for Provisional Self-Government. "
                "The opinion found that the declaration's authors were not the Provisional "
                "Institutions of Self-Government and therefore not bound by those instruments."
            ),
            "institution": "International Court of Justice",
            "document_type": "legal_advisory",
            "publish_time": "2010-07-22T00:00:00Z",
            "url": "https://www.icj-cij.org/case/141",
            "viewpoint_origin": "legal_international",
            "entities": ["International Court of Justice", "Kosovo", "Serbia",
                         "UNSC Resolution 1244", "UNMIK", "Ahtisaari Plan"],
            "geo_location": "The Hague, Netherlands",
            "license_or_terms": "ICJ public document",
        },
        {
            "title": "UN Security Council Resolution 1244 (1999)",
            "caption": (
                "UNSC Resolution 1244 (10 June 1999) ended the Kosovo War and authorised "
                "an international civil and security presence in Kosovo (UNMIK and KFOR). "
                "It reaffirmed Yugoslav/Serbian sovereignty while demanding substantial "
                "autonomy — the foundational international-law document governing Kosovo's "
                "status until its 2008 declaration of independence."
            ),
            "institution": "UN Security Council",
            "document_type": "official_resolution",
            "publish_time": "1999-06-10T00:00:00Z",
            "url": "https://undocs.org/S/RES/1244(1999)",
            "viewpoint_origin": "multilateral",
            "entities": ["UN Security Council", "Kosovo", "Serbia", "Federal Republic of Yugoslavia",
                         "UNMIK", "KFOR", "NATO"],
            "geo_location": "New York, United States",
            "license_or_terms": "UN public document",
        },
    ],

    "libya": [
        {
            "title": "UN Security Council Resolution 1973 (2011)",
            "caption": (
                "UNSC Resolution 1973 (17 March 2011) authorised member states to take "
                "'all necessary measures' to protect civilians in Libya, established a "
                "no-fly zone, extended the arms embargo, and froze Libyan state assets. "
                "Adopted 10–0 with 5 abstentions (China, Russia, Germany, Brazil, India). "
                "This resolution was the legal basis for Operation Odyssey Dawn (US-led) "
                "and NATO's Operation Unified Protector."
            ),
            "institution": "UN Security Council",
            "document_type": "official_resolution",
            "publish_time": "2011-03-17T00:00:00Z",
            "url": "https://undocs.org/S/RES/1973(2011)",
            "viewpoint_origin": "multilateral",
            "entities": ["UN Security Council", "Libya", "Muammar Gaddafi", "NATO",
                         "Operation Odyssey Dawn", "Operation Unified Protector",
                         "Arab League", "Benghazi"],
            "geo_location": "New York, United States",
            "license_or_terms": "UN public document",
        },
        {
            "title": "UN Security Council Resolution 1970 (2011)",
            "caption": (
                "UNSC Resolution 1970 (26 February 2011) — adopted unanimously — imposed "
                "a comprehensive arms embargo on Libya, a travel ban and asset freeze on "
                "Muammar Gaddafi and associates, and referred the situation to the "
                "International Criminal Court. The first Chapter VII response to the 2011 "
                "Libya crisis and the predicate for Resolution 1973."
            ),
            "institution": "UN Security Council",
            "document_type": "official_resolution",
            "publish_time": "2011-02-26T00:00:00Z",
            "url": "https://undocs.org/S/RES/1970(2011)",
            "viewpoint_origin": "multilateral",
            "entities": ["UN Security Council", "Libya", "Muammar Gaddafi",
                         "International Criminal Court", "ICC", "Arab League"],
            "geo_location": "New York, United States",
            "license_or_terms": "UN public document",
        },
        {
            "title": "Arab League Resolution on Libya — Request for No-Fly Zone (March 2011)",
            "caption": (
                "Arab League Council resolution (12 March 2011) calling on the UN Security "
                "Council to impose a no-fly zone over Libya and establish safe areas to "
                "protect civilians, while simultaneously suspending Libya's participation "
                "in Arab League meetings. The Arab League's endorsement was the critical "
                "enabling condition that persuaded wavering UNSC members to support Res. 1973."
            ),
            "institution": "League of Arab States",
            "document_type": "official_resolution",
            "publish_time": "2011-03-12T00:00:00Z",
            "url": "https://www.securitycouncilreport.org/atf/cf/%7B65BFCF9B-6D27-4E9C-8CD3-CF6E4FF96FF9%7D/Libya%20Arab%20League%20Res%207360.pdf",
            "viewpoint_origin": "arab_regional",
            "entities": ["League of Arab States", "Libya", "Muammar Gaddafi",
                         "Arab League", "no-fly zone", "UN Security Council"],
            "geo_location": "Cairo, Egypt",
            "license_or_terms": "Public official document",
        },
    ],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

CATALOG_REQUIRED_FIELDS = {
    "event_id",
    "title",
    "caption",
    "institution",
    "document_type",
    "publish_time",
    "url",
    "viewpoint_origin",
    "entities",
    "geo_location",
    "license_or_terms",
}

CATALOG_EXTRA_FIELDS = (
    "asset_id",
    "review_status",
    "reviewer",
    "review_note",
    "review_state_note",
    "review_checked_at",
    "candidate_source",
    "candidate_query",
    "schema_version",
    "schema_change_note",
    "schema_change_note_v6",
    "curation_method",
    "document_language",
    "evidence_scope",
    "source_accessed_time",
    "document_group_id",
    "canonical_language",
    "current_url_language",
    "current_url_language_name",
    "current_asset_language",
    "current_asset_language_name",
    "language",
    "language_name",
    "language_native_name",
    "language_role",
    "authoritative_status",
    "available_language_codes",
    "available_language_names_by_language",
    "available_language_native_names_by_language",
    "available_language_urls_by_language",
    "available_languages_known_to_have",
    "co_authoritative_languages",
    "co_authoritative_language_names",
    "co_authoritative_language_native_names",
    "primary_text_language",
    "primary_text_language_name",
    "primary_text_language_native",
    "primary_text_language_type",
    "primary_text_language_policy",
    "primary_text_language_policy_note",
    "preferred_primary_text_download_language",
    "preferred_primary_text_download_language_name",
    "representative_download_language",
    "representative_download_language_name",
    "is_primary_text_language",
    "is_representative_download_language",
    "download_by_default",
    "download_priority",
    "counts_as_extra_official_evidence",
    "evidence_count_weight",
    "enters_claim_evidence_hyperedge",
    "benchmark_layer",
    "benchmark_language_policy_note",
    "document_language_model",
    "document_group_activation_policy",
    "language_variant_layer_policy",
    "language_variant_policy_note_v6",
    "active_policy",
    "active_policy_taxonomy_version",
    "active_bench",
    "download_recommendation_language",
    "translation_provenance",
    "language_audit_status",
    "language_audit_note",
    "language_audit_version",
    "url_canonical",
    "url_resolver",
    "known_partial_language_shells",
    "language_variants",
    "language_role_taxonomy_version",
    "authoritative_status_taxonomy_version",
)

def _asset_id(event_id: str, url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"asset_official_{event_id}_{h}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: expected object")
            rows.append(row)
    return rows


def validate_catalog_row(row: dict, path: Path, line_no: int) -> None:
    missing = sorted(field for field in CATALOG_REQUIRED_FIELDS if field not in row or row[field] in ("", None))
    if missing:
        raise ValueError(f"Official doc catalog row {path}:{line_no} missing {missing}")
    event_id = str(row["event_id"])
    if event_id not in EVENTS:
        raise ValueError(f"Official doc catalog row {path}:{line_no} has unknown event_id={event_id}")
    if not isinstance(row["entities"], list) or not all(isinstance(entity, str) for entity in row["entities"]):
        raise ValueError(f"Official doc catalog row {path}:{line_no} entities must be list[str]")


def load_review_state(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    root = payload.get("review_state", payload) if isinstance(payload, dict) else {}
    records = root.get("records", {}) if isinstance(root, dict) else {}
    if not isinstance(records, dict):
        return {}
    return {str(key): value for key, value in records.items() if isinstance(value, dict)}


def apply_review_state(row: dict, review_records: dict[str, dict] | None) -> dict:
    if not review_records:
        return row
    asset_id = row.get("asset_id")
    if not asset_id:
        return row
    state = review_records.get(str(asset_id))
    if not state:
        return row
    out = dict(row)
    if state.get("status"):
        out["review_status"] = str(state["status"]).lower()
    if state.get("note"):
        out["review_state_note"] = state["note"]
    if state.get("checked_at"):
        out["review_checked_at"] = state["checked_at"]
    return out


def is_false(value) -> bool:
    return value is False or (isinstance(value, str) and value.lower() == "false")


def is_wiki_url(value) -> bool:
    if not isinstance(value, str) or not value:
        return False
    host = urlparse(value).netloc.lower()
    return "wikipedia.org" in host or "wikimedia.org" in host


def variant_language(value: dict) -> str | None:
    for key in ("language", "url_language", "url_language_code", "current_url_language", "link_language"):
        raw = value.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return None


def sanitize_official_doc_metadata(doc: dict) -> dict:
    clean = dict(doc)
    allowed_languages: set[str] = set()
    saw_language_variants = False
    for key in ("available_languages_known_to_have", "language_variants"):
        values = clean.get(key)
        if not isinstance(values, list):
            continue
        saw_language_variants = True
        kept = []
        for item in values:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            if is_wiki_url(item.get("url")) or is_wiki_url(item.get("url_canonical")):
                continue
            kept.append(item)
            language = variant_language(item)
            if language:
                allowed_languages.add(language)
        clean[key] = kept
    urls_by_language = clean.get("available_language_urls_by_language")
    if isinstance(urls_by_language, dict):
        filtered_urls = {language: url for language, url in urls_by_language.items() if not is_wiki_url(url)}
        clean["available_language_urls_by_language"] = filtered_urls
        allowed_languages.update(str(language) for language in filtered_urls)
    for key in ("available_language_names_by_language", "available_language_native_names_by_language"):
        values = clean.get(key)
        if isinstance(values, dict) and allowed_languages:
            clean[key] = {language: label for language, label in values.items() if str(language) in allowed_languages}
    codes = clean.get("available_language_codes")
    if isinstance(codes, list) and (allowed_languages or saw_language_variants):
        clean["available_language_codes"] = [code for code in codes if str(code) in allowed_languages]
    if str(clean.get("translation_provenance", "")).lower().startswith("wikipedia"):
        clean.pop("translation_provenance", None)
    return clean


def load_official_doc_catalog(path: Path, review_records: dict[str, dict] | None = None) -> tuple[dict[str, list[dict]], dict[str, int]]:
    docs_by_event: dict[str, list[dict]] = {}
    stats = {"approved": 0, "skipped": 0}
    rows = _read_jsonl(path)
    for idx, row in enumerate(rows, 1):
        row = apply_review_state(row, review_records)
        review_status = str(row.get("review_status", "pending")).lower()
        if review_status != "approved":
            stats["skipped"] += 1
            continue
        if is_false(row.get("active_bench")):
            stats["skipped"] += 1
            continue
        row = sanitize_official_doc_metadata(row)
        validate_catalog_row(row, path, idx)
        event_id = str(row["event_id"])
        doc = {key: value for key, value in row.items() if key != "event_id"}
        doc["review_status"] = review_status
        docs_by_event.setdefault(event_id, []).append(doc)
        stats["approved"] += 1
    return docs_by_event, stats


def _build_asset(event_id: str, doc: dict, retrieval_time: str) -> dict:
    url = doc["url"]
    asset_id = _asset_id(event_id, url)
    provenance_hash = hashlib.sha256(
        "\n".join([event_id, doc["title"], doc["caption"], doc["publish_time"], url]).encode("utf-8", errors="replace")
    ).hexdigest()
    extra = {
        "collection_channel": "official_registry",
        "record_type": doc["document_type"],
        "curation_level": "official",
        "source_temporal_coverage": _temporal_coverage(event_id, doc["publish_time"]),
        "active_policy": "primary_official_evidence",
        "title": doc["title"],
        "institution": doc["institution"],
        "active_bench": True,
        "active_status": "active",
        "retrieval_time": retrieval_time,
    }
    for key in CATALOG_EXTRA_FIELDS:
        if key in doc and doc[key] not in ("", None):
            if key == "active_policy":
                extra["catalog_active_policy"] = doc[key]
            elif key == "active_bench":
                extra["catalog_active_bench"] = doc[key]
            else:
                extra[key] = doc[key]
    return {
        "asset_id": asset_id,
        "event_id": event_id,
        "modality": "structured_document",
        "asset_source": "OFFICIAL_DOC",
        "source_layer": "official",
        "viewpoint_origin": doc["viewpoint_origin"],
        "publish_time": doc["publish_time"],
        "observed_time": doc["publish_time"],
        "geo_location": doc["geo_location"],
        "url_or_pointer": url,
        "caption_or_transcript": doc["caption"],
        "license_or_terms": doc["license_or_terms"],
        "redistribution_flag": doc.get("redistribution_flag", False),
        "perceptual_hash": provenance_hash,
        "embedding_id": f"emb_{asset_id}",
        "extracted_entities": doc["entities"],
        "extracted_claims": [],
        "evidence_role": "substantive",
        "extra": extra,
    }


# Event window upper bounds (inclusive) for temporal classification.
# Near or contemporaneous = within ~6 months of the event anchor date.
_EVENT_WINDOWS: dict[str, tuple[str, str]] = {
    "jcpoa":    ("2015-01-01", "2016-12-31"),
    "scs":      ("2016-01-01", "2017-06-30"),
    "kosovo":   ("2008-01-01", "2010-12-31"),
    "libya":    ("2011-01-01", "2012-06-30"),
}


def _temporal_coverage(event_id: str, publish_time: str) -> str:
    pub_date = publish_time[:10]
    window = _EVENT_WINDOWS.get(event_id)
    if window and window[0] <= pub_date <= window[1]:
        return "near_event_window"
    if window and pub_date < window[0]:
        return "historical_background"
    return "later_context"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        choices=sorted(EVENTS),
        default=None,
        help="Generate docs for one event (omit for all).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Optional JSONL candidate catalog. Only rows with review_status=approved become assets.",
    )
    parser.add_argument(
        "--review-state",
        type=Path,
        default=None,
        help="Optional review-state JSON exported by the link review UI. Overrides row review_status by asset_id.",
    )
    parser.add_argument(
        "--no-builtin-catalog",
        action="store_true",
        help="Use only --catalog rows instead of merging with the built-in official document catalog.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "0_external" / "external_asset_raw",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_time = _now_iso()
    docs_by_event: dict[str, list[dict]] = {}
    if not args.no_builtin_catalog:
        docs_by_event = {event_id: list(docs) for event_id, docs in OFFICIAL_DOCS.items()}
    if args.catalog:
        review_records = load_review_state(args.review_state) if args.review_state else None
        catalog_docs, stats = load_official_doc_catalog(args.catalog, review_records)
        for event_id, docs in catalog_docs.items():
            docs_by_event.setdefault(event_id, []).extend(docs)
        print(f"loaded official catalog {args.catalog}: approved={stats['approved']} skipped={stats['skipped']}")
    events = [args.event] if args.event else sorted(docs_by_event)

    for event_id in events:
        docs = docs_by_event.get(event_id, [])
        if not docs:
            print(f"no official doc assets for {event_id}")
            continue
        records = [_build_asset(event_id, doc, retrieval_time) for doc in docs]
        out_path = args.output_dir / f"official_{event_id}.jsonl"

        if args.dry_run:
            print(f"\n── {event_id} ({len(records)} docs) → {out_path} ──")
            for r in records:
                print(json.dumps(r, ensure_ascii=False, indent=2))
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} official doc assets → {out_path}")


if __name__ == "__main__":
    main()
