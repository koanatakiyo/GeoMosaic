# GeoMosaic-HG

Project-local prototype for **GeoMosaic-HG: Source-Conditioned Multimodal Evidence Hypergraphs with Provenance-Aware Indexing for Geopolitical Retrieval**.

All default paths resolve inside this repository:

- code: `script/`
- copied/local data: `data/`
- generated benchmark tables: `data/geomosaic_bench/`
- reports: `data/reports/`

## Quickstart

```bash
python3 script/export_json_schemas.py
python3 script/build_smoke_bench.py --event crimea
python3 script/build_geomosaic_bench.py
python3 script/validate_geomosaic_tables.py
python3 script/run_retrieval.py --query "Crimea territorial integrity referendum" --budget 8 --cutoff 2026-01-01T00:00:00Z
python3 script/evaluate_methods.py --query "Ukraine sovereignty territorial integrity" --budget 10 --cutoff 2026-01-01T00:00:00Z
```

## Core Tables

`script/build_geomosaic_bench.py` writes the four plan tables:

- `source_records.jsonl`
- `evidence_assets.jsonl`
- `source_asset_links.jsonl`
- `claim_evidence_hyperedges.jsonl`

The current offline build uses copied `data/0_raw` text/PDF files plus copied claim-audit outputs under `data/3_direct_scores`. It creates redistributable text assets where appropriate, restricted news PDF pointer assets, map pointer assets, and structured event metadata assets.

One-event smoke build:

```bash
python3 script/build_smoke_bench.py --event crimea
```

This writes valid JSONL to `data/smoke_bench/crimea/` and runs schema, provenance, referential-integrity, and path-locality checks.

## External Clients

Client skeletons live under `script/geomosaic_hg/clients/`:

- `WikimediaCommonsClient`: MediaWiki Action API search and `imageinfo` metadata for Wikimedia Commons files.
- `ACLEDClient`: OAuth token flow and `/api/acled/read` queries. Credentials come from `ACLED_USERNAME` or `ACLED_EMAIL`, plus `ACLED_PASSWORD`.
- `GDELTDOCClient`: public GDELT DOC 2.0 full-text news search. Rows are stored as news/article pointers plus restricted Visual GKG/social-image pointers, not curated structured-event rows.

Convenience CLIs:

```bash
python3 script/fetch_wikimedia_assets.py --event crimea --query "Crimea map" --limit 5
ACLED_USERNAME=... ACLED_PASSWORD=... python3 script/fetch_acled_assets.py --event ukraine --country Ukraine --start-date 2022-02-24 --end-date 2022-02-28
python3 script/fetch_gdelt_doc_assets.py --event hongkong --max-records 25
python3 script/collect_gdelt_doc_pointers.sh --all-events --max-records 25
python3 script/collect_external_assets.py --collect-existing data/0_external/external_asset_raw
```

By default these commands write raw `EvidenceAsset` JSONL files to `data/0_external/external_asset_raw/`. `script/collect_external_assets.py` merges those raw files into `data/0_external/external_assets.jsonl`, writes `candidate_inventory.jsonl` and `selection_decisions.jsonl`, and keeps only active benchmark assets in the merged file. Smoke and full benchmark builds load `data/0_external/external_assets.jsonl` when present.

GDELT DOC is a pointer adapter only. Article rows use `extra.record_type = "news_pointer"` with top-level `modality = "text"` for schema compatibility. Visual GKG/social-image rows use `modality = "image_restricted_pointer"`, `extra.collection_channel = "gdelt_doc_visual_gkg"`, and `redistribution_flag = false`. Default GDELT DOC collection uses each event's anchor-date window and marks rows `source_temporal_coverage = "event_window"`; ad hoc rolling `timespan` probes are diagnostic only and should not replace event-window benchmark evidence.

Official-document assets are text/pointer records in the default benchmark. Embedded media on official webpages, such as spokesperson photographs, venue images, symbolic graphics, or speech videos, are not downloaded or redistributed in R1. Their evidentiary content is represented by the official text, transcript, legal document, or press release. The active image layer is built from page-bound Wikimedia assets and separately curated map-like records.

Official multilingual metadata keeps language variants under the same document group. In the reviewed catalog, `zh` and `zh-Hans-CN` denote Simplified Chinese, `zh-Hant-HK` denotes Traditional Chinese for Hong Kong legal materials, and `uk` denotes Ukrainian. The default benchmark activates one representative official-document variant per group when possible; other language variants remain metadata-only unless a language-aware evaluation explicitly enables them.

Manual official-document materialization is parsed deterministically before any LLM enrichment. The parser reads only `data/0_external/official_doc_materialized/manifest.jsonl`, so files outside the manifest are ignored even if present on disk.

```bash
python3 script/organize_manual_materialized_docs.py
python3 script/parse_official_docs.py \
  --manifest data/0_external/official_doc_materialized/manifest.jsonl \
  --output-dir data/0_external/official_doc_parsed
python3 script/qa_official_doc_parse.py \
  --parsed-dir data/0_external/official_doc_parsed \
  --output data/0_external/official_doc_parsed/parse_qa_report.json
python3 script/plan_metadata_extraction.py \
  --parsed-dir data/0_external/official_doc_parsed \
  --external-assets data/0_external/external_assets.jsonl \
  --output-dir data/1_intermediate/metadata_extraction \
  --model-id gemini-2.5-flash \
  --dry-run
```

`script/parse_official_docs.py` writes `official_doc_text.jsonl`, `passages.jsonl`, and `parse_summary.json`; `script/qa_official_doc_parse.py` writes `parse_qa_report.json`. Text files are read directly; PDFs are parsed with `pdfminer.six`. Parsed document rows use top-level `sha256` for the source file hash, while parser diagnostics such as `text_sha256` and `page_spans` live under `extra`. Passage IDs include the document id, language, and passage index; each passage also carries `page_start` and `page_end` for citation/debugging. QA revalidates source-file hashes, passage ID format, char spans, passage text slices, and page ranges derived from `extra.page_spans`. This layer is deterministic and model-independent.

`script/plan_metadata_extraction.py` performs the Step 3 dry run only. It writes `metadata_extraction_tasks.jsonl` and `metadata_extraction_dry_run_summary.json`, plans non-load-bearing official-document metadata fields, and makes zero LLM calls. Pointer assets such as GDELT DOC `news_pointer`, Visual GKG `image_restricted_pointer`, and `map_pointer` records are counted as skipped because pointers do not need text extraction.

`script/extract_official_doc_metadata.py` executes Stage B against the planned tasks through Vertex AI Gemini. It writes a sidecar JSONL, not benchmark labels. Every output row is marked `load_bearing=false`, `stage_c_excluded=true`, and `stage_c_policy=never_use_stage_b_outputs`.

Trial run:

```bash
python3 script/extract_official_doc_metadata.py \
  --limit 3 \
  --overwrite \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --sleep-seconds 1
```

Full resumable run:

```bash
python3 script/extract_official_doc_metadata.py \
  --model-id gemini-2.5-flash \
  --resume \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --sleep-seconds 1
```

Default Stage B outputs:

- `data/1_intermediate/metadata_extraction/official_doc_metadata_enrichment.jsonl`
- `data/1_intermediate/metadata_extraction/official_doc_metadata_batch_outputs.jsonl`
- `data/1_intermediate/metadata_extraction/official_doc_metadata_enrichment_summary.json`

Frozen LLM sidecar plan:

- Stage B: Gemini 2.5 Flash may enrich parsed official documents with non-load-bearing fields only: summary, actors, dates, language note, section outline, and candidate passage hints.
- Stage B outputs never enter claim scoring, diagnostic triage, or main metrics, and never define benchmark ground truth.
- Stage C: fixed-claim plus top-k-passage judgments are retained only as diagnostic triage for passage quality, retrieval failure, model disagreement, and language retrieval issues. They are not gold labels and do not support the main experimental claims.
- E4: secondary LLM validation follows the GeoGround direct-scoring protocol over document/evidence bundles. Long documents may use document-level map-reduce scoring, but the unit of measurement is still the bundle/document, not claim-conditioned top-k passages.
- ClaimEvidenceHyperedge rows are candidate/provenance structures unless explicitly human-verified. They express possible claim-evidence incidence for retrieval and audit, not verified support/contradict ground truth.
- Main ICDE experiments remain SMPI/BPE retrieval, deterministic coverage/constraint metrics, and system-efficiency evaluation.

## Retrieval

`script/run_retrieval.py` loads the project-local tables into SMPI, applies temporal/primary-source/modality/primary-match-level/evidence-role/provenance filters, retrieves lexical seeds, expands through event/source/asset incidence lists, then runs Balanced Provenance Expansion over the capped coverage objective.

`script/evaluate_methods.py` compares:

- GeoMosaic-HG BPE
- NaiveRAG
- Metadata++
- Metadata++ + MMR
- Random-SM

## Synthetic Stress

```bash
python3 script/make_synthetic_stress.py --blocks 10
```

Synthetic block replication is only for index build time, latency, and pruning-ratio stress tests. Do not use replicated tiers for quality metrics.
