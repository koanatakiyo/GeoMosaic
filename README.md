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
- `GDELTDOCClient`: public GDELT DOC 2.0 full-text news search. Rows are stored as news/article pointers, not curated structured-event rows.

Convenience CLIs:

```bash
python3 script/fetch_wikimedia_assets.py --event crimea --query "Crimea map" --limit 5
ACLED_USERNAME=... ACLED_PASSWORD=... python3 script/fetch_acled_assets.py --event ukraine --country Ukraine --start-date 2022-02-24 --end-date 2022-02-28
python3 script/fetch_gdelt_doc_assets.py --event hongkong --max-records 25
python3 script/collect_external_assets.py --collect-existing data/0_external/external_asset_raw
```

By default these commands write raw `EvidenceAsset` JSONL files to `data/0_external/external_asset_raw/`. `script/collect_external_assets.py` merges those raw files into `data/0_external/external_assets.jsonl`, writes `candidate_inventory.jsonl` and `selection_decisions.jsonl`, and keeps only active benchmark assets in the merged file. Smoke and full benchmark builds load `data/0_external/external_assets.jsonl` when present.

Official-document assets are text/pointer records in the default benchmark. Embedded media on official webpages, such as spokesperson photographs, venue images, symbolic graphics, or speech videos, are not downloaded or redistributed in R1. Their evidentiary content is represented by the official text, transcript, legal document, or press release. The active image layer is built from page-bound Wikimedia assets and separately curated map-like records.

Official multilingual metadata keeps language variants under the same document group. In the reviewed catalog, `zh` and `zh-Hans-CN` denote Simplified Chinese, `zh-Hant-HK` denotes Traditional Chinese for Hong Kong legal materials, and `uk` denotes Ukrainian. The default benchmark activates one representative official-document variant per group when possible; other language variants remain metadata-only unless a language-aware evaluation explicitly enables them.

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
