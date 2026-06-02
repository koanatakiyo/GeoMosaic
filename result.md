# GeoMosaic-HG Results

## Smoke build Local Acceptance

Date: 2026-05-22

Environment:

```bash
conda activate geopolitic
cd /home/yandan/geopo/GeoMosaic
```

### Commands Run

```bash
python script/export_json_schemas.py
python script/build_smoke_bench.py --event crimea
python script/validate_geomosaic_tables.py --bench-dir data/smoke_bench/crimea

python script/run_retrieval.py \
  --bench-dir data/smoke_bench/crimea \
  --query "Crimea territorial integrity referendum" \
  --budget 5 \
  --cutoff 2026-01-01T00:00:00Z

python script/run_retrieval.py \
  --bench-dir data/smoke_bench/crimea \
  --query "Crimea territorial integrity referendum" \
  --source-layers official \
  --budget 3 \
  --cutoff 2026-01-01T00:00:00Z

python script/run_retrieval.py \
  --bench-dir data/smoke_bench/crimea \
  --query "Crimea territorial integrity referendum" \
  --source-layers news \
  --budget 3 \
  --cutoff 2026-01-01T00:00:00Z

python script/run_retrieval.py \
  --bench-dir data/smoke_bench/crimea \
  --query "Crimea territorial integrity referendum" \
  --max-match-level L3 \
  --budget 3 \
  --cutoff 2026-01-01T00:00:00Z

python script/evaluate_methods.py \
  --bench-dir data/smoke_bench/crimea \
  --query "Crimea territorial integrity referendum" \
  --budget 5 \
  --cutoff 2026-01-01T00:00:00Z \
  --output-json data/reports/smoke_crimea_method_comparison.json \
  --output-csv data/reports/smoke_crimea_method_comparison.csv

pytest -q
```

### Build And Validation

Schema export generated:

- `data/schema/source_records.schema.json`
- `data/schema/evidence_assets.schema.json`
- `data/schema/source_asset_links.schema.json`
- `data/schema/claim_evidence_hyperedges.schema.json`

Smoke build smoke build:

| Table | Count |
| --- | ---: |
| source_records | 16 |
| evidence_assets | 27 |
| source_asset_links | 192 |
| claim_evidence_hyperedges | 350 |

Validation:

```text
ok: true
errors: []
```

Unit tests:

```text
pytest -q
1 passed in 0.70s
```

### Retrieval Checks

| Retrieval mode | candidate_count | selected | viewpoint_coverage | viewpoint_balance | source_diversity | layer_diversity | modality_coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Crimea retrieval | 350 | 5 | 1.0 | 0.916516 | 4 | 4 | 4 |
| Primary source = official | 125 | 3 | 1.0 | 1.0 | 3 | 3 | 3 |
| Primary source = news | 225 | 3 | 0.6 | 0.682606 | 2 | 2 | 3 |
| Max primary match level = L3 | 350 | 3 | 1.0 | 1.0 | 3 | 4 | 4 |

Key acceptance checks:

- Constrained source-layer retrieval no longer returns zero candidates.
- Match-level filtering no longer gets blocked by map/context `L4` assets.
- Provenance completeness is `1.0`.
- Temporal leakage rate is `0.0`.
- Smoke build generated paths remain project-local.

### Method Comparison

Budget: 5  
Event: Crimea  
Query: `Crimea territorial integrity referendum`

| Method | viewpoint_coverage | viewpoint_balance | source_diversity | layer_diversity | expanded_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeoMosaic-HG BPE | 1.0 | 0.916516 | 4 | 4 | 350 |
| Metadata++ + MMR | 1.0 | 0.969724 | 4 | 4 | 100 |
| NaiveRAG | 0.6 | 0.59381 | 2 | 3 | 100 |
| Metadata++ | 0.6 | 0.59381 | 2 | 3 | 100 |
| Random-SM | 0.6 | 0.59381 | 2 | 4 | 100 |

Notes:

- `objective_value: null` for baselines is expected because baselines do not run the BPE optimization objective.
- Metadata++ + MMR gets slightly higher viewpoint balance on this single metric, but it operates on the ANN seed set only.
- GeoMosaic-HG BPE expands from 100 seeds to all 350 feasible hyperedges before greedy capped-coverage selection.
- The proper joint BPE score for the full Smoke build run is `objective_value = 15.113625`.

### Smoke build Verdict

Smoke build is accepted locally.

The full local acceptance chain passed:

- schema export
- one-event smoke build
- table validation
- unconstrained retrieval
- primary source constrained retrieval
- match-level constrained retrieval
- method comparison report generation
- unit tests

### Known External enrichment Follow-Up

The news-only retrieval mode has weaker viewpoint coverage on Crimea:

```text
viewpoint_coverage = 0.6
viewpoint_balance = 0.682606
```

This is a corpus coverage limitation rather than a pipeline failure. External enrichment should enrich Tier 1 external assets, especially Wikimedia maps/images and ACLED structured rows for all 8 GeoGround events.

## External enrichment Target

From the execution plan:

- Run Wikimedia pipeline for all 8 GeoGround events.
- Run ACLED pipeline for all 8 GeoGround events.
- Deliverable: Wikimedia images/maps plus ACLED rows in EvidenceAsset JSONL.

Implemented External enrichment entry point:

```bash
python script/collect_external_assets.py --dry-run
```

Default network collection:

```bash
python script/collect_external_assets.py
```

Merge existing External enrichment JSONL without network:

```bash
python script/collect_external_assets.py \
  --collect-existing data/0_external/external_asset_raw \
  --merged-output data/0_external/external_assets.jsonl
```

The merged output is placed under `data/0_external/`, where the build pipeline can load it automatically.

### External enrichment Implementation Status

Added code:

- `script/geomosaic_hg/external_assets.py`
- `script/collect_external_assets.py`
- `tests/test_external_assets_pipeline.py`

Local verification:

```text
pytest -q
20 passed in 1.31s

python -m compileall -q script tests
ok

python script/collect_external_assets.py --dry-run
ok
```

The dry-run currently plans all 8 Tier-1 events: `crimea`, `iraq`, `libya`, `kosovo`, `scs`, `jcpoa`, `ukraine`, and `hongkong`. South China Sea has no single ACLED country hint yet, so its ACLED collection is intentionally skipped until the event-specific country/window rule is refined.

## External enrichment External Asset Collection

Command sequence:

```bash
python script/collect_external_assets.py \
  --acled-env-file APIs/ACLED_api \
  --acled-window-days 7 \
  --acled-limit 200

python script/collect_external_assets.py \
  --events kosovo,scs,jcpoa,ukraine,hongkong \
  --skip-acled \
  --image-limit 1 \
  --map-limit 1 \
  --wikimedia-delay-seconds 10 \
  --wikimedia-retries 2 \
  --wikimedia-retry-backoff-seconds 120

python script/collect_external_assets.py \
  --collect-existing data/0_external/external_asset_raw \
  --acled-limit 200 \
  --acled-window-days 7

python script/collect_external_assets.py \
  --skip-acled \
  --image-limit 3 \
  --map-limit 1 \
  --acled-limit 1000 \
  --acled-window-days 7 \
  --wikimedia-delay-seconds 8 \
  --wikimedia-retries 4 \
  --wikimedia-retry-backoff-seconds 60 \
  --download-wikimedia-images \
  --image-dir data/0_external/event_images \
  --download-timeout 45 \
  --download-retries 3 \
  --download-retry-backoff-seconds 60

python script/collect_external_assets.py \
  --events hongkong,iraq \
  --skip-acled \
  --image-limit 10 \
  --map-limit 2 \
  --acled-limit 1000 \
  --acled-window-days 7 \
  --wikimedia-delay-seconds 6 \
  --wikimedia-retries 4 \
  --wikimedia-retry-backoff-seconds 60 \
  --download-wikimedia-images \
  --image-dir data/0_external/event_images \
  --download-timeout 45 \
  --download-retries 2 \
  --download-retry-backoff-seconds 45
```

Final merged output:

```text
data/0_external/external_assets.jsonl
```

Final External enrichment counts after switching Wikimedia collection from broad Commons search to event-specific Wikipedia page images and separating candidates from active benchmark assets:

Candidate inventory:

| Event | External assets |
| --- | ---: |
| crimea | 12 |
| hongkong | 149 |
| iraq | 10 |
| jcpoa | 13 |
| kosovo | 9 |
| libya | 86 |
| scs | 7 |
| ukraine | 1567 |

| Source | Count |
| --- | ---: |
| ACLED | 1768 |
| OFFICIAL_DOC | 12 |
| Wikimedia Commons | 73 |

| Modality | Count |
| --- | ---: |
| image_full | 65 |
| map_pointer | 8 |
| structured_document | 12 |
| structured_event | 1768 |

Total candidate external assets: `1853` across all 8 events.

Active benchmark assets in `data/0_external/external_assets.jsonl`:

| Event | Active external assets |
| --- | ---: |
| crimea | 6 |
| hongkong | 144 |
| iraq | 5 |
| jcpoa | 8 |
| kosovo | 8 |
| libya | 82 |
| scs | 6 |
| ukraine | 1561 |

| Source | Count |
| --- | ---: |
| ACLED | 1768 |
| OFFICIAL_DOC | 12 |
| Wikimedia Commons | 40 |

| Modality | Count |
| --- | ---: |
| image_full | 34 |
| map_pointer | 6 |
| structured_document | 12 |
| structured_event | 1768 |

Total active external assets: `1820` across all 8 events.

Downloaded Wikimedia image files:

```text
data/0_external/event_images/
```

The merged JSONL now points active Wikimedia `url_or_pointer` values to project-local files and preserves the remote Wikimedia URL in `extra.original_url`. The image cache has a review manifest at `data/0_external/event_images/manifest.jsonl` with 73 downloaded candidate rows: 40 active benchmark images/maps and 33 candidate-only images/maps. The older `data/0_external/obsolete_commons_search_images/` folder came from broad Commons search and should be treated as obsolete; `obsolete_event_images_refresh/` is superseded by the higher-limit refresh for Hong Kong and Iraq.

### Wikimedia Page-Bound Policy

External enrichment now separates file provenance from evidence discovery:

- `asset_source = "Wikimedia Commons"` means the file, license metadata, and redistributable image URL come from Commons.
- `extra.collection_channel = "wikipedia_page_bound"` means the asset was discovered because it is embedded in the event's Wikipedia article.
- Commons keyword/category search is treated as fallback only. It is not mixed into the active External enrichment benchmark unless explicitly promoted.
- `extra.record_type` distinguishes `wiki_page_asset`, `curated_conflict_event`, `official_resolution`, `arbitration_award`, `agreement_text`, and future adapter rows.
- `extra.curation_level` distinguishes community-curated Wikimedia metadata, human-curated ACLED records, and official/legal documents.
- `extra.active_policy` records whether a row is primary image evidence, optional enrichment, or fallback candidate material.

Schema policy:

- Page-bound Wikipedia images are linked to their corresponding Wiki `SourceRecord` with `match_level = "L1"`, not `L0`.
- Page-bound fallback images are not linked as `L1` unless the corresponding fallback page text is also present as a `SourceRecord`. They remain event-level `L3`/`L4` context.
- `L0` remains reserved for text assets that directly represent the source record itself.
- Fine-grained visual roles are stored in `extra.proposed_role`; the top-level `evidence_role` remains schema-safe (`complementary` or `map_like`) until a later schema migration.
- Official curated documents use `modality = "structured_document"`, `asset_source = "OFFICIAL_DOC"`, and `active_policy = "primary_official_evidence"`. They are selected into official-source hyperedges as `L2` same-layer official evidence when applicable.

Example active Wikimedia asset metadata:

```json
{
  "asset_source": "Wikimedia Commons",
  "source_layer": "wiki",
  "evidence_role": "complementary",
  "extra": {
    "collection_channel": "wikipedia_page_bound",
    "record_type": "wiki_page_asset",
    "curation_level": "community_curated",
    "active_policy": "primary_image_evidence",
    "page_bound": true,
    "active_bench": true,
    "source_page_title": "2008 Kosovo declaration of independence",
    "file_title": "File:Kosova independence Vienna 17-02-2008 b.jpg",
    "proposed_role": "substantive_event_image",
    "temporal_status": "near_event_window",
    "original_url": "https://upload.wikimedia.org/..."
  }
}
```

Current active Wikimedia curation summary:

| Field | Distribution |
| --- | --- |
| `extra.collection_channel` | `wikipedia_page_bound`: 40 |
| `extra.record_type` | `wiki_page_asset`: 40 |
| `extra.proposed_role` | `substantive_event_image`: 21; `official_context`: 8; `map_like`: 6; `protest_context`: 4; `background`: 1 |
| `extra.temporal_status` | `near_event_window`: 22; `later_context`: 7; `historical_background`: 6; `dynamic_updated_map`: 3; `contemporaneous`: 2 |
| local detected file extension | `.jpg`: 31; `.png`: 8; `.gif`: 1 |

Manifest completeness check:

| Field | Missing rows |
| --- | ---: |
| `caption` | 0 |
| `license_or_terms` | 0 |
| `source_page_title` | 0 |
| `source_page_revision_id` | 0 |
| `section_anchor` | 73 |

Current caption caveat: `caption` is present for all 73 downloaded candidate rows, but it is still derived from Wikimedia file metadata/title rather than a parsed article-local figure caption. Article section extraction remains a follow-up before claim-level image grounding.

Page-bound link audit in `data/enriched_full_bench/source_asset_links.jsonl`:

| Link type | Count |
| --- | ---: |
| Wiki page-bound media to the matching Wiki source record, `L1` | 34 |
| Other Wikimedia contextual links, `L3` | 485 |
| Other Wikimedia background/map links, `L4` | 87 |

Notes:

- The first Wikimedia pass hit Commons rate limits (`HTTP 429`) for several events.
- The second pass used slower Wikimedia settings and completed with `errors: []`.
- Wikimedia rows are now sourced from event Wikipedia pages, not generic Commons keyword search. This removed the earlier wrong images such as Crimea landscape photos, Hong Kong waterfront photos, and Ukraine train photos.
- For `scs`, the primary page remains `South China Sea Arbitration`; `Territorial disputes in the South China Sea` is added only as a Wikipedia page-bound fallback/background page for territorial-claims maps and long-running dispute context.
- Wikimedia downloads use the API-provided thumbnail URL when available and keep the original upload URL in metadata.
- Candidate inventory and active benchmark assets are now physically separate. `data/0_external/candidate_inventory.jsonl` keeps all 1853 candidate rows, `data/0_external/selection_decisions.jsonl` records why each row is active or candidate-only, and `data/0_external/external_assets.jsonl` contains only the 1820 active benchmark rows.
- Official document coverage adds 12 active structured-document rows: JCPOA, SCS, Kosovo, and Libya each have 3 curated official/legal records.
- External enrichment summary now includes explicit `warnings` for ACLED zero-row/skip coverage notes.
- Event-scoped network fetches now merge all existing raw External enrichment files by default, so a follow-up `--events ukraine` run no longer silently drops the other seven events from the default merged output.

### Official Document Multimedia Policy

Official-document rows are text/pointer evidence in the default R1 benchmark. We do not crawl or redistribute embedded images or videos from official webpages. Most official page media are spokesperson photographs, venue images, symbolic graphics, or speech videos whose evidentiary content is already represented by the official text, transcript, legal document, or press release. Downloading them would add copyright, size, format, and reproducibility risk without improving the core official/legal evidence layer.

The active visual layer is therefore constructed from page-bound Wikimedia assets and curated map-like records, while official-document media are excluded from the default benchmark unless separately curated as an independent asset. If an official page contains a map, diagram, or other visual that is independently evidentiary, it may be recorded as a future candidate, but it is not automatically downloaded or activated.

Paper wording:

```text
For official documents, we do not crawl embedded media in R1. Most official
webpage media are spokesperson photographs, venue images, symbolic graphics, or
speech videos whose evidentiary content is already represented by the official
text, transcript, legal document, or press release. To avoid redistribution and
reproducibility risks, official media are excluded from the default benchmark.
The image layer is instead constructed from page-bound Wikimedia assets and
curated map-like records.
```

Current metadata policy:

```json
{
  "multimedia_policy": "not_collected_r1",
  "multimedia_note": "Official page media were not downloaded; text/transcript/legal document content is used as the evidence record."
}
```

### Official Document Parsing Protocol

Manual official-document materialization is treated as a reviewed input layer, not as a crawler output. Files are organized into `data/0_external/official_doc_materialized/` and parsed only through its manifest:

```text
data/0_external/official_doc_materialized/
  files/{event}/
  manual_text/{event}/
  manifest.jsonl
  summary.json
```

The parser is deterministic and model-independent:

```bash
python script/parse_official_docs.py \
  --manifest data/0_external/official_doc_materialized/manifest.jsonl \
  --output-dir data/0_external/official_doc_parsed \
  --max-passage-chars 1800 \
  --overlap-chars 150 \
  --min-ok-chars 100
```

Output files:

| File | Purpose |
| --- | --- |
| `official_doc_text.jsonl` | one parsed text record per manifest document |
| `passages.jsonl` | overlapping passages with character offsets |
| `parse_summary.json` | parse QA counts, warnings, and shortest-document checks |
| `parse_qa_report.json` | deterministic Step 2 QA over parsed docs/passages |

Schema notes:

- `official_doc_text.jsonl` uses top-level `sha256` for the materialized source file hash; parser diagnostics such as `text_sha256`, `page_spans`, `expected_source_filename`, and parser warnings/errors live under `extra`.
- `passages.jsonl` stores passage-level `text_sha256` under `extra`, not as a top-level field.
- `passages.jsonl` uses globally unique IDs of the form `passage_{document_id}_{language}_{passage_index}`.
- Each passage carries `page_start` and `page_end`; manually copied text files are treated as page `1`.

Step 2 QA is deterministic and does not call an LLM:

```bash
python script/qa_official_doc_parse.py \
  --parsed-dir data/0_external/official_doc_parsed \
  --output data/0_external/official_doc_parsed/parse_qa_report.json
```

Latest parse QA:

| Metric | Count |
| --- | ---: |
| manifest rows parsed | 111 |
| PDF files parsed by `pdfminer.six` | 84 |
| manual text files parsed | 27 |
| documents `ok` | 111 |
| documents `low_text` | 0 |
| documents `failed` | 0 |
| passages | 3790 |
| total parsed characters | 5,466,035 |
| min / max document characters | 518 / 1,446,106 |
| duplicate `passage_id` rows | 0 |
| passages missing `page_start` / `page_end` | 0 |
| source `sha256` mismatches | 0 |
| passage ID format violations | 0 |
| page range mismatches from `extra.page_spans` | 0 |
| QA errors | 0 |
| QA warnings | 1 category: 5 short official-statement documents under 1,000 characters |

LLM use starts after this deterministic text layer. We separate three roles and keep only the systems layer load-bearing for the ICDE main results:

- Non-load-bearing metadata enrichment: summaries, actors, dates, language notes, section outlines, and candidate passage hints. These are versioned by `model_id`, prompt version, schema version, and input hash, but do not define benchmark ground truth.
- Diagnostic triage: fixed claims plus deterministic top-k passages may be judged by LLMs to identify passage quality issues, retrieval failures, model disagreements, and language-retrieval failures. These judgments are not gold labels and do not support the main experimental claims.
- Secondary direct scoring: D_SAS-MM validation follows the GeoGround direct-scoring protocol over document/evidence bundles. Long documents may use document-level map-reduce scoring, but the measurement unit remains the document/bundle rather than a claim-conditioned top-k passage set.

Frozen sidecar protocol:

```text
Stage B:
  model: Gemini 2.5 Flash
  scope: non-load-bearing metadata only
  outputs: summary, actors, dates, language_note, section_outline,
           candidate_passage_hints
  hard boundary: Stage B outputs never enter diagnostic triage,
                 direct scoring, or main metrics

Stage C:
  status: diagnostic triage only
  input: fixed claims + deterministic top-k passage retrieval over parsed text
  use: passage_quality_issue / retrieval_failure / model_disagreement /
       language_retrieval_failure diagnostics
  non-use: not ground truth; not used for main E1-E3 conclusions

E4:
  status: secondary validation
  protocol: GeoGround-style direct scoring over document/evidence bundles
  scorers: independent scorer models report D_SAS-MM sensitivity
```

This freezes LLM use as a sidecar. The paper main line remains SMPI/BPE retrieval, deterministic coverage/constraint metrics, and systems-efficiency evaluation. Stage C outputs are retained as diagnostics, while verified grounding claims are limited to explicitly human-audited subsets.

Step 3 dry run plans only the first role and makes no LLM calls:

```bash
python script/plan_metadata_extraction.py \
  --parsed-dir data/0_external/official_doc_parsed \
  --external-assets data/0_external/external_assets.jsonl \
  --output-dir data/1_intermediate/metadata_extraction \
  --model-id gemini-2.5-flash \
  --dry-run
```

Current dry-run outputs:

| File | Purpose |
| --- | --- |
| `data/1_intermediate/metadata_extraction/metadata_extraction_tasks.jsonl` | one planned non-load-bearing extraction task per parsed official document |
| `data/1_intermediate/metadata_extraction/metadata_extraction_dry_run_summary.json` | task counts, batching strategy, and pointer skip diagnostics |

Dry-run result:

| Metric | Count |
| --- | ---: |
| official documents planned | 111 |
| LLM calls made | 0 |
| planned extraction batches | 244 |
| full-text tasks | 97 |
| passage-map-reduce tasks | 14 |
| pointer assets skipped | 29 |
| skipped `news_pointer` rows | 23 |
| skipped `map_pointer` rows | 6 |

The planner deliberately skips pointer assets, including GDELT DOC article pointers and Visual GKG/social-image restricted pointers. These records already carry pointer metadata and do not require text extraction.

Step 4 executes Stage B as a sidecar, still restricted to non-load-bearing metadata:

```bash
python script/extract_official_doc_metadata.py \
  --limit 3 \
  --overwrite \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --sleep-seconds 1
```

After the trial run succeeds, run the full resumable pass:

```bash
python script/extract_official_doc_metadata.py \
  --model-id gemini-2.5-flash \
  --resume \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --sleep-seconds 1
```

Execution outputs:

| File | Purpose |
| --- | --- |
| `data/1_intermediate/metadata_extraction/official_doc_metadata_enrichment.jsonl` | one Stage B metadata sidecar row per official document task |
| `data/1_intermediate/metadata_extraction/official_doc_metadata_batch_outputs.jsonl` | batch-level intermediate rows for long documents |
| `data/1_intermediate/metadata_extraction/official_doc_metadata_enrichment_summary.json` | attempted/completed/failed counts and LLM-call count |

The executor enforces the Stage B/Stage C boundary in the artifact itself: every row is written with `load_bearing=false`, `stage_c_excluded=true`, and `stage_c_policy=never_use_stage_b_outputs`. Its schema accepts only `summary`, `actors`, `dates`, `language_note`, `section_outline`, and `candidate_passage_hints`; load-bearing fields such as `document_position`, support/contradict labels, and claim-grounding decisions are discarded if a model returns them.

Latest Stage B completion status:

| Metric | Value |
| --- | ---: |
| document-level metadata rows | 111 |
| completed rows | 111 |
| failed rows | 0 |
| long-document batch rows | 147 |
| SCS arbitration batches | 47/47 |
| SCS arbitration final reduce | Gemini reduce, `llm_calls=48` |

The final SCS arbitration task was rerun with a selected-task execution and `max_output_tokens=40000`; it replaced the earlier deterministic fallback with a successful Gemini reduce while preserving `prompt_version=official_doc_metadata_v0`.

Paper wording:

```text
Official-document text is produced by a deterministic parsing layer before any
LLM-assisted annotation. Manual materialization is recorded in a manifest, and
only manifest-listed files are parsed. LLM enrichment is split into
non-load-bearing metadata extraction, diagnostic passage-level triage, and
GeoGround-style direct scoring. Stage B metadata and Stage C triage outputs do
not define benchmark ground truth or main system metrics. Stage C is used to
identify passage-quality and retrieval-failure cases for audit; direct-scoring
results are reported only as secondary validation over fixed document/evidence
bundles.
```

For multilingual official documents, the default benchmark activates one representative row per `document_group_id` when possible, normally the `download_priority=true` language variant. Other official language variants are retained in metadata through `available_languages_known_to_have`, `language_role`, `authoritative_status`, and `url_resolver`, but are not counted as separate official evidence rows in the default evaluation.

Language-code convention for the reviewed official-document catalog:

- `zh` and `zh-Hans-CN` are treated as Simplified Chinese official-language variants.
- `zh-Hant-HK` is treated as Traditional Chinese as used in Hong Kong official legal materials.
- `uk` is treated as Ukrainian.
- UN documents with `canonical_language = ["ar", "zh", "en", "fr", "ru", "es"]` are modeled as co-authoritative multilingual documents rather than as English-original documents. The active benchmark selects one representative language variant, while the other UN language URLs remain metadata-only variants under the same `document_group_id`.
- Rows left as `pending` after human review are excluded from the default official-document asset build. Pending reasons include invalid or missing links, and pages where only the title is officially translated while the body text remains in another language.

Current External enrichment warnings:

- `crimea`: Ukraine ACLED coverage begins in `1/2018`, so the 2014 Crimea anchor date is outside country coverage.
- `iraq`: Iraq ACLED coverage begins in `1/2016`, so the 2003 Iraq anchor date is outside country coverage.
- `kosovo`: Kosovo ACLED coverage begins in `1/2018`, so the 2008 Kosovo anchor date is outside country coverage.
- `jcpoa`: Iran ACLED coverage begins in `1/2016`, so the 2015 JCPOA anchor date is outside country coverage; JCPOA is also diplomatic rather than a conflict-event anchor.
- `scs`: skipped by ACLED because no single-country filter is configured; China ACLED coverage begins in `1/2018`, after the 2016 arbitration anchor date.
- `ukraine`: pagination resolved the earlier `--acled-limit` truncation warning; `acled_ukraine.jsonl` now contains `1555` ACLED rows.

External enrichment full benchmark after pagination, event-page Wikimedia filtering, and local image materialization:

| Table | Count |
| --- | ---: |
| source_records | 121 |
| evidence_assets | 2026 |
| source_asset_links | 28713 |
| claim_evidence_hyperedges | 2800 |

Validation: `ok: true`, `errors: []`.

### Evaluation Protocol After ACLED Coverage Audit

The External enrichment ACLED layer is useful, but it is not uniformly available across all 8 events. It should be treated as an external enrichment layer with source-specific coverage limits, not as a balanced source present for every event.

Event groups:

| Group | Events | Interpretation |
| --- | --- | --- |
| Core all-event benchmark | `crimea`, `iraq`, `libya`, `kosovo`, `scs`, `jcpoa`, `ukraine`, `hongkong` | Fair all-8 comparison should use the core corpus, preferably `--no-external-assets`, or report external coverage separately. |
| ACLED-covered External enrichment subset | `libya`, `ukraine`, `hongkong` | Valid for analyzing ACLED structured-event enrichment. |
| Non-ACLED External enrichment subset | `crimea`, `iraq`, `kosovo`, `scs`, `jcpoa` | Valid for text + Wikimedia/map enrichment, but not comparable to ACLED-heavy events on structured-event volume. |

Reporting rule:

- Across all 8 events, do not interpret `structured_event` volume, `modality_coverage`, `layer_diversity`, or `source_diversity` as pure method quality when ACLED assets are included.
- Within a single event, method comparison is still fair because all methods see the same candidate pool.
- For cross-event averages, report stratified results: all-8 core, ACLED-covered subset, and non-ACLED subset.
- If the goal is balanced structured external evidence for all 8 events, add a historically broader event source such as GDELT rather than treating ACLED-zero events as failed examples.

Additional implementation notes:

- ACLED-covered subset results must be reported per event, not as one aggregate. `ukraine`, `hongkong`, and `libya` have very different ACLED row counts, so an aggregate would mostly report Ukraine.
- External enrichment performance numbers are not directly comparable across events because `source_asset_links` grows with event-local external assets. The full External enrichment benchmark has `28713` links after active image and official-document selection, compared with `1407` in the no-external full benchmark.
- A Smoke build vs External enrichment delta layer is required for ACLED-covered events. This answers whether external structured assets improve retrieval for the same event, instead of only asking whether event-level asset counts increased.
- ACLED and future GDELT rows can share `modality=structured_event`, but their semantics should not be aggregated blindly. `asset_source` must remain part of reporting because ACLED conflict-event rows and GDELT CAMEO/news-coded rows are different evidence types.

### GDELT Adapter Boundary

Implemented a first GDELT connector as a pointer adapter:

- `script/geomosaic_hg/clients/gdelt.py`
- `script/fetch_gdelt_doc_assets.py`

The current adapter targets GDELT DOC 2.0 and writes pointer assets only:

```json
{
  "asset_source": "GDELT_DOC",
  "modality": "text",
  "source_layer": "news",
  "evidence_role": "context",
  "extra": {
    "collection_channel": "gdelt_doc_search",
    "record_type": "news_pointer",
    "curation_level": "machine_indexed_news_pointer",
    "active_policy": "pointer_enrichment"
  }
}
```

If a DOC article carries Visual GKG/social-image metadata, the fetcher emits a second restricted image pointer:

```json
{
  "asset_source": "GDELT_DOC",
  "modality": "image_restricted_pointer",
  "source_layer": "news",
  "redistribution_flag": false,
  "extra": {
    "collection_channel": "gdelt_doc_visual_gkg",
    "record_type": "image_restricted_pointer",
    "curation_level": "machine_indexed_image_pointer",
    "active_policy": "pointer_enrichment"
  }
}
```

This intentionally does not treat GDELT DOC as an ACLED replacement. GDELT DOC is a full-text news/search and visual-metadata surface; CAMEO-style structured events should be handled by a later `GDELT_EVENTS` adapter. Hong Kong 2020 and Ukraine 2022 are marked as `source_temporal_coverage = "event_window"` by default. Iraq 2003, Kosovo 2008, Libya 2011, Crimea 2014, JCPOA 2015, and SCS 2016 are marked as `source_temporal_coverage = "retrospective_context"` rather than event-window evidence.

Implemented event-scoped retrieval:

```bash
python script/run_retrieval.py \
  --bench-dir data/enriched_full_bench \
  --events ukraine \
  --query "Ukraine sovereignty territorial integrity" \
  --budget 3 \
  --cutoff 2026-01-01T00:00:00Z
```

Implemented ACLED subset delta report:

```bash
python script/evaluate_enrichment_delta.py \
  --events libya,ukraine,hongkong \
  --budget 5 \
  --output-json data/reports/acled_enrichment_delta.json \
  --output-csv data/reports/acled_enrichment_delta.csv
```

This report is intentionally per-event. It should not be collapsed into a single ACLED-subset average unless macro-averaging is explicitly used.

Latest ACLED-covered delta after preserving event-level `structured_event` assets when external assets are attached:

| Event | Delta objective | Delta modality coverage | Delta layer diversity | Delta viewpoint coverage |
| --- | ---: | ---: | ---: | ---: |
| libya | +1.0 | +1 | 0 | 0.0 |
| ukraine | 0.0 | 0 | 0 | 0.0 |
| hongkong | +1.0 | +1 | 0 | 0.0 |

Interpretation: the earlier negative deltas were a construction artifact where external `image_full` assets displaced the event-level `structured_event` context inside capped hyperedge evidence sets. After the fix, External enrichment no longer reduces coverage metrics. The current BPE objective still does not directly reward the volume of ACLED rows, so ACLED should be reported as provenance enrichment unless a source-specific ACLED usage metric or query task is added.
