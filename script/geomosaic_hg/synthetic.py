"""Distribution-preserving block replication for system stress tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .build import load_tables, table_paths
from .io import stable_hash, write_json, write_jsonl
from .paths import BENCH_DIR, ensure_dir, relative_to_project


def _replicate_id(value: str, block: int) -> str:
    return f"{value}__rep{block}"


def replicate_tables(input_dir: Path = BENCH_DIR, output_dir: Path | None = None, blocks: int = 10) -> dict[str, Any]:
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_synthetic_{blocks}x"
    ensure_dir(output_dir)
    tables = load_tables(input_dir)

    source_rows = []
    asset_rows = []
    link_rows = []
    edge_rows = []
    for block in range(blocks):
        for row in tables["source_records"]:
            r = dict(row)
            r["source_id"] = _replicate_id(row["source_id"], block)
            r["event_id"] = _replicate_id(row["event_id"], block)
            r["extra"] = {**r.get("extra", {}), "synthetic_block": block}
            source_rows.append(r)
        for row in tables["evidence_assets"]:
            r = dict(row)
            r["asset_id"] = _replicate_id(row["asset_id"], block)
            r["event_id"] = _replicate_id(row["event_id"], block)
            if r.get("extra", {}).get("source_id"):
                r["extra"]["source_id"] = _replicate_id(r["extra"]["source_id"], block)
            r["embedding_id"] = _replicate_id(row.get("embedding_id", row["asset_id"]), block)
            r["perceptual_hash"] = stable_hash(f"{row['perceptual_hash']}:{block}", 64)
            r["extra"] = {**r.get("extra", {}), "synthetic_block": block}
            asset_rows.append(r)
        for row in tables["source_asset_links"]:
            r = dict(row)
            r["source_id"] = _replicate_id(row["source_id"], block)
            r["asset_id"] = _replicate_id(row["asset_id"], block)
            r["extra"] = {**r.get("extra", {}), "synthetic_block": block}
            link_rows.append(r)
        for row in tables["claim_evidence_hyperedges"]:
            r = dict(row)
            r["hyperedge_id"] = _replicate_id(row["hyperedge_id"], block)
            r["claim_id"] = _replicate_id(row["claim_id"], block)
            r["event_id"] = _replicate_id(row["event_id"], block)
            r["source_record_set"] = [_replicate_id(v, block) for v in row.get("source_record_set", [])]
            r["evidence_asset_set"] = [_replicate_id(v, block) for v in row.get("evidence_asset_set", [])]
            r["provenance_trace"] = [f"{item}:rep{block}" for item in row.get("provenance_trace", [])]
            r["extra"] = {**r.get("extra", {}), "synthetic_block": block}
            edge_rows.append(r)

    counts = {
        "source_records": write_jsonl(output_dir / "source_records.jsonl", source_rows),
        "evidence_assets": write_jsonl(output_dir / "evidence_assets.jsonl", asset_rows),
        "source_asset_links": write_jsonl(output_dir / "source_asset_links.jsonl", link_rows),
        "claim_evidence_hyperedges": write_jsonl(output_dir / "claim_evidence_hyperedges.jsonl", edge_rows),
    }
    summary = {
        "input_dir": relative_to_project(input_dir),
        "output_dir": relative_to_project(output_dir),
        "blocks": blocks,
        "counts": counts,
        "warning": "Synthetic replicated data is for index build time, latency, and pruning ratio only; do not use for quality metrics.",
    }
    write_json(output_dir / "summary.json", summary)
    return summary
