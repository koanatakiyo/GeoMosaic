#!/usr/bin/env python3
"""Organize manually materialized official documents into parse-ready folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("data/0_external/extra_official_links_multi_lang")
DEFAULT_STATE = DEFAULT_INPUT_DIR / "geomosaic_manual_materialization_state.json"
DEFAULT_OUTPUT_DIR = Path("data/0_external/official_doc_materialized")
DOCUMENT_SUFFIXES = {".pdf", ".txt"}
FILENAME_CORRECTIONS = {
    "jcpoa__x__x__en__jcpoa-iaea-infcirc-887.pdf": "jcpoa__multilateral__anchor_document__en__jcpoa-iaea-infcirc-887.pdf",
    "scs__x__x__zh-Hans-CN__scs-fmprc-position-paper-jurisdiction-2014-12-07.txt": "scs__china__legal_background__zh-Hans-CN__scs-fmprc-position-paper-jurisdiction-2014-12-07.txt",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_materialized_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower().lstrip(".")
    parts = stem.split("__", 4)
    if len(parts) != 5:
        return {
            "event_id": "unknown",
            "viewpoint_origin": "unknown",
            "evidence_scope": "unknown",
            "language": "unknown",
            "document_id": stem,
            "file_extension": suffix,
            "filename_parse_status": "failed",
        }
    event_id, viewpoint_origin, evidence_scope, language, document_id = parts
    return {
        "event_id": event_id,
        "viewpoint_origin": viewpoint_origin,
        "evidence_scope": evidence_scope,
        "language": language,
        "document_id": document_id,
        "file_extension": suffix,
        "filename_parse_status": "ok",
    }


def corrected_materialized_filename(filename: str) -> str:
    """Return the curation-corrected filename used in parse-ready outputs."""
    return FILENAME_CORRECTIONS.get(filename, filename)


def output_kind(filename: str, method: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt" or method in {"manual_copy_text", "manual_copy_html"}:
        return "manual_text"
    return "files"


def destination_for(output_dir: Path, filename: str, method: str | None) -> Path:
    parsed = parse_materialized_filename(filename)
    kind = output_kind(filename, method)
    return output_dir / kind / parsed["event_id"] / filename


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha256_file(src) == sha256_file(dst):
        return
    shutil.copy2(src, dst)


def resolve_source_file(filename: str, actual_files: dict[str, Path]) -> tuple[Path | None, str, str]:
    """Resolve a materialized source, allowing manual PDF/TXT suffix corrections."""
    exact = actual_files.get(filename)
    if exact:
        return exact, filename, "exact"

    expected = Path(filename)
    matches = [
        path
        for path in actual_files.values()
        if path.stem == expected.stem and path.suffix.lower() in DOCUMENT_SUFFIXES
    ]
    if len(matches) == 1:
        actual = matches[0]
        return actual, actual.name, "alternate_extension"
    if len(matches) > 1:
        return None, filename, "ambiguous_alternate_extension"
    return None, filename, "missing"


def state_rows(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = state.get("state", {})
    if not isinstance(rows, dict):
        raise ValueError("manual materialization state must contain a dict field named 'state'")
    return rows


def organize(input_dir: Path, state_path: Path, output_dir: Path, include_unmatched: bool = False) -> dict[str, Any]:
    exported = read_json(state_path)
    rows_by_task = state_rows(exported)
    organized_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    materialized_by_filename: dict[str, tuple[str, dict[str, Any]]] = {}
    for task_key, row in rows_by_task.items():
        filename = row.get("filename")
        if row.get("work_status") == "materialized" and filename:
            materialized_by_filename[str(filename)] = (task_key, row)

    actual_files = {
        p.name: p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in DOCUMENT_SUFFIXES
    }
    consumed_actual_files: set[str] = set()

    manifest_rows: list[dict[str, Any]] = []
    copied = 0
    missing_source_files: list[str] = []

    for filename in sorted(materialized_by_filename):
        task_key, row = materialized_by_filename[filename]
        src, actual_filename, source_resolution = resolve_source_file(filename, actual_files)
        corrected_filename = corrected_materialized_filename(actual_filename)
        corrected = corrected_filename != actual_filename
        parsed = parse_materialized_filename(corrected_filename)
        method = row.get("materialization_method")
        dst = destination_for(output_dir, corrected_filename, method)
        manifest: dict[str, Any] = {
            "task_key": task_key,
            "state_source_filename": filename,
            "expected_source_filename": corrected_filename,
            "actual_source_filename": actual_filename,
            "source_filename": corrected_filename,
            "source_resolution": source_resolution,
            "metadata_correction_applied": corrected,
            "source_path": (src.as_posix() if src else (input_dir / filename).as_posix()),
            "source_url": row.get("source_url", ""),
            "materialization_method": method,
            "work_status": row.get("work_status"),
            "output_kind": output_kind(corrected_filename, method),
            "organized_at": organized_at,
            **parsed,
        }
        if src:
            consumed_actual_files.add(src.name)
            copy_if_needed(src, dst)
            copied += 1
            manifest.update(
                {
                    "organize_status": "copied" if source_resolution == "exact" else f"copied_{source_resolution}",
                    "local_path": dst.as_posix(),
                    "sha256": sha256_file(dst),
                    "size_bytes": dst.stat().st_size,
                }
            )
        else:
            missing_source_files.append(filename)
            manifest.update(
                {
                    "organize_status": "missing_source_file",
                    "local_path": "",
                    "sha256": "",
                    "size_bytes": 0,
                }
            )
        manifest_rows.append(manifest)

    unmatched_files = sorted(set(actual_files) - consumed_actual_files)
    if include_unmatched:
        for filename in unmatched_files:
            src = actual_files[filename]
            parsed = parse_materialized_filename(filename)
            dst = destination_for(output_dir, filename, "manual_file_present")
            copy_if_needed(src, dst)
            copied += 1
            manifest_rows.append(
                {
                    "task_key": "",
                    "source_filename": filename,
                    "source_path": src.as_posix(),
                    "source_url": "",
                    "materialization_method": "manual_file_present",
                    "work_status": "unmatched_file",
                    "output_kind": output_kind(filename, None),
                    "organized_at": organized_at,
                    "organize_status": "copied_unmatched",
                    "local_path": dst.as_posix(),
                    "sha256": sha256_file(dst),
                    "size_bytes": dst.stat().st_size,
                    **parsed,
                }
            )

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"
    write_jsonl(manifest_path, manifest_rows)

    summary = {
        "input_dir": input_dir.as_posix(),
        "state_path": state_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "organized_at": organized_at,
        "materialized_state_rows": len(materialized_by_filename),
        "actual_source_files": len(actual_files),
        "manifest_rows": len(manifest_rows),
        "copied_files": copied,
        "missing_source_files": missing_source_files,
        "unmatched_source_files": unmatched_files,
        "unmatched_policy": "copied" if include_unmatched else "skipped",
        "by_event": dict(sorted(Counter(row["event_id"] for row in manifest_rows if row.get("organize_status", "").startswith("copied")).items())),
        "by_language": dict(sorted(Counter(row["language"] for row in manifest_rows if row.get("organize_status", "").startswith("copied")).items())),
        "by_output_kind": dict(sorted(Counter(row["output_kind"] for row in manifest_rows if row.get("organize_status", "").startswith("copied")).items())),
        "by_organize_status": dict(sorted(Counter(row["organize_status"] for row in manifest_rows).items())),
    }
    write_json(summary_path, summary)

    readme_path.write_text(
        "\n".join(
            [
                "# Manual Official Document Materialization",
                "",
                "This directory is generated by `script/organize_manual_materialized_docs.py`.",
                "Original manually downloaded/copied files remain in `data/0_external/extra_official_links_multi_lang/`.",
                "",
                "- `files/{event}/`: PDF and other file-like official documents.",
                "- `manual_text/{event}/`: manually copied text documents.",
                "- `manifest.jsonl`: one row per materialized task/file with checksums.",
                "- `summary.json`: aggregate counts and missing/unmatched file diagnostics.",
                "",
                "The organizer copies files rather than moving them so the manual review workspace remains intact.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-unmatched", action="store_true", help="Copy source files not referenced by materialized state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = organize(args.input_dir, args.state, args.output_dir, include_unmatched=args.include_unmatched)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
