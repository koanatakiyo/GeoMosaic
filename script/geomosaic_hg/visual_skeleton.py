"""Visual asset manifesting and optional CLIP/SigLIP embedding index."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, stable_hash, write_json, write_jsonl


VISUAL_MODALITIES = {"image_full", "map_pointer", "image_restricted_pointer"}
EMBEDDABLE_MODALITIES = {"image_full", "map_pointer"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg")


def detect_mime_magic(path: Path) -> str:
    try:
        head = path.read_bytes()[:512]
    except OSError:
        return ""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"RIFF") and b"WEBP" in head[:16]:
        return "image/webp"
    stripped = head.lstrip()
    if stripped.startswith(b"<svg") or b"<svg" in stripped[:128]:
        return "image/svg+xml"
    return ""


def dimensions_from_magic(path: Path, mime: str | None = None) -> tuple[int, int]:
    try:
        data = path.read_bytes()
    except OSError:
        return 0, 0
    mime = mime or detect_mime_magic(path)
    if mime == "image/png" and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if mime == "image/gif" and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if mime == "image/jpeg" and data.startswith(b"\xff\xd8"):
        idx = 2
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while idx + 9 < len(data):
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            idx += 2
            if marker in {0xD8, 0xD9}:
                continue
            if idx + 2 > len(data):
                break
            length = int.from_bytes(data[idx : idx + 2], "big")
            if length < 2 or idx + length > len(data):
                break
            if marker in sof_markers and length >= 7:
                height = int.from_bytes(data[idx + 3 : idx + 5], "big")
                width = int.from_bytes(data[idx + 5 : idx + 7], "big")
                return width, height
            idx += length
    return 0, 0


def _project_path(path: str | Path) -> Path:
    return Path(path)


def _existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return None
    path = _project_path(value)
    return path if path.exists() else None


def resolve_local_image_path(asset: dict[str, Any], image_root: Path) -> Path | None:
    extra = asset.get("extra", {}) if isinstance(asset.get("extra"), dict) else {}
    for value in (extra.get("local_image_path"), asset.get("url_or_pointer")):
        path = _existing_path(str(value) if value else None)
        if path:
            return path

    event_dir = image_root / str(asset.get("event_id", "unknown"))
    asset_id = str(asset.get("asset_id", ""))
    for suffix in IMAGE_SUFFIXES:
        path = event_dir / f"{asset_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(event_dir.glob(f"{asset_id}.*")) if event_dir.exists() else []
    return matches[0] if matches else None


def image_dimensions(path: Path) -> tuple[int, int, str]:
    mime = detect_mime_magic(path)
    width, height = dimensions_from_magic(path, mime)
    if width > 0 and height > 0:
        return width, height, ""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return int(img.width), int(img.height), ""
    except Exception as exc:  # noqa: BLE001 - manifest should record parse failures, not abort.
        return 0, 0, str(exc)


def average_hash(path: Path, size: int = 8) -> str:
    try:
        from PIL import Image

        with Image.open(path) as img:
            gray = img.convert("L").resize((size, size))
            pixels = list(gray.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
        return f"{int(bits, 2):0{size * size // 4}x}"
    except Exception:
        return ""


def visual_status_for(asset: dict[str, Any], local_path: Path | None, width: int, parse_error: str) -> str:
    modality = str(asset.get("modality", ""))
    if modality == "image_restricted_pointer":
        return "pointer_only"
    if not local_path:
        extra = asset.get("extra", {}) if isinstance(asset.get("extra"), dict) else {}
        pointer = str(asset.get("url_or_pointer", ""))
        if modality == "map_pointer" and (
            bool(extra.get("generated_pointer"))
            or "wikipedia.org/wiki/" in pointer
            or "Wikipedia/Wikimedia pointer" in str(asset.get("asset_source", ""))
        ):
            return "pointer_only"
        return "missing_file"
    if parse_error or width <= 0:
        return "parse_failed"
    return "full_image"


def manifest_row(asset: dict[str, Any], *, image_root: Path, backend: str, model_path: Path, external_output_dir: Path) -> dict[str, Any]:
    extra = asset.get("extra", {}) if isinstance(asset.get("extra"), dict) else {}
    local_path = resolve_local_image_path(asset, image_root)
    width = height = 0
    parse_error = ""
    if local_path:
        width, height, parse_error = image_dimensions(local_path)
    status = visual_status_for(asset, local_path, width, parse_error)
    mime_magic = detect_mime_magic(local_path) if local_path else ""
    file_sha = sha256_file(local_path) if local_path and local_path.exists() else ""
    visual_hash = average_hash(local_path) if local_path and status == "full_image" else ""
    embedding_filename = f"image_embeddings_{backend}.npy"
    embedding_path = external_output_dir / embedding_filename
    return {
        "asset_id": asset.get("asset_id", ""),
        "event_id": asset.get("event_id", ""),
        "modality": asset.get("modality", ""),
        "asset_source": asset.get("asset_source", ""),
        "source_layer": asset.get("source_layer", ""),
        "viewpoint_origin": asset.get("viewpoint_origin", ""),
        "collection_channel": extra.get("collection_channel", ""),
        "record_type": extra.get("record_type", ""),
        "page_bound": bool(extra.get("page_bound", False)),
        "proposed_role": extra.get("proposed_role", ""),
        "temporal_status": extra.get("temporal_status") or extra.get("source_temporal_coverage", ""),
        "caption": extra.get("caption") or asset.get("caption_or_transcript", ""),
        "url_or_pointer": asset.get("url_or_pointer", ""),
        "local_path": local_path.as_posix() if local_path else "",
        "file_exists": bool(local_path and local_path.exists()),
        "mime_magic": mime_magic,
        "width": width,
        "height": height,
        "parse_error": parse_error,
        "file_sha256": file_sha,
        "visual_hash": visual_hash,
        "visual_hash_algorithm": "average_hash_8x8" if visual_hash else "",
        "visual_status": status,
        "license_or_terms": asset.get("license_or_terms", ""),
        "redistribution_flag": bool(asset.get("redistribution_flag", False)),
        "embedding_backend": backend,
        "embedding_model_path": model_path.as_posix(),
        "embedding_id": f"{backend}:{stable_hash(str(asset.get('asset_id', '')), 16)}" if status == "full_image" else "",
        "embedding_path": embedding_path.as_posix() if status == "full_image" else "",
        "embedding_row": None,
    }


def load_visual_assets(bench_dir: Path) -> list[dict[str, Any]]:
    assets_path = bench_dir / "evidence_assets.jsonl"
    return [row for row in read_jsonl(assets_path) if row.get("modality") in VISUAL_MODALITIES]


def coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[(str(row.get("event_id", "")), str(row.get("modality", "")), str(row.get("visual_status", "")))] += 1
    return [
        {"event_id": event_id, "modality": modality, "visual_status": status, "count": count}
        for (event_id, modality, status), count in sorted(counts.items())
    ]


def summarize_manifest(rows: list[dict[str, Any]], *, embedding_rows: int, backend: str, model_path: Path, embedding_path: Path) -> dict[str, Any]:
    return {
        "visual_assets_total": len(rows),
        "embedding_backend": backend,
        "embedding_model_path": model_path.as_posix(),
        "embedding_path": embedding_path.as_posix(),
        "embedding_rows": embedding_rows,
        "by_status": dict(Counter(str(row.get("visual_status", "")) for row in rows)),
        "by_modality": dict(Counter(str(row.get("modality", "")) for row in rows)),
        "by_event": dict(Counter(str(row.get("event_id", "")) for row in rows)),
    }


def write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "modality", "visual_status", "count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_embedder(model_path: Path, device: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    model.to(device)
    return processor, model, torch


def _image_for_embedding(path: Path):
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def build_embeddings(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    model_path: Path,
    external_output_dir: Path,
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, Any]], Path, int]:
    import numpy as np

    processor, model, torch = _load_embedder(model_path, device)
    embeddable = [row for row in rows if row.get("visual_status") == "full_image" and row.get("local_path")]
    vectors = []
    index_rows = []
    output_path = external_output_dir / f"image_embeddings_{backend}.npy"
    external_output_dir.mkdir(parents=True, exist_ok=True)

    row_idx = 0
    for start in range(0, len(embeddable), batch_size):
        batch = embeddable[start : start + batch_size]
        images = [_image_for_embedding(Path(row["local_path"])) for row in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(**inputs)
            else:
                output = model(**inputs)
                feats = getattr(output, "image_embeds")
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        batch_vectors = feats.detach().cpu().float().numpy()
        vectors.append(batch_vectors)
        for row in batch:
            row["embedding_row"] = row_idx
            row["embedding_path"] = output_path.as_posix()
            index_rows.append(
                {
                    "embedding_row": row_idx,
                    "embedding_id": row.get("embedding_id", ""),
                    "asset_id": row.get("asset_id", ""),
                    "event_id": row.get("event_id", ""),
                    "modality": row.get("modality", ""),
                    "local_path": row.get("local_path", ""),
                }
            )
            row_idx += 1

    matrix = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype="float32")
    np.save(output_path, matrix)
    write_jsonl(external_output_dir / f"image_embedding_index_{backend}.jsonl", index_rows)
    return rows, output_path, row_idx


def build_visual_manifest(
    *,
    bench_dir: Path,
    image_root: Path,
    output_dir: Path,
    external_output_dir: Path,
    backend: str,
    model_path: Path,
    metadata_only: bool = False,
    batch_size: int = 16,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        manifest_row(asset, image_root=image_root, backend=backend, model_path=model_path, external_output_dir=external_output_dir)
        for asset in load_visual_assets(bench_dir)
    ]
    embedding_path = external_output_dir / f"image_embeddings_{backend}.npy"
    embedding_rows = 0
    if not metadata_only:
        rows, embedding_path, embedding_rows = build_embeddings(
            rows,
            backend=backend,
            model_path=model_path,
            external_output_dir=external_output_dir,
            batch_size=batch_size,
            device=device,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "visual_asset_manifest.jsonl", rows)
    write_coverage_csv(output_dir / "visual_asset_coverage.csv", coverage_rows(rows))
    summary = summarize_manifest(rows, embedding_rows=embedding_rows, backend=backend, model_path=model_path, embedding_path=embedding_path)
    write_json(output_dir / "visual_index_summary.json", summary)
    return rows, summary
