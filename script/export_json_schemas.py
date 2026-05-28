#!/usr/bin/env python3
"""Export JSON Schema files for the four core JSONL tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from geomosaic_hg.paths import DATA_DIR, ensure_dir
from geomosaic_hg.schema import TABLE_CLASSES


def python_type_to_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if origin is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": python_type_to_schema(item_type)}
    if origin is dict:
        return {"type": "object"}
    return {}


def schema_for_table(name: str, cls: type) -> dict[str, Any]:
    props = {}
    required = []
    hints = get_type_hints(cls)
    for f in fields(cls):
        props[f.name] = python_type_to_schema(hints.get(f.name, f.type))
        if f.default is MISSING and f.default_factory is MISSING:
            required.append(f.name)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": name,
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": required,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "schema")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    for name, cls in TABLE_CLASSES.items():
        path = args.output_dir / f"{name}.schema.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(schema_for_table(name, cls), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        print(path)


if __name__ == "__main__":
    main()
