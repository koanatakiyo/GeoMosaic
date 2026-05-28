"""Dependency-free text scoring utilities."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 1 and t not in STOPWORDS]


def term_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def lexical_score(query: str, document: str) -> float:
    q = term_counter(query)
    d = term_counter(document)
    return cosine_counter(q, d)


def metadata_atoms(row: dict) -> set[str]:
    atoms: set[str] = set()
    for field in ("source_layer_set", "modality_set", "viewpoint_origin_set", "evidence_role_multiset"):
        for value in row.get(field, []) or []:
            atoms.add(f"{field}:{value}")
    return atoms


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
