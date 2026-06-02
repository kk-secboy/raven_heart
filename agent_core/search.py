"""Small deterministic relevance ranking used by core registries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class SearchDocument(Generic[T]):
    item: T
    text: str
    priority: int = 0
    name: str = ""


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(str(text or "")) if token)


def rank_documents(
    query: str,
    documents: tuple[SearchDocument[T], ...],
    *,
    limit: int = 8,
) -> tuple[T, ...]:
    query_terms = tokenize(query)
    if not query_terms or not documents:
        return ()

    tokenized = [tokenize(document.text) for document in documents]
    avg_len = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    doc_freq: dict[str, int] = {}
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scored: list[tuple[float, int, str, T]] = []
    for document, tokens in zip(documents, tokenized, strict=True):
        if not tokens:
            continue
        score = 0.0
        token_counts: dict[str, int] = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        for term in query_terms:
            score += _bm25_term_score(
                term=term,
                term_frequency=token_counts.get(term, 0),
                doc_count=len(documents),
                doc_frequency=doc_freq.get(term, 0),
                doc_len=len(tokens),
                avg_doc_len=avg_len,
            )
            if term in document.name.lower():
                score += 1.5
            elif term in document.text.lower():
                score += 0.35
        if score > 0:
            scored.append((score, document.priority, document.name, document.item))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    if limit <= 0:
        limit = len(scored)
    return tuple(item for _, _, _, item in scored[:limit])


def _bm25_term_score(
    *,
    term: str,
    term_frequency: int,
    doc_count: int,
    doc_frequency: int,
    doc_len: int,
    avg_doc_len: float,
) -> float:
    if not term_frequency or not doc_frequency:
        return 0.0
    k1 = 1.2
    b = 0.75
    idf = math.log(1 + (doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
    denominator = term_frequency + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
    return idf * ((term_frequency * (k1 + 1)) / max(denominator, 1e-9))

