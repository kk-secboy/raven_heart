from __future__ import annotations

from agent_core.search import SearchDocument, rank_documents, tokenize


def test_tokenize_keeps_hyphen_and_underscore_terms() -> None:
    assert tokenize("http_probe code-review!") == ("http_probe", "code-review")


def test_rank_documents_uses_relevance_priority_and_name_boost() -> None:
    docs = (
        SearchDocument(item="generic", text="probe probe service", priority=0, name="generic"),
        SearchDocument(item="http", text="http probe service", priority=0, name="http_probe"),
        SearchDocument(item="priority", text="http service", priority=10, name="priority"),
    )

    ranked = rank_documents("http probe", docs, limit=3)

    assert ranked[0] == "http"
    assert set(ranked) == {"generic", "http", "priority"}

