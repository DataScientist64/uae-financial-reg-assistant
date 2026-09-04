import pytest
from src.ingest import get_or_create_vector_store
from src.retrieve import retrieve, format_context


@pytest.fixture
def vector_store():
    return get_or_create_vector_store()


def test_retrieve_returns_results(vector_store):
    results = retrieve("consumer protection regulation", vector_store)
    assert len(results) > 0


def test_retrieve_returns_correct_number(vector_store):
    results = retrieve("Basel III capital requirements", vector_store)
    assert len(results) <= 4


def test_format_context_not_empty(vector_store):
    docs = retrieve("consumer protection", vector_store)
    context = format_context(docs)
    assert len(context) > 0


def test_format_context_contains_source(vector_store):
    docs = retrieve("consumer protection", vector_store)
    context = format_context(docs)
    assert "Source:" in context


def test_out_of_scope_returns_empty_context():
    context = format_context([])
    assert context == ""