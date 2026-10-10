"""Tests for query rewriting module with mocked Groq and fallback modes."""

from unittest.mock import MagicMock, patch
from incident_rag.query_rewriting import QueryRewriter


def test_rule_based_fallback():
    # Force no API key to test deterministic fallback
    rewriter = QueryRewriter(api_key="")
    assert rewriter.client is None

    res = rewriter.rewrite("What caused the high vibration on turbine HP-17?", num_rewrites=2)
    assert res.original_query == "What caused the high vibration on turbine HP-17?"
    assert len(res.rewritten_queries) > 0
    # Machine ID must be preserved in rewritten variants
    assert any("HP-17" in q for q in res.rewritten_queries)


def test_mock_groq_query_rewriting():
    with patch("incident_rag.query_rewriting.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '{"original_query": "HP-17 trip", "rewritten_queries": ["HP-17 vibration trip limit", "HP-17 bearing babbitt rub"]}'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        rewriter = QueryRewriter(api_key="fake-test-key")
        res = rewriter.rewrite("HP-17 trip", num_rewrites=2)

        assert len(res.rewritten_queries) == 2
        assert "HP-17 vibration trip limit" in res.rewritten_queries
        assert "HP-17 bearing babbitt rub" in res.rewritten_queries
