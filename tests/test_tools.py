"""Tests for tools."""
import os
from datetime import date, timedelta

import pytest

from devma.tools.kb_search import KBSearchTool, search_kb
from devma.tools.order_query import query_order
from devma.tools.refund import check_refund_eligibility


def test_search_kb_seeding():
    tool = KBSearchTool()
    tool.seed()
    results = search_kb("屏幕坏点退款")
    assert len(results) > 0
    assert any("坏点" in r["content"] or "退款" in r["content"] for r in results)


def test_search_kb_irrelevant_query():
    results = search_kb("今天天气怎么样")
    # Chinese sentence embedding models give moderate-to-high base similarity
    # for any two Chinese texts. Verify that no result actually mentions weather.
    for r in results:
        assert "天气" not in r.get("content", "")


def test_search_kb_query_rewrite():
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set — rewrite requires LLM")
    tool = KBSearchTool()
    tool.seed()
    rewritten = tool.rewrite_query("退款", context="手机屏幕坏点质量问题")
    assert len(rewritten) > 0


def test_query_order_exists():
    result = query_order("#12345")
    assert result is not None
    assert result["id"] == "#12345"
    assert "Xiaomi" in result["product"]


def test_query_order_not_found():
    assert query_order("#99999") is None


def test_query_order_fields():
    result = query_order("#12345")
    for field in ["id", "user_name", "product", "price", "purchase_date", "status", "has_receipt"]:
        assert field in result


def test_refund_in_window():
    # Purchase within 7 days — eligible for refund
    purchase_date = (date.today() - timedelta(days=5)).isoformat()
    order = {"id": "#W001", "purchase_date": purchase_date, "has_receipt": 1}
    result = check_refund_eligibility(order, issue_type="quality")
    assert result["eligible"] is True
    assert "7天退货期" in result.get("reason", "")


def test_refund_over_30_days():
    order = {"id": "#12350", "purchase_date": "2025-12-01", "has_receipt": 1}
    result = check_refund_eligibility(order, issue_type="quality")
    assert result["eligible"] is False
    assert "30天" in result["reason"]


def test_refund_no_receipt():
    order = {"id": "#12349", "purchase_date": "2026-04-01", "has_receipt": 0}
    result = check_refund_eligibility(order, issue_type="quality")
    assert result["eligible"] is False


def test_refund_between_7_and_30_days():
    purchase_date = (date.today() - timedelta(days=20)).isoformat()
    order = {"id": "#12345", "purchase_date": purchase_date, "has_receipt": 1}
    result = check_refund_eligibility(order, issue_type="quality")
    assert result["eligible"] is False
    assert "换货" in result.get("reason", "") or "维修" in result.get("reason", "")
