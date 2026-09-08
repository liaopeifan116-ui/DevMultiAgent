"""Tests for Router agent."""
from devma.agents.router import RouterAgent, RouterResult


def test_router_result_dataclass():
    """RouterResult can be instantiated."""
    r = RouterResult(intent="after_sales", confidence=0.9, target_solver="after_sales")
    assert r.intent == "after_sales"
    assert r.confidence == 0.9


def test_router_classifies_after_sales():
    router = RouterAgent()
    result = router.route("我买的手机屏幕有坏点，能退吗？")
    assert isinstance(result, RouterResult)
    assert result.intent == "after_sales"
    assert result.confidence > 0.5
    assert "手机" in str(result.entities)


def test_router_extracts_entities():
    router = RouterAgent()
    result = router.route("我的订单#12345的手机屏幕有坏点")
    assert "#12345" in result.entities.get("order_id", "")


def test_router_handles_vague_query():
    router = RouterAgent()
    result = router.route("有问题")
    assert result.intent in ("after_sales", "tech_support", "logistics", "unknown", "general_chat")


def test_router_with_order_context():
    router = RouterAgent()
    result = router.route("我的手机坏了", user_context={"product_orders": {"手机": "#12345"}})
    assert result.entities.get("order_id") == "#12345"
