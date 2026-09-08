"""Tests for Solver agent."""
from devma.agents.solver import AfterSalesSolver


def test_solver_creates_plan():
    solver = AfterSalesSolver()
    result = solver.solve(
        user_message="手机屏幕有坏点能退吗",
        entities={"product": "手机", "issue": "屏幕坏点", "order_id": "#12345"},
    )
    assert len(result["plan"]) >= 2
    assert len(result["output"]) > 0


def test_solver_calls_tools():
    solver = AfterSalesSolver()
    result = solver.solve(
        user_message="帮我查一下订单#12345能不能退款",
        entities={"product": "手机", "issue": "屏幕坏点", "order_id": "#12345"},
    )
    assert len(result["tool_calls"]) > 0
    tools_used = [tc["tool"] for tc in result["tool_calls"]]
    assert any(t in tools_used for t in ["get_order", "search_kb", "check_refund_eligibility"])


def test_solver_generates_chinese_response():
    solver = AfterSalesSolver()
    result = solver.solve(
        user_message="手机屏幕有坏点能退吗",
        entities={"product": "手机", "issue": "屏幕坏点", "order_id": "#12345"},
    )
    assert any('一' <= c <= '鿿' for c in result["output"])


def test_solver_without_order_id():
    """Solver handles case where no order_id is provided."""
    solver = AfterSalesSolver()
    result = solver.solve(
        user_message="手机屏幕有坏点能退吗",
        entities={"product": "手机", "issue": "屏幕坏点"},
    )
    assert len(result["output"]) > 0
    assert len(result["plan"]) >= 1
