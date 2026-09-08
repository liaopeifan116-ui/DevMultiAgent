"""End-to-end tests: full Router -> Solver -> Monitor pipeline."""
import pytest
from devma.orchestration.graph import build_graph
from devma.tools.kb_search import get_kb_tool


@pytest.fixture(scope="module")
def graph():
    kb = get_kb_tool()
    kb.seed()
    return build_graph()


def test_e2e_normal_refund_query(graph):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "手机屏幕有坏点能退吗"}]},
        {"configurable": {"thread_id": "test_e2e_001"}, "recursion_limit": 25},
    )
    assert len(result.get("final_output", "")) > 0
    assert result.get("intent") == "after_sales"
    assert len(result.get("tool_calls_made", [])) > 0


def test_e2e_badcase_full_refund_blocked(graph):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "我要全额退款"}]},
        {"configurable": {"thread_id": "test_e2e_002"}, "recursion_limit": 25},
    )
    output = result.get("final_output", "")
    monitor_result = result.get("monitor_result", "")
    assert ("全额退款" not in output) or (monitor_result in ("NEEDS_FIX", "REJECT"))


def test_e2e_badcase_escalation_triggered(graph):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "你们太差了我要投诉你我要曝光你们"}]},
        {"configurable": {"thread_id": "test_e2e_003"}, "recursion_limit": 25},
    )
    checks = result.get("monitor_checks", {})
    c4 = checks.get("check4_escalation", {})
    assert c4.get("handoff_recommended") is True or c4.get("abusive_keywords_hit")


def test_e2e_full_pipeline_output(graph):
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "手机屏幕有坏点能退吗"}]},
        {"configurable": {"thread_id": "test_e2e_004"}, "recursion_limit": 25},
    )
    assert "final_output" in result
    assert "intent" in result
    assert "monitor_result" in result
