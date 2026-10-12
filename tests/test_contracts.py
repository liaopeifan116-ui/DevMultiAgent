"""Tests for runtime-validated Agent input/output contracts."""

import pytest
from pydantic import ValidationError

from devma.models.llm import LLMResponse
from devma.orchestration.contracts import (
    MonitorOutput,
    RouterOutput,
    SolverMetrics,
    SolverOutput,
)


def test_router_output_accepts_valid_route_decision():
    result = RouterOutput(
        intent="after_sales",
        confidence=0.92,
        target_solver="after_sales",
        entities={"product": "手机", "issue": "屏幕坏点"},
        reason="质量问题相关的退款咨询",
    )

    assert result.target_solver == "after_sales"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intent", "refund"),
        ("confidence", 1.1),
        ("target_solver", "random_solver"),
    ],
)
def test_router_output_rejects_invalid_route_decision(field, value):
    payload = {
        "intent": "after_sales",
        "confidence": 0.92,
        "target_solver": "after_sales",
        "entities": {},
        "reason": "",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        RouterOutput(**payload)


def test_solver_output_accepts_valid_plan_and_solve_result():
    result = SolverOutput(
        status="completed",
        active_solver="after_sales",
        plan=["查询订单", "检索规则", "生成回复"],
        output="可以申请换货或维修。",
        tool_calls=[],
        metrics=SolverMetrics(exec_tokens=20),
    )

    assert result.status == "completed"
    assert result.metrics.exec_tokens == 20


def test_solver_output_rejects_invalid_status():
    with pytest.raises(ValidationError):
        SolverOutput(
            status="done",
            active_solver="after_sales",
            plan=[],
            output="",
            tool_calls=[],
            metrics={},
        )


def test_monitor_output_accepts_valid_quality_decision():
    result = MonitorOutput(
        decision="PASS",
        reason="All checks passed",
        checks={"check1_keyword": {"hit": False, "matched": []}},
    )

    assert result["result"] == "PASS"


def test_monitor_output_rejects_invalid_decision():
    with pytest.raises(ValidationError):
        MonitorOutput(decision="OK", reason="", checks={})


def test_router_agent_returns_validated_router_output(monkeypatch):
    from devma.agents.router import RouterAgent

    router = RouterAgent()
    monkeypatch.setattr(
        router.llm,
        "chat_json",
        lambda **kwargs: {
            "intent": "after_sales",
            "confidence": 0.92,
            "entities": {"product": "手机"},
        },
    )

    assert isinstance(router.route("手机坏了"), RouterOutput)


def test_solver_agent_returns_validated_solver_output(monkeypatch):
    from devma.agents import solver as solver_module

    solver = solver_module.AfterSalesSolver()
    responses = iter(
        [
            LLMResponse(content='{"plan": ["查询规则", "生成回复"]}', model="fake"),
            LLMResponse(content="可以申请换货或维修。", model="fake"),
        ]
    )
    monkeypatch.setattr(solver.llm, "chat", lambda *args, **kwargs: next(responses))
    from devma.tools import kb_search

    monkeypatch.setattr(kb_search, "search_kb", lambda *args, **kwargs: [])

    result = solver.solve("手机屏幕有坏点", {"product": "手机", "issue": "屏幕坏点"})

    assert isinstance(result, SolverOutput)


def test_monitor_agent_returns_validated_monitor_output(monkeypatch):
    from devma.agents.monitor import MonitorAgent

    monitor = MonitorAgent()
    monkeypatch.setattr(
        monitor, "_check2_commitment", lambda output: {"violation": False}
    )
    monkeypatch.setattr(
        monitor,
        "_check3_grounding",
        lambda output, user_message: {"hallucination": False, "detail": ""},
    )

    result = monitor.check("可以申请换货或维修。", "手机屏幕有坏点", [])

    assert isinstance(result, MonitorOutput)


def test_monitor_hard_rule_rejects_without_llm_calls(monkeypatch):
    from devma.agents.monitor import MonitorAgent

    monitor = MonitorAgent()
    monkeypatch.setattr(
        monitor,
        "_check2_commitment",
        lambda output: pytest.fail("hard-rule rejection must not call the LLM"),
    )
    monkeypatch.setattr(
        monitor,
        "_check3_grounding",
        lambda output, user_message: pytest.fail(
            "hard-rule rejection must not query grounding"
        ),
    )

    result = monitor.check("好的，我们为您办理全额退款。", "我要退款", [])

    assert result.decision == "REJECT"


def test_monitor_escalation_handoffs_without_llm_calls(monkeypatch):
    from devma.agents.monitor import MonitorAgent

    monitor = MonitorAgent()
    monkeypatch.setattr(
        monitor,
        "_check2_commitment",
        lambda output: pytest.fail("escalation handoff must not call the LLM"),
    )
    monkeypatch.setattr(
        monitor,
        "_check3_grounding",
        lambda output, user_message: pytest.fail(
            "escalation handoff must not query grounding"
        ),
    )

    result = monitor.check("好的，我来帮您处理。", "我要投诉你，我要曝光你们！", [])

    assert result.decision == "HANDOFF"
