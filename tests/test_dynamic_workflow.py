"""Functional contracts, not evidence for resume performance claims."""

import threading

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from devma.agents.monitor import MonitorAgent
from devma.agents.router import RouterAgent
from devma.agents.solver import AfterSalesSolver, LogisticsSolver, TechSupportSolver
from devma.orchestration.contracts import AfterSalesPlan, RouterOutput, SubTask
from devma.orchestration.graph import build_graph, open_graph
from devma.orchestration.scheduler import (
    blocked_results,
    invalidated_tasks,
    ready_tasks,
)
from devma.orchestration.state import merge_task_results
from devma.tools import kb_search, order_query


def task(key="t1", solver="logistics", deps=(), order_id="#1234", **overrides):
    return SubTask(
        task_id=key,
        intent=solver,
        target_solver=solver,
        confidence=0.9,
        span="查询订单" if solver == "logistics" else "可以退款吗",
        entities={"order_id": order_id},
        depends_on=list(deps),
        resource_keys=[f"order:{order_id}"],
        **overrides,
    ).model_dump()


def config(thread="dynamic-test"):
    return {"configurable": {"thread_id": thread}, "recursion_limit": 60}


def test_same_solver_independent_tasks_parallel():
    route = RouterAgent().route("手机#1234到哪了？还有电脑#5678到哪了？")
    assert route.execution_mode == "parallel"
    assert len(ready_tasks([t.model_dump() for t in route.tasks], {})) == 2


def test_conditional_tasks_depend_on_previous():
    route = RouterAgent().route("手机开不了机，如果是质量问题能退吗？")
    assert route.tasks[1].depends_on == [route.tasks[0].task_id]
    assert route.tasks[1].entities["product"] == "手机"


def test_order_query_then_eligibility_dependency():
    route = RouterAgent().route("查订单#1234到哪了，然后看看这个订单能退吗？")
    assert route.tasks[1].depends_on == [route.tasks[0].task_id]


def test_order_id_not_leaked_into_another_product():
    route = RouterAgent().route("手机#1234能退吗？还有耳机到哪了？")
    assert "order_id" not in route.tasks[1].entities


@pytest.mark.parametrize(
    "tasks",
    [
        [task("t1"), task("t1")],
        [task("t1", deps=("missing",))],
        [task("t1", deps=("t2",)), task("t2", deps=("t1",))],
    ],
)
def test_invalid_dags_rejected(tasks):
    with pytest.raises(ValueError):
        RouterOutput(
            intent="multi_intent",
            confidence=0.9,
            target_solver="logistics",
            tasks=tasks,
        )


def test_solver_semantic_mapping_is_validated():
    with pytest.raises(ValidationError):
        SubTask(
            task_id="t1",
            intent="after_sales",
            target_solver="logistics",
            span="退款",
            confidence=0.9,
        )


def test_resource_write_conflict_serializes_but_reads_do_not():
    read = task("a")
    write = task("b", access="write")
    assert len(ready_tasks([read, write], {})) == 1
    assert len(ready_tasks([read, task("b")], {})) == 2


def test_failed_dependency_blocks_only_downstream():
    tasks = [task("a"), task("b", deps=("a",)), task("c")]
    results = {"a": {"status": "failed"}}
    assert set(blocked_results(tasks, results)) == {"b"}
    assert [t["task_id"] for t in ready_tasks(tasks, results)] == ["c"]


def test_retry_invalidates_transitive_consumers_not_independent_results():
    tasks = [task("a"), task("b", deps=("a",)), task("c", deps=("b",)), task("d")]
    assert invalidated_tasks(tasks, ["a"]) == {"a", "b", "c"}


def test_worker_result_cannot_write_a_different_task_key():
    with pytest.raises(ValueError):
        merge_task_results({}, {"a": {"task_id": "b"}})


def test_conflicting_same_task_writes_rejected():
    with pytest.raises(ValueError):
        merge_task_results(
            {"a": {"task_id": "a", "output": "one"}},
            {"a": {"task_id": "a", "output": "two"}},
        )


def test_parallel_workers_really_overlap(monkeypatch):
    barrier = threading.Barrier(2, timeout=5)
    original = LogisticsSolver.solve

    def concurrent(self, text, entities):
        barrier.wait()
        return original(self, text, entities)

    monkeypatch.setattr(LogisticsSolver, "solve", concurrent)
    result = build_graph().invoke(
        {
            "messages": [
                {"role": "user", "content": "手机#1234到哪了？还有电脑#5678到哪了？"}
            ]
        },
        config(),
    )
    assert len(result["task_results"]) == 2
    assert all(r["status"] == "completed" for r in result["task_results"].values())


def test_upstream_order_facts_are_reused(monkeypatch):
    original, calls = order_query.query_order, []

    def count(order_id):
        calls.append(order_id)
        return original(order_id)

    # Both modules have imported tools, patch both boundaries.
    import devma.agents.solver as solver_module

    monkeypatch.setattr(order_query, "query_order", count)
    monkeypatch.setattr(solver_module, "query_order", count)
    result = build_graph().invoke(
        {
            "messages": [
                {"role": "user", "content": "查询手机#1234到哪了，然后看看能退吗？"}
            ]
        },
        config(),
    )
    assert calls == ["#1234"]
    assert any(c.get("reused_from") for c in result["tool_calls_made"])


def test_failed_task_does_not_erase_independent_success(monkeypatch):
    original = LogisticsSolver.solve

    def fail_one(self, text, entities):
        if entities["order_id"] == "#1234":
            raise TimeoutError("demo timeout")
        return original(self, text, entities)

    monkeypatch.setattr(LogisticsSolver, "solve", fail_one)
    result = build_graph().invoke(
        {
            "messages": [
                {"role": "user", "content": "手机#1234到哪了？还有电脑#5678到哪了？"}
            ]
        },
        config(),
    )
    assert result["handoff_needed"] is True
    assert "ThinkPad" in result["final_output"]
    assert len(result["handoff_context"]["completed_results"]) == 1


def test_tech_react_graph_has_a_loop_and_real_observations():
    agent = TechSupportSolver()
    graph = agent.build_subgraph()
    assert ("action", "think") in {
        (e.source, e.target) for e in graph.get_graph().edges
    }
    state = graph.invoke(
        {"user_message": "手机开不了机怎么办", "entities": {"product": "手机"}}
    )
    assert state["observations"]
    assert state["payload"]["evidence"]
    assert state["steps"] <= 4


def test_after_sales_plan_rejects_eligibility_before_order():
    with pytest.raises(ValidationError):
        AfterSalesPlan(
            steps=[
                {"action": a}
                for a in ("check_eligibility", "lookup_order", "retrieve_policy")
            ]
        )


def test_general_policy_does_not_require_an_order():
    result = AfterSalesSolver().solve("退款多久到账？", {})
    assert result.status == "completed"
    assert {c["tool"] for c in result.tool_calls} == {"search_kb"}
    assert result.evidence


def test_monitor_reuses_provided_evidence_without_search(monkeypatch):
    import devma.agents.monitor as module

    monkeypatch.setattr(
        module, "search_kb", lambda *a, **k: pytest.fail("must reuse provided evidence")
    )
    review = MonitorAgent().check(
        "可申请换货或维修。", "质量问题", ["7-30天质量问题可换修"], []
    )
    assert review.decision == "PASS"
    assert review.checks["check3_grounding"]["evidence_count"] == 1


def test_existing_order_does_not_authorize_refund():
    tools = [
        {"tool": "get_order", "result": {"id": "#1234", "product": "手机"}},
        {
            "tool": "check_refund_eligibility",
            "result": {"eligible": False, "action": "reject"},
        },
    ]
    result = MonitorAgent().check("可以退款。", "能退吗", ["示例政策"], tools)
    assert result.decision == "NEEDS_FIX"


def test_denial_of_prohibited_promise_not_false_blocked():
    result = MonitorAgent().check(
        "不能承诺全额退款，请先核验订单。", "能全额退款吗", ["示例政策"]
    )
    assert result.decision == "PASS"


def test_monitor_api_failure_cannot_fail_open(monkeypatch):
    import devma.agents.monitor as module

    monkeypatch.setattr(module, "DEEPSEEK_API_KEY", "fake")
    monitor = MonitorAgent()
    monkeypatch.setattr(
        monitor.llm, "chat_json", lambda **kw: (_ for _ in ()).throw(TimeoutError())
    )
    assert monitor.check("按政策处理。", "能退吗", ["政策"]).decision == "HANDOFF"


def test_persistent_pause_survives_graph_and_connection_recreation(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    with open_graph(db) as graph:
        paused = graph.invoke(
            {
                "messages": [{"role": "user", "content": "手机#1234到哪了"}],
                "pause_before_execution": True,
            },
            config("persistent"),
        )
        assert paused["__interrupt__"]
        assert not paused["task_results"]
    with open_graph(db) as graph:
        resumed = graph.invoke(Command(resume={"approved": True}), config("persistent"))
        assert resumed["final_output"]
        assert len(resumed["task_results"]) == 1


def test_subgraph_missing_order_can_interrupt_and_resume(tmp_path):
    db = str(tmp_path / "child.db")
    with open_graph(db) as graph:
        paused = graph.invoke(
            {
                "messages": [{"role": "user", "content": "我想退款"}],
                "pause_on_missing_info": True,
            },
            config("missing-order"),
        )
        assert paused["__interrupt__"][0].value["kind"] == "missing_order"
    with open_graph(db) as graph:
        resumed = graph.invoke(
            Command(resume={"order_id": "#1234"}), config("missing-order")
        )
        assert resumed["final_output"]
        assert (
            next(iter(resumed["task_results"].values()))["facts"]["order"]["id"]
            == "#1234"
        )


def test_declined_approval_does_not_execute_tools():
    graph = build_graph()
    cfg = config("declined")
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "手机#1234到哪了"}],
            "pause_before_execution": True,
        },
        cfg,
    )
    result = graph.invoke(Command(resume={"approved": False}), cfg)
    assert result["tool_calls_made"] == []
    assert result["handoff_needed"]


def test_new_turn_clears_old_task_ledger_and_preserves_product_context():
    graph, cfg = build_graph(), config("two-turn")
    first = graph.invoke(
        {"messages": [{"role": "user", "content": "手机订单号是1234到哪了"}]}, cfg
    )
    second = graph.invoke(
        {"messages": [{"role": "user", "content": "我的手机多久到"}]}, cfg
    )
    assert set(first["task_results"]).isdisjoint(second["task_results"])
    assert second["entities"]["order_id"] == "#1234"
    assert len(second["task_results"]) == 1


def test_threads_do_not_share_order_bindings():
    graph = build_graph()
    graph.invoke(
        {"messages": [{"role": "user", "content": "手机订单号是1234到哪了"}]},
        config("one"),
    )
    other = graph.invoke(
        {"messages": [{"role": "user", "content": "我的手机多久到"}]}, config("two")
    )
    assert "order_id" not in other["entities"]


def test_dual_query_fusion_preserves_original_branch(monkeypatch):
    tool = kb_search.KBSearchTool()
    monkeypatch.setattr(tool, "rewrite_query", lambda *a: "expanded")

    def search(query, **kwargs):
        return [{"id": query, "content": query, "source": "demo", "score": 1.0}]

    monkeypatch.setattr(tool, "search", search)
    hits = tool.search_with_rewrite("precise-id", "context", top_k=5)
    assert {r["id"] for r in hits} == {"precise-id", "expanded"}


def test_long_markdown_sections_have_bounded_chunks():
    chunks = kb_search._split_long_section("测试段落\n\n" * 600)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)


def test_evidence_has_source_heading_id_and_version_hash():
    hits = kb_search.search_kb("手机无法开机怎么办", top_k=5)
    assert hits
    assert all(
        {"id", "source", "heading", "content", "content_hash"} <= r.keys() for r in hits
    )


def test_model_cannot_invent_an_existing_order_id(monkeypatch):
    import devma.agents.router as module

    monkeypatch.setattr(module, "DEEPSEEK_API_KEY", "fake")
    router = RouterAgent()
    monkeypatch.setattr(
        router.llm,
        "chat_json",
        lambda **kw: {
            "intent": "logistics",
            "confidence": 0.9,
            "entities": {"product": "手机", "order_id": "#5678"},
        },
    )
    result = router.route("手机#1234到哪了")
    assert result.tasks[0].entities["order_id"] == "#1234"


def test_valid_model_plan_is_actually_executed_in_chosen_order(monkeypatch):
    import devma.orchestration.subgraphs as module

    monkeypatch.setattr(module, "DEEPSEEK_API_KEY", "fake")
    agent = AfterSalesSolver()
    monkeypatch.setattr(
        agent.llm,
        "chat_json",
        lambda **kw: {
            "steps": [
                {"action": a}
                for a in ("retrieve_policy", "lookup_order", "check_eligibility")
            ]
        },
    )
    result = agent.solve("订单#1234能退吗", {"order_id": "#1234"})
    assert [c["tool"] for c in result.tool_calls] == [
        "search_kb",
        "get_order",
        "check_refund_eligibility",
    ]
    assert result.llm_used


def test_invalid_model_plan_falls_back_to_safe_order(monkeypatch):
    import devma.orchestration.subgraphs as module

    monkeypatch.setattr(module, "DEEPSEEK_API_KEY", "fake")
    agent = AfterSalesSolver()
    monkeypatch.setattr(
        agent.llm, "chat_json", lambda **kw: {"steps": [{"action": "execute_refund"}]}
    )
    result = agent.solve("订单#1234能退吗", {"order_id": "#1234"})
    assert [c["tool"] for c in result.tool_calls] == [
        "get_order",
        "search_kb",
        "check_refund_eligibility",
    ]


def test_positive_promise_not_hidden_by_adjacent_denial():
    review = MonitorAgent().check("不能保证到账，但我们赔偿您100元。", "售后问题", [])
    assert review.decision == "REJECT"


def test_monitor_retry_retains_independent_task_and_reexecutes_consumers(monkeypatch):
    import devma.orchestration.graph as module
    from devma.orchestration.contracts import MonitorOutput

    tasks = [
        task("a"),
        task("b", "after_sales", deps=("a",)),
        task("c", order_id="#5678"),
    ]
    route = RouterOutput(
        intent="multi_intent", confidence=0.9, target_solver="logistics", tasks=tasks
    )
    monkeypatch.setattr(module.router, "route", lambda *a, **kw: route)
    seen, original = [], LogisticsSolver.solve

    def counted(self, text, entities):
        seen.append(entities["order_id"])
        return original(self, text, entities)

    monkeypatch.setattr(LogisticsSolver, "solve", counted)
    rejected = []

    def review(output, *args, **kw):
        if output.startswith("订单 #1234") and not rejected:
            rejected.append(True)
            return MonitorOutput(
                decision="NEEDS_FIX", reason="test repair", retry_feedback="修正解释"
            )
        return MonitorOutput(decision="PASS", reason="ok")

    monkeypatch.setattr(module.monitor, "check", review)
    result = build_graph().invoke(
        {"messages": [{"role": "user", "content": "多个查询"}]}, config("retry")
    )
    assert seen.count("#1234") == 2
    assert seen.count("#5678") == 1
    assert result["solver_retry_count"] == 1
    assert len(result["repair_history"][0]["invalidated"]) == 2
    assert len(result["repair_history"][0]["retained"]) == 1


def test_repeated_monitor_rejection_stops_after_two_repairs(monkeypatch):
    import devma.orchestration.graph as module
    from devma.orchestration.contracts import MonitorOutput

    monkeypatch.setattr(
        module.monitor,
        "check",
        lambda *a, **kw: MonitorOutput(
            decision="NEEDS_FIX", reason="cannot verify", retry_feedback="修正"
        ),
    )
    result = build_graph().invoke(
        {"messages": [{"role": "user", "content": "手机#1234到哪了"}]},
        config("max-retry"),
    )
    assert result["solver_retry_count"] == 2
    assert result["handoff_needed"]


def test_semantically_wrong_solver_routes_back_to_router(monkeypatch):
    import devma.orchestration.graph as module

    wrong = task("t1", solver="after_sales")
    wrong["span"] = "手机#1234到哪了"
    right = task("t1")
    right["span"] = wrong["span"]
    routes = iter(
        [
            RouterOutput(
                intent="after_sales",
                confidence=0.9,
                target_solver="after_sales",
                tasks=[wrong],
            ),
            RouterOutput(
                intent="logistics",
                confidence=0.9,
                target_solver="logistics",
                tasks=[right],
            ),
        ]
    )
    monkeypatch.setattr(module.router, "route", lambda *a, **kw: next(routes))
    result = build_graph().invoke(
        {"messages": [{"role": "user", "content": wrong["span"]}]}, config("reroute")
    )
    assert result["active_solver"] == "logistics"
    assert result["solver_retry_count"] == 1
    assert result["repair_history"][0]["kind"] == "router_replan"


def test_verbose_execution_is_not_invoked_twice(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from typer.testing import CliRunner

    import devma.main as main

    @contextmanager
    def isolated_graph():
        with open_graph(str(tmp_path / "cli.db")) as graph:
            yield graph

    monkeypatch.setattr(main, "open_graph", isolated_graph)
    monkeypatch.setattr(main, "save_evidence", lambda *args: "test-trace")
    original, seen = LogisticsSolver.solve, []

    def counted(self, text, entities):
        seen.append(True)
        return original(self, text, entities)

    monkeypatch.setattr(LogisticsSolver, "solve", counted)
    result = CliRunner().invoke(main.app, ["chat", "手机#1234到哪了", "--verbose"])
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
