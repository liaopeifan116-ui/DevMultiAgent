"""Router -> dependency scheduler -> Solver subgraphs -> per-task Monitor."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import Overwrite, Send, interrupt

from devma.agents.monitor import MonitorAgent
from devma.agents.router import RouterAgent
from devma.agents.solver import get_solver
from devma.config import (
    CHECKPOINT_DB_PATH,
    DEEPSEEK_API_KEY,
    EVIDENCE_DIR,
    MAX_SOLVER_RETRY,
)
from devma.models.llm import DeepSeekLLM
from devma.orchestration.contracts import SolverOutput, SubTask, validate_dependencies
from devma.orchestration.conversation import finalize_turn, prepare_turn
from devma.orchestration.scheduler import (
    blocked_results,
    invalidated_tasks,
    ready_tasks,
)
from devma.orchestration.state import AgentState
from devma.tools.privacy import redact_sensitive_fields

router = RouterAgent()
monitor = MonitorAgent()


def _user_message(state: dict) -> str:
    return next(
        (m.content for m in reversed(state.get("messages", [])) if m.type == "human"),
        "",
    )


def router_node(state: AgentState) -> dict:
    user_msg = _user_message(state)
    request_id = uuid.uuid4().hex[:12]
    prepared = prepare_turn(router, state, user_msg, request_id)
    tasks, route = prepared.pop("tasks"), prepared.pop("route")
    cached = prepared.pop("cached")
    if tasks and (not route or route.tasks or prepared["turn_task_ids"]):
        focus = next(
            (t for t in tasks if t["task_id"] in prepared["turn_task_ids"]), tasks[0]
        )
        result = router._result(
            [SubTask.model_validate(t) for t in tasks],
            focus["entities"],
            "Cross-turn execution projection",
        )
        tasks = [t.model_dump() for t in result.tasks]
    elif route:
        result = route
    else:
        from devma.orchestration.contracts import RouterOutput

        result = RouterOutput(intent="unknown", confidence=1, target_solver="handoff")
    validate_dependencies([SubTask.model_validate(task) for task in tasks])
    return {
        **prepared,
        "request_id": request_id,
        "tasks": tasks,
        "task_results": Overwrite(cached),
        "task_feedback": {},
        "repair_task_ids": [],
        "repair_history": [],
        "route_repair_needed": False,
        "intent": result.intent,
        "confidence": result.confidence,
        "entities": result.entities,
        "sub_intents": tasks,
        "execution_mode": result.execution_mode,
        "target_solver": result.target_solver,
        "solver_retry_count": 0,
        "transfer_count": 0,
        "tool_calls_made": [],
        "evidence": [],
        "solver_plan": [],
        "solver_output": "",
        "final_output": "",
        "handoff_needed": False,
        "monitor_result": "",
        "monitor_checks": {},
        "monitor_reason": "",
        "monitor_feedback": "",
        "execution_approved": False,
        "metrics": {
            "router_intent": result.intent,
            "router_confidence": result.confidence,
            "execution_mode": result.execution_mode,
            "sub_intent_count": len(tasks),
        },
    }


def decide_after_router(state: AgentState) -> str:
    if state.get("turn_clarification") and (
        not state.get("turn_task_ids") or not state.get("tasks")
    ):
        return "conversation_reply"
    if state.get("target_solver") == "general_chat":
        return "general_chat"
    if state.get("intent") == "unknown" or state.get("target_solver") == "handoff":
        return "clarify"
    return "execution_gate"


def execution_gate(state: AgentState) -> dict:
    if not state.get("pause_before_execution"):
        return {"execution_approved": True}
    approval = interrupt(
        {
            "kind": "before_execution",
            "tasks": redact_sensitive_fields(state["tasks"]),
            "expected": {"approved": True},
        }
    )
    approved = isinstance(approval, dict) and approval.get("approved") is True
    return {
        "execution_approved": approved,
        "monitor_reason": "" if approved else "用户未批准执行",
    }


def schedule_node(state: AgentState) -> dict:
    return {
        "task_results": blocked_results(state["tasks"], state.get("task_results", {}))
    }


def dispatch_ready(state: AgentState):
    ready = ready_tasks(state["tasks"], state.get("task_results", {}))
    if not ready:
        return "monitor"
    return [
        Send(
            f"run_{task['target_solver']}",
            {
                "task": task,
                "dependency_results": {
                    dep: state["task_results"][dep] for dep in task["depends_on"]
                },
                "feedback": state.get("task_feedback", {}).get(task["task_id"], ""),
                "followups": state.get("task_ledger", {})
                .get(task["task_id"], {})
                .get("followups", []),
                "previous_result": state.get("task_ledger", {})
                .get(task["task_id"], {})
                .get("result_history", []),
                "turn_number": state.get("turn_number", 0),
                "pause_on_missing_info": state.get("pause_on_missing_info", False),
            },
        )
        for task in ready
    ]


def _worker(solver_name: str):
    agent = get_solver(solver_name)
    subgraph = agent.build_subgraph() if solver_name != "logistics" else None

    def run(state: dict) -> dict:
        task = SubTask.model_validate(state["task"])
        try:
            if task.target_solver != solver_name or task.access != "read":
                raise ValueError(
                    "worker target/access mismatch; write tools are not implemented"
                )
            if subgraph is not None:
                child = subgraph.invoke(state)
                result = SolverOutput.model_validate(child["payload"])
            else:
                result = agent.solve(task.span, task.entities)
            if result.active_solver != solver_name:
                raise ValueError("Solver returned a result for a different module")
            payload = result.model_dump()
        except GraphInterrupt:
            raise  # Let LangGraph persist and propagate the pause; never convert it to a failure.
        except Exception as exc:
            payload = SolverOutput(
                status="failed",
                active_solver=solver_name,
                output="该子任务执行失败，需进一步核实。",
                error=redact_sensitive_fields(str(exc)),
            ).model_dump()
        payload["task_id"] = task.task_id
        payload["revision"] = task.revision
        payload["executed_turn"] = state.get("turn_number", 0)
        payload["executed_at"] = time.time()
        payload["depends_on"] = task.depends_on
        payload["tool_calls"] = [
            {**call, "task_id": task.task_id} for call in payload["tool_calls"]
        ]
        payload["evidence"] = [
            {**item, "task_id": task.task_id} for item in payload["evidence"]
        ]
        return {"task_results": {task.task_id: payload}}

    return run


def monitor_node(state: AgentState) -> dict:
    task_results = state.get("task_results", {})
    if len(task_results) != len(state["tasks"]):
        return {
            "monitor_result": "HANDOFF",
            "monitor_reason": "依赖调度未能覆盖全部任务",
        }
    per_task, repair_ids, outputs, tool_calls, evidence, plan = {}, [], [], [], [], []
    decisions = []
    route_repair_needed = False
    user_message = _user_message(state)
    for task in state["tasks"]:
        result = task_results[task["task_id"]]
        if task["task_id"] in state.get("turn_task_ids", []) or result["status"] in {
            "needs_input",
            "blocked",
        }:
            outputs.append((task["target_solver"], result["output"]))
        cached = result.get("executed_turn") != state.get("turn_number")
        tool_calls.extend(
            {
                **call,
                **(
                    {"cached": True, "reused_from_turn": result.get("executed_turn")}
                    if cached
                    else {}
                ),
            }
            for call in result.get("tool_calls", [])
        )
        evidence.extend(result.get("evidence", []))
        plan.extend(result.get("plan", []))
        if result["status"] in {"failed", "transfer"}:
            decisions.append("HANDOFF")
            per_task[task["task_id"]] = {
                "decision": "HANDOFF",
                "reason": result.get("error") or "执行失败或需人工",
            }
            continue
        if result["status"] == "blocked":
            per_task[task["task_id"]] = {
                "decision": "BLOCKED",
                "reason": "前置任务未完成",
            }
            continue
        cached_review = (
            state.get("task_ledger", {}).get(task["task_id"], {}).get("review", {})
        )
        if (
            result.get("executed_turn") != state.get("turn_number")
            and cached_review.get("decision") == "PASS"
        ):
            per_task[task["task_id"]] = cached_review
            decisions.append("PASS")
            continue
        expected = router._rule_intent(task["span"])
        if expected != "unknown" and expected != task["target_solver"]:
            decisions.append("NEEDS_FIX")
            repair_ids.append(task["task_id"])
            route_repair_needed = True
            per_task[task["task_id"]] = {
                "decision": "NEEDS_FIX",
                "reason": f"任务类型与目标 Solver 不匹配，建议 {expected}",
                "retry_feedback": "重新识别任务并路由",
                "checks": {
                    "routing": {"expected": expected, "actual": task["target_solver"]}
                },
            }
            continue
        check = monitor.check(
            result["output"],
            task["span"] + "\n本轮用户消息：" + user_message,
            [r["content"] for r in result.get("evidence", [])],
            tool_context=result.get("tool_calls", []),
        )
        per_task[task["task_id"]] = check.model_dump()
        if result["status"] == "needs_input" and check.decision == "PASS":
            per_task[task["task_id"]]["decision"] = "WAITING"
        decisions.append(check.decision)
        if check.decision == "NEEDS_FIX":
            repair_ids.append(task["task_id"])
    decision = next(
        (d for d in ("REJECT", "HANDOFF", "NEEDS_FIX") if d in decisions), "PASS"
    )
    first_checks = next(
        (r.get("checks", {}) for r in per_task.values() if r.get("checks")), {}
    )
    labels = {
        "after_sales": "关于售后",
        "tech_support": "关于故障排查",
        "logistics": "关于物流",
    }
    output = (
        outputs[0][1]
        if len(outputs) == 1
        else "\n\n".join(f"{labels[name]}：{text}" for name, text in outputs)
    )
    reason = "；".join(
        r.get("reason", "") for r in per_task.values() if r.get("decision") != "PASS"
    )
    return {
        "solver_output": output,
        "solver_status": "completed" if decision == "PASS" else "failed",
        "active_solver": "+".join(
            dict.fromkeys(t["target_solver"] for t in state["tasks"])
        ),
        "solver_plan": plan,
        "tool_calls_made": tool_calls,
        "evidence": evidence,
        "monitor_result": decision,
        "monitor_reason": reason or "All checks passed",
        "monitor_checks": {**first_checks, "per_task": per_task},
        "repair_task_ids": repair_ids,
        "route_repair_needed": route_repair_needed,
        "monitor_feedback": reason,
        "metrics": {
            **state.get("metrics", {}),
            "solver_tool_calls": sum(not call.get("cached") for call in tool_calls),
            "solver_plan_steps": sum(
                len(r.get("plan", []))
                for r in task_results.values()
                if r.get("executed_turn") == state.get("turn_number")
            ),
            "solver_tokens": sum(
                r.get("metrics", {}).get("exec_tokens", 0)
                for r in task_results.values()
                if r.get("executed_turn") == state.get("turn_number")
            ),
            "llm_used": any(r.get("llm_used") for r in task_results.values()),
            "monitor_result": decision,
        },
    }


def decide_after_monitor(state: AgentState) -> str:
    if state.get("monitor_result") == "PASS":
        return "respond"
    if (
        state.get("monitor_result") == "NEEDS_FIX"
        and state.get("solver_retry_count", 0) < MAX_SOLVER_RETRY
    ):
        return "repair"
    return "handoff"


def repair_node(state: AgentState) -> dict:
    if state.get("route_repair_needed"):
        route = router.route(
            _user_message(state),
            user_context={
                **state.get("user_context", {}),
                "routing_feedback": state.get("monitor_feedback", ""),
            },
        )
        if not route.tasks:
            return {
                "solver_retry_count": MAX_SOLVER_RETRY,
                "route_repair_needed": False,
                "monitor_result": "HANDOFF",
                "monitor_reason": "重新路由后无有效业务任务",
            }
        prefix = state["request_id"]
        ids = {
            t.task_id: f"{prefix}:r{state.get('solver_retry_count', 0) + 1}:{t.task_id}"
            for t in route.tasks
        }
        tasks = [
            t.model_copy(
                update={
                    "task_id": ids[t.task_id],
                    "depends_on": [ids[d] for d in t.depends_on],
                }
            ).model_dump()
            for t in route.tasks
        ]
        ledger = {
            key: dict(value) for key, value in state.get("task_ledger", {}).items()
        }
        for previous in state["tasks"]:
            if previous["task_id"] in ledger:
                ledger[previous["task_id"]].update(
                    status="cancelled",
                    result=None,
                    question="",
                    superseded_by=list(ids.values()),
                )
        for task in tasks:
            ledger[task["task_id"]] = {
                **task,
                "status": "pending",
                "created_turn": state.get("turn_number", 0),
                "updated_turn": state.get("turn_number", 0),
                "followups": [],
                "continuations": 0,
                "result": None,
                "review": {},
            }
        return {
            "task_ledger": ledger,
            "turn_task_ids": list(ids.values()),
            "tasks": tasks,
            "sub_intents": tasks,
            "task_results": Overwrite({}),
            "task_feedback": {},
            "route_repair_needed": False,
            "intent": route.intent,
            "confidence": route.confidence,
            "entities": route.entities,
            "target_solver": route.target_solver,
            "execution_mode": route.execution_mode,
            "solver_retry_count": state.get("solver_retry_count", 0) + 1,
            "repair_history": state.get("repair_history", [])
            + [
                {
                    "kind": "router_replan",
                    "invalidated": list(state["task_results"]),
                    "retained": [],
                    "reason": state["monitor_reason"],
                }
            ],
        }
    invalid = invalidated_tasks(state["tasks"], state.get("repair_task_ids", []))
    # Retain unrelated successful tasks; remove stale downstream results/evidence.
    retained = {
        key: value for key, value in state["task_results"].items() if key not in invalid
    }
    feedback = dict(state.get("task_feedback", {}))
    for task_id in invalid:
        review = state.get("monitor_checks", {}).get("per_task", {}).get(task_id, {})
        feedback[task_id] = review.get("retry_feedback") or state.get(
            "monitor_feedback", ""
        )
    return {
        "task_results": Overwrite(retained),
        "task_feedback": feedback,
        "solver_retry_count": state.get("solver_retry_count", 0) + 1,
        "repair_history": state.get("repair_history", [])
        + [
            {
                "invalidated": sorted(invalid),
                "retained": sorted(retained),
                "reason": state.get("monitor_reason", ""),
            }
        ],
    }


def respond_node(state: AgentState) -> dict:
    output = redact_sensitive_fields(state.get("solver_output", ""))
    if state.get("turn_clarification"):
        output = "\n\n".join(filter(None, [output, state["turn_clarification"]]))
    return {
        "final_output": output,
        "messages": [AIMessage(content=output)],
        "handoff_needed": False,
    }


def handoff_node(state: AgentState) -> dict:
    per_task = state.get("monitor_checks", {}).get("per_task", {})
    approved = {
        key: result["output"]
        for key, result in state.get("task_results", {}).items()
        if per_task.get(key, {}).get("decision") == "PASS"
    }
    context = redact_sensitive_fields(
        {
            "user_summary": _user_message(state),
            "intent": state.get("intent"),
            "confidence": state.get("confidence"),
            "completed_results": approved,
            "blocker": state.get("monitor_reason", ""),
            "key_entities": state.get("entities", {}),
            "pending_tasks": [
                t["task_id"]
                for t in state.get("tasks", [])
                if t["task_id"] not in approved
            ],
        }
    )
    partial = "\n\n".join(redact_sensitive_fields(value) for value in approved.values())
    output = (
        partial + "\n\n" if partial else ""
    ) + "剩余问题需要人工进一步核实，已生成处理摘要。"
    # This flag is an integration boundary, NOT a claim that an external CRM was called.
    return {
        "final_output": output,
        "messages": [AIMessage(content=output)],
        "handoff_needed": True,
        "handoff_context": context,
    }


def clarify_node(state: AgentState) -> dict:
    output = "您好，我可以帮您处理售后退款、故障排查或物流查询。请您补充具体问题、订单号或故障现象。"
    return {
        "solver_status": "completed",
        "active_solver": "clarify",
        "solver_output": output,
        "final_output": output,
        "messages": [AIMessage(content=output)],
        "monitor_result": "PASS",
        "monitor_checks": {"clarify": {"hit": True}},
        "handoff_needed": False,
    }


def general_chat_node(state: AgentState) -> dict:
    output = "当前未接入实时资讯服务，无法确认天气等实时信息。请使用相应服务查询。"
    if DEEPSEEK_API_KEY:
        try:
            output = (
                DeepSeekLLM()
                .chat(
                    [
                        {
                            "role": "system",
                            "content": "用简洁中文回复，不编造实时信息、订单或售后政策。",
                        },
                        {"role": "user", "content": _user_message(state)},
                    ],
                    max_tokens=400,
                )
                .content
            )
        except Exception:
            pass
    return {
        "solver_output": output,
        "final_output": output,
        "messages": [AIMessage(content=output)],
        "monitor_result": "PASS",
        "handoff_needed": False,
        "active_solver": "general_chat",
    }


def conversation_reply_node(state: AgentState) -> dict:
    output = redact_sensitive_fields(
        state.get("turn_clarification") or "已保留当前任务，请补充对应信息。"
    )
    handoff = any(
        e["status"] == "handoff" and e.get("updated_turn") == state.get("turn_number")
        for e in state.get("task_ledger", {}).values()
    )
    return {
        "final_output": output,
        "solver_output": output,
        "messages": [AIMessage(content=output)],
        "monitor_result": "HANDOFF" if handoff else "PASS",
        "monitor_checks": {},
        "handoff_needed": handoff,
    }


def build_graph(checkpointer=None):
    """Memory by default for tests; open_graph() owns a durable SQLite connection."""
    workflow = StateGraph(AgentState)
    for name, node in {
        "router": router_node,
        "execution_gate": execution_gate,
        "scheduler": schedule_node,
        "monitor": monitor_node,
        "repair": repair_node,
        "respond": respond_node,
        "handoff": handoff_node,
        "clarify": clarify_node,
        "general_chat": general_chat_node,
        "conversation_reply": conversation_reply_node,
        "finalize_turn": finalize_turn,
        "run_tech_support": _worker("tech_support"),
        "run_after_sales": _worker("after_sales"),
        "run_logistics": _worker("logistics"),
    }.items():
        workflow.add_node(name, node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        decide_after_router,
        ["general_chat", "clarify", "execution_gate", "conversation_reply"],
    )
    workflow.add_conditional_edges(
        "execution_gate",
        lambda s: "scheduler" if s["execution_approved"] else "handoff",
        ["scheduler", "handoff"],
    )
    workflow.add_conditional_edges(
        "scheduler",
        dispatch_ready,
        ["run_tech_support", "run_after_sales", "run_logistics", "monitor"],
    )
    for name in ("run_tech_support", "run_after_sales", "run_logistics"):
        workflow.add_edge(name, "scheduler")
    workflow.add_conditional_edges(
        "monitor", decide_after_monitor, ["respond", "repair", "handoff"]
    )
    workflow.add_conditional_edges(
        "repair",
        lambda s: "handoff" if s.get("monitor_result") == "HANDOFF" else "scheduler",
        ["handoff", "scheduler"],
    )
    for name in ("respond", "handoff", "clarify", "general_chat", "conversation_reply"):
        workflow.add_edge(name, "finalize_turn")
    workflow.add_edge("finalize_turn", END)
    return workflow.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )


@contextmanager
def open_graph(db_path: str = CHECKPOINT_DB_PATH):
    """Close SQLite cleanly; reuse thread_id and Command(resume=...) after restart."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(db_path) as saver:
        yield build_graph(checkpointer=saver)


def save_evidence(state: dict, run_id: str = "") -> str:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["traces", "logs", "reports"]:
        (EVIDENCE_DIR / sub).mkdir(exist_ok=True)
    if not run_id:
        run_id = str(uuid.uuid4())
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_id = run_id[:8]

    trace_path = EVIDENCE_DIR / "traces" / f"trace_{ts}_{short_id}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "timestamp": ts,
                "intent": state.get("intent"),
                "confidence": state.get("confidence"),
                "entities": redact_sensitive_fields(state.get("entities")),
                "plan": state.get("solver_plan"),
                "output": redact_sensitive_fields(state.get("final_output")),
                "monitor_result": state.get("monitor_result"),
                "monitor_reason": state.get("monitor_reason"),
                "monitor_checks": redact_sensitive_fields(
                    {k: str(v) for k, v in state.get("monitor_checks", {}).items()}
                ),
                "handoff": state.get("handoff_needed"),
                "metrics": state.get("metrics"),
                "tasks": redact_sensitive_fields(state.get("tasks", [])),
                "task_results": redact_sensitive_fields(state.get("task_results", {})),
                "task_ledger": redact_sensitive_fields(state.get("task_ledger", {})),
                "turn_number": state.get("turn_number"),
                "task_events": redact_sensitive_fields(state.get("task_events", [])),
                "evidence": redact_sensitive_fields(state.get("evidence", [])),
                "repair_history": redact_sensitive_fields(
                    state.get("repair_history", [])
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    tool_path = EVIDENCE_DIR / "logs" / f"tools_{ts}_{short_id}.json"
    with open(tool_path, "w", encoding="utf-8") as f:
        json.dump(
            redact_sensitive_fields(state.get("tool_calls_made", [])),
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    if state.get("monitor_result") in ("NEEDS_FIX", "REJECT"):
        reject_path = EVIDENCE_DIR / "logs" / f"reject_{ts}_{short_id}.json"
        with open(reject_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "result": state.get("monitor_result"),
                    "reason": state.get("monitor_reason"),
                    "checks": {
                        k: str(v) for k, v in state.get("monitor_checks", {}).items()
                    },
                    "output_snippet": state.get("final_output", "")[:500],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return str(trace_path)
