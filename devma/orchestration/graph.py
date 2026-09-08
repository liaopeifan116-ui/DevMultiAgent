"""LangGraph state machine: Router -> Solver -> Monitor -> Retry/Handoff."""
import time
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from devma.orchestration.state import AgentState
from devma.agents.router import RouterAgent
from devma.agents.solver import get_solver
from devma.agents.monitor import MonitorAgent
from devma.models.llm import DeepSeekLLM
from devma.config import MAX_SOLVER_RETRY, EVIDENCE_DIR
from devma.tools.kb_search import get_kb_tool
from devma.tools.privacy import redact_sensitive_fields

router = RouterAgent()
monitor = MonitorAgent()


def router_node(state: AgentState) -> dict:
    msgs = state["messages"]
    user_msg = msgs[-1].content if msgs else ""

    user_context = state.get("user_context", {})
    result = router.route(user_msg, user_context=user_context)
    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "entities": result.entities,
        "sub_intents": result.sub_intents,
        "execution_mode": result.execution_mode,
        "target_solver": result.target_solver,
        "user_context": user_context,
        "transfer_count": state.get("transfer_count", 0),
        "solver_retry_count": 0,
        "metrics": {
            **state.get("metrics", {}),
            "router_intent": result.intent,
            "router_confidence": result.confidence,
            "execution_mode": result.execution_mode,
            "sub_intent_count": len(result.sub_intents),
        },
    }


def solver_node(state: AgentState) -> dict:
    msgs = state["messages"]
    user_msg = msgs[-1].content if msgs else ""
    entities = state.get("entities", {})

    sub_intents = state.get("sub_intents") or [{
        "target_solver": state.get("target_solver", "after_sales"),
        "entities": entities,
        "span": user_msg,
        "sub_intent": state.get("intent", "after_sales"),
    }]
    execution_mode = state.get("execution_mode", "single")
    results = _run_solvers(sub_intents, user_msg, execution_mode)
    merged_output = _merge_solver_outputs(results)
    plan = [step for result in results for step in result["plan"]]
    tool_calls = [call for result in results for call in result["tool_calls"]]
    active_solver = "+".join(dict.fromkeys(result["active_solver"] for result in results))
    status = "completed" if all(result["status"] == "completed" for result in results) else "failed"

    return {
        "solver_status": status,
        "active_solver": active_solver,
        "solver_output": merged_output,
        "solver_plan": plan,
        "tool_calls_made": tool_calls,
        "llm_used": False,
        "metrics": {
            **state.get("metrics", {}),
            "solver_tool_calls": len(tool_calls),
            "solver_plan_steps": len(plan),
            "solver_tokens": sum(result["metrics"].get("plan_tokens", 0) + result["metrics"].get("exec_tokens", 0) for result in results),
            "llm_used": any(result.get("llm_used") for result in results),
        },
    }


def decide_after_router(state: AgentState) -> str:
    if state.get("target_solver") == "general_chat":
        return "general_chat"
    if state.get("intent") == "unknown" or state.get("target_solver") == "handoff":
        return "clarify"
    return "solver"


def clarify_node(state: AgentState) -> dict:
    return {
        "solver_status": "completed",
        "active_solver": "clarify",
        "solver_output": "您好，我可以帮您处理售后退款、故障排查或物流查询。请您补充一下具体问题，例如订单号、商品名称或遇到的故障现象。",
        "solver_plan": ["识别为问候或信息不足", "不调用业务工具", "请求用户补充具体问题"],
        "tool_calls_made": [],
        "monitor_result": "PASS",
        "monitor_reason": "Clarification response bypasses business tools",
        "monitor_checks": {"clarify": {"hit": True}},
        "final_output": "您好，我可以帮您处理售后退款、故障排查或物流查询。请您补充一下具体问题，例如订单号、商品名称或遇到的故障现象。",
        "handoff_needed": False,
    }


def monitor_node(state: AgentState) -> dict:
    solver_output = state.get("solver_output", "")
    user_msg = state["messages"][-1].content if state["messages"] else ""
    kb_tool = get_kb_tool()
    kb_results = kb_tool.search(solver_output[:500], top_k=3)
    kb_context = [r["content"] for r in kb_results]
    result = monitor.check(solver_output, user_msg, kb_context, tool_context=state.get("tool_calls_made", []))
    return {
        "monitor_result": result["result"],
        "monitor_reason": result["reason"],
        "monitor_checks": result["checks"],
        "monitor_feedback": result["retry_feedback"],
        "metrics": {
            **state.get("metrics", {}),
            "monitor_result": result["result"],
        },
    }


def decide_after_monitor(state: AgentState) -> str:
    result = state.get("monitor_result", "PASS")
    retry_count = state.get("solver_retry_count", 0)
    # Check 4 escalation: handoff even if monitor_result is PASS
    checks = state.get("monitor_checks", {})
    if checks.get("check4_escalation", {}).get("handoff_recommended"):
        return "handoff"
    if result == "PASS":
        return "respond"
    if result == "NEEDS_FIX" and retry_count < MAX_SOLVER_RETRY:
        return "retry_solver"
    return "handoff"


def retry_solver_node(state: AgentState) -> dict:
    msgs = state["messages"]
    user_msg = msgs[-1].content if msgs else ""
    entities = state.get("entities", {})
    reason = state.get("monitor_reason", "")
    enhanced_msg = f"{user_msg}\n\n[Quality feedback: {reason}. Please revise your response.]"
    result = get_solver(state.get("target_solver", "after_sales")).solve(enhanced_msg, entities)
    return {
        "solver_status": result["status"],
        "active_solver": result["active_solver"],
        "solver_output": result["output"],
        "solver_plan": result["plan"],
        "tool_calls_made": state.get("tool_calls_made", []) + result["tool_calls"],
        "llm_used": result.get("llm_used", False),
        "solver_retry_count": state.get("solver_retry_count", 0) + 1,
    }


def respond_node(state: AgentState) -> dict:
    return {"final_output": redact_sensitive_fields(state.get("solver_output", "")), "handoff_needed": False}


def handoff_node(state: AgentState) -> dict:
    context = {
        "user_summary": redact_sensitive_fields(f"用户咨询: {state['messages'][-1].content if state['messages'] else ''}"),
        "intent": state.get("intent", ""),
        "confidence": state.get("confidence", 0),
        "actions_taken": [tc["tool"] for tc in state.get("tool_calls_made", [])],
        "blocker": state.get("monitor_reason", ""),
        "key_entities": redact_sensitive_fields(state.get("entities", {})),
    }
    handoff_msg = (
        "很抱歉，我暂时无法完全解决您的问题。"
        "已将您的咨询转接给人工客服，以下是处理摘要：\n\n"
        f"`{json.dumps(context, ensure_ascii=False, indent=2)}`"
    )
    return {
        "final_output": redact_sensitive_fields(handoff_msg),
        "handoff_needed": True,
        "handoff_context": context,
    }


def general_chat_node(state: AgentState) -> dict:
    """Handle non-business questions with a free LLM chat, no business tools."""
    msgs = state["messages"]
    user_msg = msgs[-1].content if msgs else ""

    llm = DeepSeekLLM()
    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": "你是通用AI助手。用简洁中文回答用户问题。不要编造订单信息或售后政策。"},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        output = resp.content.strip()
    except Exception:
        output = "抱歉，我暂时无法回答这个问题。您可以尝试问我关于订单、售后或物流的问题。"

    return {
        "intent": "general_chat",
        "confidence": 1.0,
        "solver_status": "completed",
        "active_solver": "general_chat",
        "solver_output": output,
        "final_output": output,
        "solver_plan": ["识别为非业务问题", "LLM自由回复"],
        "tool_calls_made": [],
        "monitor_result": "PASS",
        "monitor_reason": "General chat bypasses business checks",
        "monitor_checks": {},
        "handoff_needed": False,
        "metrics": {
            "router_intent": "general_chat",
            "router_confidence": 1.0,
            "execution_mode": "single",
            "llm_used": True,
        },
    }


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("general_chat", general_chat_node)
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("retry_solver", retry_solver_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("handoff", handoff_node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router", decide_after_router,
        {"solver": "solver", "clarify": "clarify", "general_chat": "general_chat"},
    )
    workflow.add_edge("solver", "monitor")
    workflow.add_conditional_edges(
        "monitor", decide_after_monitor,
        {"respond": "respond", "retry_solver": "retry_solver", "handoff": "handoff"},
    )
    workflow.add_edge("retry_solver", "monitor")
    workflow.add_edge("respond", END)
    workflow.add_edge("handoff", END)
    workflow.add_edge("clarify", END)
    workflow.add_edge("general_chat", END)
    return workflow.compile(checkpointer=MemorySaver())


def _run_solvers(sub_intents: list[dict], user_msg: str, execution_mode: str) -> list[dict]:
    if execution_mode == "parallel" and len(sub_intents) > 1:
        with ThreadPoolExecutor(max_workers=min(len(sub_intents), 4)) as pool:
            futures = [pool.submit(_run_one_solver, item, user_msg) for item in sub_intents]
            return [future.result() for future in futures]

    results = []
    context = ""
    for item in sub_intents:
        span = item.get("span") or user_msg
        if context:
            span = f"{span}\n\n上一 Solver 结论：{context}"
        result = _run_one_solver(item, span)
        results.append(result)
        context = result["output"]
    return results


def _run_one_solver(sub_intent: dict, fallback_message: str) -> dict:
    solver_name = sub_intent.get("target_solver", "after_sales")
    message = sub_intent.get("span") or fallback_message
    entities = sub_intent.get("entities") or {}
    return get_solver(solver_name).solve(message, entities)


def _merge_solver_outputs(results: list[dict]) -> str:
    if len(results) == 1:
        return results[0]["output"]
    labels = {
        "after_sales": "关于售后",
        "tech_support": "关于故障排查",
        "logistics": "关于物流",
    }
    parts = []
    for result in results:
        label = labels.get(result["active_solver"], result["active_solver"])
        parts.append(f"{label}：{result['output']}")
    return "\n\n".join(parts)


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
        json.dump({
            "run_id": run_id, "timestamp": ts,
            "intent": state.get("intent"), "confidence": state.get("confidence"),
            "entities": redact_sensitive_fields(state.get("entities")), "plan": state.get("solver_plan"),
            "output": redact_sensitive_fields(state.get("final_output")),
            "monitor_result": state.get("monitor_result"),
            "monitor_reason": state.get("monitor_reason"),
            "monitor_checks": redact_sensitive_fields({k: str(v) for k, v in state.get("monitor_checks", {}).items()}),
            "handoff": state.get("handoff_needed"),
            "metrics": state.get("metrics"),
        }, f, ensure_ascii=False, indent=2, default=str)

    tool_path = EVIDENCE_DIR / "logs" / f"tools_{ts}_{short_id}.json"
    with open(tool_path, "w", encoding="utf-8") as f:
        json.dump(redact_sensitive_fields(state.get("tool_calls_made", [])), f, ensure_ascii=False, indent=2, default=str)

    if state.get("monitor_result") in ("NEEDS_FIX", "REJECT"):
        reject_path = EVIDENCE_DIR / "logs" / f"reject_{ts}_{short_id}.json"
        with open(reject_path, "w", encoding="utf-8") as f:
            json.dump({
                "result": state.get("monitor_result"),
                "reason": state.get("monitor_reason"),
                "checks": {k: str(v) for k, v in state.get("monitor_checks", {}).items()},
                "output_snippet": state.get("final_output", "")[:500],
            }, f, ensure_ascii=False, indent=2)

    return str(trace_path)
