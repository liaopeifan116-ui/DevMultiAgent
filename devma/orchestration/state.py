"""LangGraph state definition for the agent workflow."""

from typing import Annotated, Any

from langgraph.graph import MessagesState


def merge_task_results(current: dict, updates: dict) -> dict:
    """A worker may publish only its task result, not replace the whole ledger."""
    merged = dict(current)
    for task_id, result in updates.items():
        if result.get("task_id") != task_id:
            raise ValueError("result key must match task_id")
        if task_id in merged and merged[task_id] != result:
            raise ValueError(
                f"conflicting writes for task {task_id}; invalidate before retry"
            )
        merged[task_id] = result
    return merged


class AgentState(MessagesState):
    request_id: str
    tasks: list[dict[str, Any]]
    task_results: Annotated[dict[str, dict[str, Any]], merge_task_results]
    task_feedback: dict[str, str]
    repair_task_ids: list[str]
    route_repair_needed: bool
    repair_history: list[dict[str, Any]]
    pause_before_execution: bool
    pause_on_missing_info: bool
    execution_approved: bool
    evidence: list[dict[str, Any]]
    handoff_context: dict[str, Any]
    llm_used: bool
    intent: str
    confidence: float
    entities: dict[str, str]
    user_context: dict[str, Any]
    sub_intents: list[dict[str, Any]]
    execution_mode: str
    target_solver: str
    transfer_count: int
    solver_retry_count: int
    solver_plan: list[str]
    solver_status: str
    active_solver: str
    solver_output: str
    tool_calls_made: list[dict[str, Any]]
    monitor_result: str
    monitor_reason: str
    monitor_checks: dict[str, Any]
    monitor_feedback: str
    final_output: str
    handoff_needed: bool
    metrics: dict[str, Any]
