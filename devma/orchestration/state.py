"""LangGraph state definition for the agent workflow."""
from typing import Any
from langgraph.graph import MessagesState


class AgentState(MessagesState):
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
