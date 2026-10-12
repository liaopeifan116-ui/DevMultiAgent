"""Runtime-validated input/output contracts for Agent boundaries.

These Pydantic models are DTOs, not Agents. Agent classes produce them and
LangGraph nodes copy their validated fields into the shared AgentState.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SubTask(BaseModel):
    """A bounded, read-only business task and its explicit DAG dependencies."""

    task_id: str = Field(min_length=1, max_length=120)
    intent: Literal["tech_support", "after_sales", "logistics"]
    target_solver: Literal["tech_support", "after_sales", "logistics"]
    span: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    entities: dict[str, str] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    resource_keys: list[str] = Field(default_factory=list)
    access: Literal["read", "write"] = "read"

    @model_validator(mode="after")
    def consistent_target(self):
        if self.intent != self.target_solver:
            raise ValueError("intent and target_solver must agree")
        if self.task_id in self.depends_on:
            raise ValueError("a task cannot depend on itself")
        return self


def validate_dependencies(tasks: list[SubTask]) -> None:
    """Reject duplicates, dangling edges and cycles before dispatch."""
    ids = {task.task_id for task in tasks}
    if len(ids) != len(tasks):
        raise ValueError("duplicate task_id")
    remaining = {task.task_id: set(task.depends_on) for task in tasks}
    if any(not deps <= ids for deps in remaining.values()):
        raise ValueError("dependency references a missing task")
    while remaining:
        ready = {key for key, deps in remaining.items() if not deps}
        if not ready:
            raise ValueError("task dependencies contain a cycle")
        remaining = {
            key: deps - ready for key, deps in remaining.items() if key not in ready
        }


class PlanStep(BaseModel):
    action: Literal["lookup_order", "retrieve_policy", "check_eligibility"]
    reason: str = ""


class AfterSalesPlan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def safe_order(self):
        actions = [step.action for step in self.steps]
        if actions == ["retrieve_policy"]:
            return self  # General policy questions do not need private order facts.
        if set(actions) != {"lookup_order", "retrieve_policy", "check_eligibility"}:
            raise ValueError("plan must include each allowed action exactly once")
        if actions.index("lookup_order") > actions.index("check_eligibility"):
            raise ValueError("eligibility requires order facts")
        return self


class TechDecision(BaseModel):
    action: Literal["search_kb", "finish", "ask_user", "handoff"]
    query: str = ""
    reason: str = ""


class ContractModel(BaseModel):
    """Small mapping-compatible base for gradual adoption in existing nodes."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class RouterOutput(ContractModel):
    """Validated routing decision produced by RouterAgent."""

    intent: Literal[
        "tech_support",
        "after_sales",
        "logistics",
        "unknown",
        "multi_intent",
        "general_chat",
    ]
    confidence: float = Field(ge=0, le=1)
    target_solver: Literal[
        "tech_support", "after_sales", "logistics", "handoff", "general_chat"
    ]
    entities: dict[str, str] = Field(default_factory=dict)
    sub_intents: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal["single", "parallel", "sequential"] = "single"
    reason: str = ""
    raw_response: str = ""
    tasks: list[SubTask] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def valid_task_graph(self):
        validate_dependencies(self.tasks)
        return self


class SolverMetrics(ContractModel):
    """Token and latency evidence produced by a Solver."""

    exec_tokens: int = Field(default=0, ge=0)
    exec_latency_ms: float = Field(default=0.0, ge=0)


class SolverOutput(ContractModel):
    """Common output protocol for current and future Solver Agents."""

    status: Literal["completed", "transfer", "failed", "needs_input", "blocked"]
    active_solver: str
    plan: list[str] = Field(default_factory=list)
    output: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    metrics: SolverMetrics = Field(default_factory=SolverMetrics)
    llm_used: bool = False
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class MonitorOutput(ContractModel):
    """Validated quality decision produced by MonitorAgent."""

    decision: Literal["PASS", "NEEDS_FIX", "REJECT", "HANDOFF"]
    reason: str
    checks: dict[str, Any] = Field(default_factory=dict)
    retry_feedback: str = ""

    def __getitem__(self, key: str) -> Any:
        if key == "result":
            return self.decision
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "result":
            return self.decision
        return super().get(key, default)
