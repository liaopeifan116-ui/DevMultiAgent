"""Runtime-validated input/output contracts for Agent boundaries.

These Pydantic models are DTOs, not Agents. Agent classes produce them and
LangGraph nodes copy their validated fields into the shared AgentState.
"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class ContractModel(BaseModel):
    """Small mapping-compatible base for gradual adoption in existing nodes."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class RouterOutput(ContractModel):
    """Validated routing decision produced by RouterAgent."""

    intent: Literal["tech_support", "after_sales", "logistics", "unknown", "multi_intent", "general_chat"]
    confidence: float = Field(ge=0, le=1)
    target_solver: Literal["tech_support", "after_sales", "logistics", "handoff", "general_chat"]
    entities: dict[str, str] = Field(default_factory=dict)
    sub_intents: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal["single", "parallel", "sequential"] = "single"
    reason: str = ""
    raw_response: str = ""


class SolverMetrics(ContractModel):
    """Token and latency evidence produced by a Solver."""

    exec_tokens: int = Field(default=0, ge=0)
    exec_latency_ms: float = Field(default=0.0, ge=0)


class SolverOutput(ContractModel):
    """Common output protocol for current and future Solver Agents."""

    status: Literal["completed", "transfer", "failed"]
    active_solver: str
    plan: list[str] = Field(default_factory=list)
    output: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    metrics: SolverMetrics = Field(default_factory=SolverMetrics)
    llm_used: bool = False


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
