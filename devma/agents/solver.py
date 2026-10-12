"""Solver agents for after-sales, tech support, and logistics tasks."""

from __future__ import annotations

import time
from typing import Any

from devma.config import DEEPSEEK_API_KEY
from devma.models.llm import DeepSeekLLM
from devma.orchestration.contracts import SolverMetrics, SolverOutput
from devma.tools.order_query import (
    find_orders_by_text,
    query_order,
    query_orders_by_product,
)


class BaseSolver:
    """Shared helper methods for current Solver implementations."""

    active_solver = "after_sales"

    def __init__(self):
        # Kept as a compatibility extension point for tests and future model
        # powered variants. The MVP path remains deterministic without API keys.
        self.llm = DeepSeekLLM()

    def _tool_call(self, tool: str, tool_input: Any, result: Any) -> dict[str, Any]:
        return {
            "tool": tool,
            "input": tool_input,
            "result": result,
            "timestamp": time.time(),
        }

    def _metrics(self, started_at: float) -> SolverMetrics:
        return SolverMetrics(
            exec_latency_ms=round((time.perf_counter() - started_at) * 1000, 1)
        )

    def _generate_with_llm(
        self, system: str, user: str, fallback: str
    ) -> tuple[str, int, bool]:
        if not DEEPSEEK_API_KEY:
            return fallback, 0, False
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.25,
                max_tokens=700,
            )
            text = resp.content.strip()
            return (text or fallback), resp.tokens.total, True
        except Exception:
            return fallback, 0, False


class AfterSalesSolver(BaseSolver):
    """Plan-and-Execute over a small, code-validated read-only action set."""

    active_solver = "after_sales"

    def build_subgraph(self):
        from devma.orchestration.subgraphs import build_after_sales_subgraph

        return build_after_sales_subgraph(self)

    def solve(self, user_message: str, entities: dict) -> SolverOutput:
        state = self.build_subgraph().invoke(
            {"user_message": user_message, "entities": entities}
        )
        return SolverOutput.model_validate(state["payload"])

    def _build_after_sales_reply(
        self,
        order: dict | None,
        eligibility: dict | None,
        kb_results: list[dict],
        order_id: str,
    ) -> str:
        if not order_id:
            return "请提供您的订单号，这样我可以帮您查询具体的售后状态。您可以在订单详情页找到订单号（如 #12345）。"
        if not order:
            return f"我还没有查到订单 {order_id} 的记录。请您核对订单号，或补充购买手机号后转人工继续核实。"

        product = order.get("product", "商品")
        days_text = (
            eligibility.get("reason", "") if eligibility else "暂时无法判断售后资格。"
        )
        source = kb_results[0].get("source", "售后政策") if kb_results else "售后政策"
        action = eligibility.get("action", "") if eligibility else "handoff"

        if eligibility and eligibility.get("eligible"):
            next_step = "可以提交退货申请，退款金额以订单实际支付金额为准。"
        elif action == "approve_exchange_or_repair":
            next_step = "建议申请换货或维修，我可以继续帮您整理售后材料。"
        elif action == "handoff":
            next_step = "该情况需要人工进一步核实，我会保留当前处理摘要。"
        else:
            next_step = (
                "当前不满足退款条件，如您认为商品存在特殊质量问题，建议转人工复核。"
            )

        return (
            f"已为您查询订单 {order.get('id')}，商品为 {product}。"
            f"{days_text}依据 {source} 的售后规则，{next_step}"
        )


class TechSupportSolver(BaseSolver):
    """A bounded ReAct graph: think -> read-only action -> observe -> think."""

    active_solver = "tech_support"

    def build_subgraph(self):
        from devma.orchestration.subgraphs import build_tech_subgraph

        return build_tech_subgraph(self)

    def solve(self, user_message: str, entities: dict) -> SolverOutput:
        state = self.build_subgraph().invoke(
            {"user_message": user_message, "entities": entities}
        )
        return SolverOutput.model_validate(state["payload"])


class LogisticsSolver(BaseSolver):
    """Simple ToolUse solver for order and logistics status."""

    active_solver = "logistics"

    def solve(self, user_message: str, entities: dict) -> SolverOutput:
        started_at = time.perf_counter()
        order_id = entities.get("order_id") or _infer_order_id(user_message, default="")
        product = entities.get("product") or _infer_product(user_message)
        # If no order_id, try matching product name from user message against DB
        if not order_id and not product:
            by_text = find_orders_by_text(user_message)
            if len(by_text) == 1:
                order_id = by_text[0]["id"]
                product = by_text[0]["product"]
        if not order_id:
            if product:
                matches = query_orders_by_product(product)
                if len(matches) == 1:
                    order_id = matches[0]["id"]
                elif len(matches) > 1:
                    summary = "；".join(
                        f"{item['id']}（{item['product']}，{item['status']}）"
                        for item in matches[:3]
                    )
                    return SolverOutput(
                        status="needs_input",
                        active_solver="logistics",
                        plan=[
                            "识别物流/订单查询意图",
                            f"按商品关键词查询：{product}",
                            "发现多个候选订单，请用户确认",
                        ],
                        output=f"我查到多个和“{product}”相关的订单：{summary}。请您确认要查询哪一个订单。",
                        tool_calls=[
                            self._tool_call(
                                "query_orders_by_product", product, matches[:3]
                            )
                        ],
                        metrics=self._metrics(started_at),
                    )
                else:
                    return SolverOutput(
                        status="needs_input",
                        active_solver="logistics",
                        plan=[
                            "识别物流/订单查询意图",
                            f"按商品关键词查询：{product}",
                            "未找到匹配订单",
                        ],
                        output=f"我没有查到和“{product}”匹配的订单。请您补充订单号或购买账号，我再继续查。",
                        tool_calls=[
                            self._tool_call("query_orders_by_product", product, [])
                        ],
                        metrics=self._metrics(started_at),
                    )
            else:
                plan = [
                    "识别物流/订单查询意图",
                    "缺少订单号和商品信息",
                    "请求用户补充订单号、商品或购买账号",
                ]
                output = "我可以帮您查询订单或物流状态。请您补充订单号，或说明要查询的商品和购买账号。"
                return SolverOutput(
                    status="needs_input",
                    active_solver="logistics",
                    plan=plan,
                    output=output,
                    tool_calls=[],
                    metrics=self._metrics(started_at),
                )
        order = query_order(order_id) if order_id else None
        tool_calls = [self._tool_call("get_order", order_id, order)]
        plan = ["查询订单状态", "读取物流字段", "格式化物流回复"]

        if not order:
            output = f"暂时没有查到订单 {order_id}。请核对订单号，或补充购买手机号后转人工核实。"
        else:
            status = order.get("status", "")
            product = order.get("product", "商品")
            if status == "completed":
                output = f"订单 {order.get('id')}（{product}）当前为已完成状态。如您需要具体物流轨迹，请提供运单号或联系人工查询承运商明细。"
            elif status == "shipped":
                output = f"订单 {order.get('id')}（{product}）已发货，正在配送中。"
            else:
                output = f"订单 {order.get('id')}（{product}）当前状态为 {status or '暂无物流更新'}。"
        tokens, llm_used = (
            0,
            False,
        )  # Facts are formatted directly, no planning/model call.

        return SolverOutput(
            status="completed" if order else "needs_input",
            active_solver="logistics",
            plan=plan,
            output=output,
            tool_calls=tool_calls,
            metrics=SolverMetrics(
                exec_tokens=tokens,
                exec_latency_ms=round((time.perf_counter() - started_at) * 1000, 1),
            ),
            llm_used=llm_used,
            facts={"order": order},
        )


def get_solver(name: str) -> BaseSolver:
    if name == "tech_support":
        return TechSupportSolver()
    if name == "logistics":
        return LogisticsSolver()
    return AfterSalesSolver()


def _infer_order_id(text: str, default: str = "") -> str:
    import re

    match = re.search(
        r"#\d{4,}|\bORDER[-_]?\d+\b|订单号(?:是|为)?\s*(\d{4,})", text, re.IGNORECASE
    )
    if not match:
        return default
    return f"#{match.group(1)}" if match.group(1) else match.group(0)


def _is_quality_issue(user_message: str, entities: dict) -> bool:
    text = f"{user_message} {entities.get('issue', '')}"
    return any(
        word in text for word in ["坏", "坏点", "质量", "故障", "开不了机", "黑屏"]
    )


def _infer_product(text: str) -> str:
    import re as _re

    for product in ["手机", "电脑", "耳机", "包裹", "订单"]:
        if product in text:
            return product
    # Fallback: try matching against actual product names in the database
    from devma.tools.order_query import get_all_product_tokens

    text_lower = text.lower()
    for token in get_all_product_tokens():
        t_lower = token.lower()
        if t_lower in text_lower:
            return token
        # Fuzzy match for typos
        if len(token) >= 4:
            from difflib import SequenceMatcher

            for word in _re.findall(r"[a-zA-Z0-9]+", text_lower):
                if (
                    len(word) >= 3
                    and SequenceMatcher(None, t_lower, word).ratio() >= 0.65
                ):
                    return token
    return ""
