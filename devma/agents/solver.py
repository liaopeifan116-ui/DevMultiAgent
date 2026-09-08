"""Solver agents for after-sales, tech support, and logistics tasks."""

from __future__ import annotations

import time
from typing import Any

from devma.orchestration.contracts import SolverMetrics, SolverOutput
from devma.models.llm import DeepSeekLLM
from devma.config import DEEPSEEK_API_KEY
from devma.tools.kb_search import search_kb
from devma.tools.order_query import query_order, query_orders_by_product, find_orders_by_text
from devma.tools.privacy import redact_sensitive_fields
from devma.tools.refund import check_refund_eligibility


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
        return SolverMetrics(exec_latency_ms=round((time.perf_counter() - started_at) * 1000, 1))

    def _generate_with_llm(self, system: str, user: str, fallback: str) -> tuple[str, int, bool]:
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
    """Plan-and-Solve after-sales flow: order -> eligibility -> policy -> reply."""

    active_solver = "after_sales"

    def solve(self, user_message: str, entities: dict) -> SolverOutput:
        started_at = time.perf_counter()
        plan = ["查询订单事实", "核实退款/换货/维修资格", "检索政策依据", "生成解释性回复"]
        tool_calls: list[dict[str, Any]] = []

        order_id = entities.get("order_id") or _infer_order_id(user_message, default="")
        order = query_order(order_id) if order_id else None
        # If no order_id found, try matching product name from user message against DB
        if not order:
            by_text = find_orders_by_text(user_message)
            if len(by_text) == 1:
                order = by_text[0]
                order_id = order["id"]
            elif len(by_text) > 1:
                # Multiple matches: list them for user to confirm
                summary = "；".join(
                    f"{item['id']}（{item['product']}，{item['purchase_date']}）"
                    for item in by_text[:5]
                )
                return SolverOutput(
                    status="completed",
                    active_solver="after_sales",
                    plan=["查询订单事实", "发现多个候选订单，请用户确认"],
                    output=f"根据您提到的商品，我找到了以下相关订单：{summary}。请问您要咨询哪一个？",
                    tool_calls=tool_calls,
                    metrics=self._metrics(started_at),
                )
        tool_calls.append(self._tool_call("get_order", order_id, order))

        issue_type = "quality" if _is_quality_issue(user_message, entities) else "buyer_remorse"
        eligibility = check_refund_eligibility(order, issue_type=issue_type) if order else None
        if order:
            tool_calls.append(self._tool_call("check_refund_eligibility", {"order_id": order_id, "issue_type": issue_type}, eligibility))

        query = f"{entities.get('product', '')} {entities.get('issue', '')} 退款 换货 维修".strip()
        kb_results = search_kb(query or user_message, context=user_message, top_k=3)
        tool_calls.append(
            self._tool_call(
                "search_kb",
                query or user_message,
                [{"source": r.get("source"), "score": r.get("score"), "retrieval": r.get("retrieval")} for r in kb_results],
            )
        )

        output = self._build_after_sales_reply(order, eligibility, kb_results, order_id)
        output, tokens, llm_used = self._generate_with_llm(
            system=(
                "你是电商售后客服，只能基于提供的订单、规则判断和知识库证据回复。"
                "不得承诺全额退款、额外赔偿、优惠券、加急处理或任何未授权动作。"
                "必须以规则判断 action 和 reason 作为当前订单的最终结论；"
                "如果 action=reject，不要把其他时间窗口的可换修规则说成当前订单仍可享受。"
                "如果规则判断不支持退款，要礼貌解释原因并给出换货/维修/人工复核等下一步。"
                "用简洁中文回复，不要输出 JSON。"
            ),
            user=(
                f"用户问题：{user_message}\n"
                f"订单：{redact_sensitive_fields(order)}\n"
                f"规则判断：{eligibility}\n"
                f"知识库证据：{[{'source': r.get('source'), 'content': r.get('content')[:300]} for r in kb_results]}\n"
                f"兜底回复：{output}"
            ),
            fallback=output,
        )
        return SolverOutput(
            status="completed",
            active_solver="after_sales",
            plan=plan,
            output=output,
            tool_calls=tool_calls,
            metrics=SolverMetrics(exec_tokens=tokens, exec_latency_ms=round((time.perf_counter() - started_at) * 1000, 1)),
            llm_used=llm_used,
        )

    def _build_after_sales_reply(self, order: dict | None, eligibility: dict | None, kb_results: list[dict], order_id: str) -> str:
        if not order_id:
            return "请提供您的订单号，这样我可以帮您查询具体的售后状态。您可以在订单详情页找到订单号（如 #12345）。"
        if not order:
            return f"我还没有查到订单 {order_id} 的记录。请您核对订单号，或补充购买手机号后转人工继续核实。"

        product = order.get("product", "商品")
        days_text = eligibility.get("reason", "") if eligibility else "暂时无法判断售后资格。"
        source = kb_results[0].get("source", "售后政策") if kb_results else "售后政策"
        action = eligibility.get("action", "") if eligibility else "handoff"

        if eligibility and eligibility.get("eligible"):
            next_step = "可以提交退货申请，退款金额以订单实际支付金额为准。"
        elif action == "approve_exchange_or_repair":
            next_step = "建议申请换货或维修，我可以继续帮您整理售后材料。"
        elif action == "handoff":
            next_step = "该情况需要人工进一步核实，我会保留当前处理摘要。"
        else:
            next_step = "当前不满足退款条件，如您认为商品存在特殊质量问题，建议转人工复核。"

        return (
            f"已为您查询订单 {order.get('id')}，商品为 {product}。"
            f"{days_text}依据 {source} 的售后规则，{next_step}"
        )


class TechSupportSolver(BaseSolver):
    """ReAct-like tech solver grounded in troubleshooting knowledge."""

    active_solver = "tech_support"

    def solve(self, user_message: str, entities: dict) -> SolverOutput:
        started_at = time.perf_counter()
        query = f"{entities.get('product', '')} {entities.get('issue', '')} 故障 排查".strip() or user_message
        kb_results = search_kb(query, context=user_message, top_k=3)
        tool_calls = [self._tool_call("search_kb", query, [{"source": r.get("source"), "score": r.get("score")} for r in kb_results])]
        plan = ["检索故障排查知识", "根据用户现象给出安全排查步骤", "无法确认时建议售后或人工"]

        if "开不了机" in user_message or "无法开机" in user_message:
            output = (
                "可以先按顺序排查：1. 连接原装充电器充电 10 分钟；"
                "2. 长按电源键 15 秒尝试强制重启；3. 检查是否有进水、摔落或屏幕损坏痕迹。"
                "如果充电指示灯不亮且强制重启无效，可能需要售后检测。"
            )
        else:
            output = "我先根据知识库给您做基础排查。请补充故障现象、发生时间以及是否进水或摔落。"
        output, tokens, llm_used = self._generate_with_llm(
            system=(
                "你是技术支持客服，只能做安全的知识库驱动排查。"
                "不得假装远程诊断设备，不得承诺退款或赔偿。"
                "用步骤化中文回复，必要时建议售后检测或人工处理。"
            ),
            user=(
                f"用户问题：{user_message}\n"
                f"实体：{entities}\n"
                f"知识库证据：{[{'source': r.get('source'), 'content': r.get('content')[:300]} for r in kb_results]}\n"
                f"兜底回复：{output}"
            ),
            fallback=output,
        )

        return SolverOutput(
            status="completed",
            active_solver="tech_support",
            plan=plan,
            output=output,
            tool_calls=tool_calls,
            metrics=SolverMetrics(exec_tokens=tokens, exec_latency_ms=round((time.perf_counter() - started_at) * 1000, 1)),
            llm_used=llm_used,
        )


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
                    summary = "；".join(f"{item['id']}（{item['product']}，{item['status']}）" for item in matches[:3])
                    return SolverOutput(
                        status="completed",
                        active_solver="logistics",
                        plan=["识别物流/订单查询意图", f"按商品关键词查询：{product}", "发现多个候选订单，请用户确认"],
                        output=f"我查到多个和“{product}”相关的订单：{summary}。请您确认要查询哪一个订单。",
                        tool_calls=[self._tool_call("query_orders_by_product", product, matches[:3])],
                        metrics=self._metrics(started_at),
                    )
                else:
                    return SolverOutput(
                        status="completed",
                        active_solver="logistics",
                        plan=["识别物流/订单查询意图", f"按商品关键词查询：{product}", "未找到匹配订单"],
                        output=f"我没有查到和“{product}”匹配的订单。请您补充订单号或购买账号，我再继续查。",
                        tool_calls=[self._tool_call("query_orders_by_product", product, [])],
                        metrics=self._metrics(started_at),
                    )
            else:
                plan = ["识别物流/订单查询意图", "缺少订单号和商品信息", "请求用户补充订单号、商品或购买账号"]
                output = "我可以帮您查询订单或物流状态。请您补充订单号，或说明要查询的商品和购买账号。"
                return SolverOutput(
                    status="completed",
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
        output, tokens, llm_used = self._generate_with_llm(
            system=(
                "你是物流客服，只能基于订单查询结果回复。"
                "如果订单没有预计送达时间或运单轨迹，不得自行编造。"
                "用简洁中文回复。"
            ),
            user=f"用户问题：{user_message}\n订单查询结果：{redact_sensitive_fields(order)}\n兜底回复：{output}",
            fallback=output,
        )

        return SolverOutput(
            status="completed",
            active_solver="logistics",
            plan=plan,
            output=output,
            tool_calls=tool_calls,
            metrics=SolverMetrics(exec_tokens=tokens, exec_latency_ms=round((time.perf_counter() - started_at) * 1000, 1)),
            llm_used=llm_used,
        )


def get_solver(name: str) -> BaseSolver:
    if name == "tech_support":
        return TechSupportSolver()
    if name == "logistics":
        return LogisticsSolver()
    return AfterSalesSolver()


def _infer_order_id(text: str, default: str = "") -> str:
    import re

    match = re.search(r"#\d{4,}|\bORDER[-_]?\d+\b|订单号(?:是|为)?\s*(\d{4,})", text, re.IGNORECASE)
    if not match:
        return default
    return f"#{match.group(1)}" if match.group(1) else match.group(0)


def _is_quality_issue(user_message: str, entities: dict) -> bool:
    text = f"{user_message} {entities.get('issue', '')}"
    return any(word in text for word in ["坏", "坏点", "质量", "故障", "开不了机", "黑屏"])


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
                if len(word) >= 3 and SequenceMatcher(None, t_lower, word).ratio() >= 0.65:
                    return token
    return ""
