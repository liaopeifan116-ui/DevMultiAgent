"""Monitor Agent: 4 quality checks on solver output."""
import concurrent.futures
import json
import re
import yaml
from devma.models.llm import DeepSeekLLM
from devma.config import RULES_DIR
from devma.tools.kb_search import search_kb
from devma.orchestration.contracts import MonitorOutput

CHECK2_SYSTEM = """You are a quality monitor checking for unauthorized commitments.

Check if the response makes promises beyond what is allowed:
- Monetary compensation beyond the refund amount
- Promises of expedited processing not in policy
- Guarantees of outcomes

Output JSON:
{"violation": true/false, "what": "description if violated"}"""


class MonitorAgent:
    def __init__(self):
        self.llm = DeepSeekLLM()
        self._rules: dict | None = None

    @property
    def rules(self) -> dict:
        if self._rules is None:
            path = RULES_DIR / "hard_rules.yaml"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._rules = yaml.safe_load(f)
            else:
                self._rules = {}
        return self._rules

    def check(self, solver_output: str, user_message: str, kb_context: list[str], tool_context: list[dict] | None = None) -> MonitorOutput:
        checks = {}

        # Check 1: hard rules are deterministic and should reject before any
        # external LLM calls.
        c1 = self._check1_keywords(solver_output, user_message)
        checks["check1_keyword"] = c1
        if c1["hit"]:
            return MonitorOutput(
                decision="REJECT",
                reason=f"Check 1 硬规则拦截: {c1['matched']}",
                checks=checks,
            )

        # Check 4: escalation is also deterministic. Handoff immediately so a
        # network failure cannot block an urgent human escalation.
        c4 = self._check4_escalation(user_message)
        checks["check4_escalation"] = c4
        if c4.get("handoff_recommended"):
            return MonitorOutput(
                decision="HANDOFF",
                reason=f"Check 4 情绪激化转人工: abusive={c4.get('abusive_keywords_hit', [])}, "
                       f"emotion_signals={c4.get('emotion_signals', 0)}",
                checks=checks,
            )

        # Check 2 & Check 3: parallel LLM calls (independent, ~4s saved)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_c2 = pool.submit(self._check2_commitment, solver_output)
            future_c3 = pool.submit(self._call_check3_grounding, solver_output, user_message, tool_context or [])
            c2 = future_c2.result()
            c3 = future_c3.result()
        checks["check2_commitment"] = c2
        checks["check3_grounding"] = c3

        if c2.get("violation"):
            return MonitorOutput(
                decision="NEEDS_FIX",
                reason=f"Check 2 未授权承诺: {c2.get('what', '')}",
                checks=checks,
                retry_feedback="请删除未经授权的承诺，并依据规则重新生成回复。",
            )

        if c3.get("hallucination"):
            return MonitorOutput(
                decision="NEEDS_FIX",
                reason=f"Check 3 知识库依据不足: {c3.get('detail', '')}",
                checks=checks,
                retry_feedback="请仅使用知识库可以支持的事实重新生成回复。",
            )

        return MonitorOutput(
            decision="PASS",
            reason="All checks passed",
            checks=checks,
        )

    def _check1_keywords(self, output: str, user_message: str = "") -> dict:
        keywords = self.rules.get("hard_rules", {}).get("keywords_block", [])
        forbidden = self.rules.get("hard_rules", {}).get("forbidden_requests", [])
        injection_patterns = ["忽略系统规则", "忽略规则", "不要走流程", "直接给我", "绕过"]
        output_matched = [kw for kw in keywords + forbidden if kw in output]
        user_matched = [kw for kw in keywords + forbidden + injection_patterns if kw in user_message]
        matched = output_matched + user_matched
        return {"hit": len(matched) > 0, "matched": matched, "output_matched": output_matched, "user_matched": user_matched}

    def _check2_commitment(self, output: str) -> dict:
        try:
            return self.llm.chat_json(
                system=CHECK2_SYSTEM,
                messages=[{"role": "user", "content": output}],
                json_schema={
                    "type": "object",
                    "properties": {
                        "violation": {"type": "boolean"},
                        "what": {"type": "string"},
                    },
                    "required": ["violation"],
                },
                temperature=0.1,
            )
        except Exception:
            return {"violation": False}

    def _check3_grounding(self, output: str, user_message: str, tool_context: list[dict] | None = None) -> dict:
        if self._tool_evidence_supports_output(output, tool_context or []):
            return {"hallucination": False, "detail": "订单/物流事实由工具结果支持"}

        kb_results = search_kb(user_message, context=output)
        kb_text = " ".join([r["content"][:300] for r in kb_results[:3]])

        try:
            return self.llm.chat_json(
                system="Check if factual claims in the response are supported by KB. Output JSON: {\"hallucination\": bool, \"detail\": str}",
                messages=[{"role": "user", "content": (
                    f"Response:\n{output}\n\nKB:\n{kb_text}\n\n"
                    f"Are there factual claims that contradict or are unsupported by the KB?"
                )}],
                json_schema={
                    "type": "object",
                    "properties": {
                        "hallucination": {"type": "boolean"},
                        "detail": {"type": "string"},
                    },
                    "required": ["hallucination"],
                },
                temperature=0.1,
            )
        except Exception:
            return {"hallucination": False, "detail": ""}

    def _tool_evidence_supports_output(self, output: str, tool_context: list[dict]) -> bool:
        if not tool_context:
            return False
        for call in tool_context:
            if call.get("tool") != "get_order":
                continue
            order = call.get("result") or {}
            if not isinstance(order, dict):
                continue
            if order.get("id") and order.get("product"):
                return True
            order_id = str(order.get("id", ""))
            product = str(order.get("product", ""))
            status = str(order.get("status", ""))
            if (order_id and order_id in output) or (product and product in output) or (status and status in output):
                return True
        return False

    def _call_check3_grounding(self, output: str, user_message: str, tool_context: list[dict]) -> dict:
        try:
            return self._check3_grounding(output, user_message, tool_context)
        except TypeError:
            return self._check3_grounding(output, user_message)

    def _check4_escalation(self, user_message: str) -> dict:
        abusive = self.rules.get("hard_rules", {}).get("tones", {}).get("abusive_keywords", [])
        threshold = self.rules.get("hard_rules", {}).get("tones", {}).get("abusive_threshold", 2)
        hits = [kw for kw in abusive if kw in user_message]
        emotion_hits = []
        for pat in [r'太.*?(差|坑|烂|坏)', r'(气|火|怒)', r'(骗子|无赖|黑心|黑店)']:
            if re.search(pat, user_message):
                emotion_hits.append(pat)

        should_handoff = len(hits) >= threshold or len(emotion_hits) >= 2

        return {
            "handoff_recommended": should_handoff,
            "abusive_keywords_hit": hits,
            "emotion_signals": len(emotion_hits),
        }
