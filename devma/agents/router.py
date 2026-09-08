"""Router Agent: intent classification + entity extraction + solver selection."""
import json
import re
from devma.models.llm import DeepSeekLLM
from devma.config import DEEPSEEK_API_KEY, INTENTS
from devma.orchestration.contracts import RouterOutput


RouterResult = RouterOutput


SOLVER_MAP = {
    "after_sales": "after_sales",
    "tech_support": "tech_support",
    "logistics": "logistics",
    "general_chat": "general_chat",
    "unknown": "general_chat",
}

GREETING_OR_EMPTY_PATTERNS = [
    re.compile(r"^\s*[?？!！。.,，、]*\s*$"),
    re.compile(r"^\s*(你好|您好|hello|hi|嗨|在吗|有人吗)[?？!！。.,，、]*\s*$", re.IGNORECASE),
]

ROUTER_SYSTEM = """You are an intent classifier for a customer service system.

Classify the user's message into ONE of these intents:
- after_sales: Returns, refunds, complaints, quality issues, warranty, repairs, exchanges
- tech_support: Technical problems, troubleshooting, how-to questions
- logistics: Shipping status, delivery tracking, order lookup, purchase history
- unknown: Greeting, empty message, or truly unclassifiable

Also extract key entities from the message.

Output ONLY a JSON object:
{
  "intent": "after_sales|tech_support|logistics|unknown",
  "confidence": 0.0-1.0,
  "entities": {
    "product": "product name or empty string",
    "issue": "problem description or empty string",
    "order_id": "order number (e.g. #12345) or empty string"
  }
}"""

MULTI_ROUTER_SYSTEM = """You are an intent classifier for a customer service system.
The user's message contains MULTIPLE independent requests, split below.
Classify EACH part separately.

Output ONLY a JSON array, one object per part:
[{
  "intent": "after_sales|tech_support|logistics|unknown",
  "confidence": 0.0-1.0,
  "entities": {
    "product": "product name or empty string",
    "issue": "problem description or empty string",
    "order_id": "order number (e.g. #12345) or empty string"
  }
}]"""


class RouterAgent:
    def __init__(self):
        self.llm = DeepSeekLLM()

    def route(self, user_message: str, user_context: dict | None = None) -> RouterOutput:
        # Greeting / empty → clarify, no LLM needed
        if self._is_greeting_or_empty(user_message):
            return self._greeting_result(user_message)

        # Split into parts for multi-intent detection
        parts = self._split_message(user_message)
        entities = self._extract_entities(user_message)

        # No API key → best-effort without LLM
        if not DEEPSEEK_API_KEY:
            return self._no_llm_result(parts, entities, user_message)

        # Multiple parts → classify each with LLM
        if len(parts) >= 2:
            return self._multi_llm_route(parts, entities, user_message, user_context)

        # Single part → standard LLM classification
        return self._single_llm_route(user_message, entities, user_context)

    # ── greeting ──────────────────────────────────────────────────────

    def _is_greeting_or_empty(self, message: str) -> bool:
        return any(pattern.match(message) for pattern in GREETING_OR_EMPTY_PATTERNS)

    def _greeting_result(self, user_message: str) -> RouterOutput:
        item = {
            "intent": "unknown",
            "sub_intent": "clarify",
            "confidence": 0.95,
            "target_solver": "handoff",
            "entities": {},
            "span": user_message,
        }
        return RouterOutput(
            intent="unknown", confidence=0.95, target_solver="handoff",
            entities={}, sub_intents=[item], execution_mode="single",
            reason="Greeting or empty message requires clarification",
            raw_response=json.dumps(item, ensure_ascii=False),
        )

    # ── message splitting ─────────────────────────────────────────────

    def _split_message(self, message: str) -> list[str]:
        normalized = message.replace("？", "。").replace("！", "。")
        chunks = [part.strip("，, ") for part in re.split(r"[。；;]|还有|另外|以及|如果", normalized) if part.strip("，, ")]
        return chunks or [message]

    # ── entity extraction ─────────────────────────────────────────────

    def _extract_entities(self, text: str) -> dict[str, str]:
        entities: dict[str, str] = {}
        order_match = re.search(r"#\d{4,}|\bORDER[-_]?\d+\b|订单号(?:是|为)?\s*(\d{4,})", text, re.IGNORECASE)
        if order_match:
            entities["order_id"] = f"#{order_match.group(1)}" if order_match.group(1) else order_match.group(0)

        for product in ["手机", "耳机", "屏幕", "包裹", "订单", "电脑"]:
            if product in text:
                entities["product"] = product
                break

        if not entities.get("product"):
            from devma.tools.order_query import get_all_product_tokens
            text_lower = text.lower()
            for token in get_all_product_tokens():
                t_lower = token.lower()
                if t_lower in text_lower:
                    entities["product"] = token
                    break
                if len(token) >= 4:
                    from difflib import SequenceMatcher
                    for word in re.findall(r"[a-zA-Z0-9]+", text_lower):
                        if len(word) >= 3 and SequenceMatcher(None, t_lower, word).ratio() >= 0.65:
                            entities["product"] = token
                            break
                    if entities.get("product"):
                        break

        issue_words = ["坏点", "坏", "开不了机", "无法开机", "黑屏", "质量问题", "故障", "退款", "物流"]
        for issue in issue_words:
            if issue in text:
                entities["issue"] = issue
                break
        return entities

    # ── no-LLM fallback ───────────────────────────────────────────────

    def _no_llm_result(self, parts: list[str], entities: dict, user_message: str) -> RouterOutput:
        if len(parts) >= 2:
            sub_intents = [self._make_sub("after_sales", part, 0.3) for part in parts]
            return RouterOutput(
                intent="multi_intent", confidence=0.3, target_solver="after_sales",
                entities=entities, sub_intents=sub_intents, execution_mode="sequential",
                reason="No LLM available, best-effort routing",
                raw_response=json.dumps(sub_intents, ensure_ascii=False),
            )
        item = self._make_sub("after_sales", user_message, 0.3)
        return RouterOutput(
            intent="after_sales", confidence=0.3, target_solver="after_sales",
            entities=entities, sub_intents=[item], execution_mode="single",
            reason="No LLM available, defaulting to after_sales",
            raw_response=json.dumps(item, ensure_ascii=False),
        )

    def _make_sub(self, intent: str, span: str, confidence: float) -> dict:
        return {
            "intent": intent, "sub_intent": "general",
            "confidence": confidence,
            "target_solver": SOLVER_MAP.get(intent, "after_sales"),
            "entities": self._extract_entities(span),
            "span": span,
        }

    # ── single LLM routing ────────────────────────────────────────────

    def _single_llm_route(self, user_message: str, entities: dict, user_context: dict | None) -> RouterOutput:
        result = self._call_llm(ROUTER_SYSTEM, user_message, user_context)
        intent = result.get("intent", "unknown")
        confidence = float(result.get("confidence", 0.5))
        llm_entities = result.get("entities", {})

        # Merge rule-extracted entities with LLM entities
        for key in ["order_id", "product", "issue"]:
            if not llm_entities.get(key) and entities.get(key):
                llm_entities[key] = entities[key]
        if llm_entities.get("order_id") and not str(llm_entities["order_id"]).startswith("#"):
            llm_entities["order_id"] = f"#{llm_entities['order_id']}"

        if user_context:
            product_orders = user_context.get("product_orders", {})
            product = llm_entities.get("product", "")
            if product and not llm_entities.get("order_id") and product_orders.get(product):
                llm_entities["order_id"] = product_orders[product]

        target_solver = SOLVER_MAP.get(intent, "after_sales")
        return RouterOutput(
            intent=intent, confidence=confidence, target_solver=target_solver,
            entities=llm_entities,
            sub_intents=[{
                "intent": intent, "sub_intent": "general", "confidence": confidence,
                "target_solver": target_solver, "entities": llm_entities, "span": user_message,
            }],
            execution_mode="single",
            reason=f"LLM route {intent} to {target_solver}",
            raw_response=json.dumps(result, ensure_ascii=False),
        )

    # ── multi LLM routing ─────────────────────────────────────────────

    def _multi_llm_route(self, parts: list[str], entities: dict, user_message: str, user_context: dict | None) -> RouterOutput:
        # Build a prompt listing each part
        parts_text = "\n".join(f"[{i}] {part}" for i, part in enumerate(parts))
        prompt = f"Message parts:\n{parts_text}\n\nClassify each part."

        try:
            results = self.llm.chat_json(
                system=MULTI_ROUTER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                json_schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "intent": {"type": "string", "enum": INTENTS + ["unknown"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "entities": {
                                "type": "object",
                                "properties": {
                                    "product": {"type": "string"},
                                    "issue": {"type": "string"},
                                    "order_id": {"type": "string"},
                                },
                            },
                        },
                        "required": ["intent", "confidence", "entities"],
                    },
                },
            )
        except Exception:
            # LLM failed → fall back to single-part classification for the whole message
            return self._single_llm_route(user_message, entities, user_context)

        if not isinstance(results, list) or len(results) != len(parts):
            return self._single_llm_route(user_message, entities, user_context)

        sub_intents = []
        for i, result in enumerate(results):
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.5))
            llm_entities = result.get("entities", {})
            part_entities = self._extract_entities(parts[i])

            for key in ["order_id", "product", "issue"]:
                if not llm_entities.get(key) and part_entities.get(key):
                    llm_entities[key] = part_entities[key]
            if llm_entities.get("order_id") and not str(llm_entities["order_id"]).startswith("#"):
                llm_entities["order_id"] = f"#{llm_entities['order_id']}"

            target_solver = SOLVER_MAP.get(intent, "after_sales")
            sub_intents.append({
                "intent": intent,
                "sub_intent": "general",
                "confidence": confidence,
                "target_solver": target_solver,
                "entities": llm_entities,
                "span": parts[i],
            })

        solvers = [s["target_solver"] for s in sub_intents]
        execution_mode = "parallel" if len(set(solvers)) == len(solvers) else "sequential"

        return RouterOutput(
            intent="multi_intent",
            confidence=min(s["confidence"] for s in sub_intents),
            target_solver=sub_intents[0]["target_solver"],
            entities=entities,
            sub_intents=sub_intents,
            execution_mode=execution_mode,
            reason=f"{len(sub_intents)} sub-intents routed by LLM as {execution_mode}",
            raw_response=json.dumps(results, ensure_ascii=False),
        )

    # ── LLM helper ────────────────────────────────────────────────────

    def _call_llm(self, system: str, user_message: str, user_context: dict | None) -> dict:
        ctx_text = ""
        if user_context:
            ctx_text = f"\nAdditional user context: {json.dumps(user_context, ensure_ascii=False)}"
        return self.llm.chat_json(
            system=system,
            messages=[{"role": "user", "content": user_message + ctx_text}],
            json_schema={
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": INTENTS + ["unknown"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "entities": {
                        "type": "object",
                        "properties": {
                            "product": {"type": "string"},
                            "issue": {"type": "string"},
                            "order_id": {"type": "string"},
                        },
                    },
                },
                "required": ["intent", "confidence", "entities"],
            },
        )
