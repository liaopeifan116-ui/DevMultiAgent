"""DeepSeek LLM adapter via openai SDK."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import openai

from devma.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


@dataclass
class TokenCount:
    prompt: int = 0
    completion: int = 0
    total: int = 0


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens: TokenCount = field(default_factory=TokenCount)
    latency_ms: float = 0.0


class DeepSeekLLM:
    """Unified DeepSeek API caller."""

    def __init__(self, model: str | None = None):
        self.model = model or DEEPSEEK_MODEL
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            if not DEEPSEEK_API_KEY:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY not set. Set environment variable "
                    "or pass api_key explicitly."
                )
            self._client = openai.OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                timeout=30.0,
                max_retries=1,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send chat completion, return LLMResponse."""
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0]
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            tokens=TokenCount(
                prompt=prompt_tokens,
                completion=completion_tokens,
                total=total_tokens,
            ),
            latency_ms=round(elapsed, 1),
        )

    def chat_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Chat completion with structured JSON output."""
        full_messages = [{"role": "system", "content": system}] + messages
        full_messages.append(
            {
                "role": "system",
                "content": f"Respond ONLY with a JSON object matching this schema. No markdown, no explanation:\n{json.dumps(json_schema, ensure_ascii=False)}",
            }
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""

        # Try direct parse first, then extract from markdown code block
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
