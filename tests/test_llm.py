"""Tests for LLM adapter."""
from devma.models.llm import DeepSeekLLM, LLMResponse


def test_llm_simple_chat():
    """Basic sanity: LLM returns a response with content."""
    llm = DeepSeekLLM()
    resp = llm.chat([
        {"role": "system", "content": "Reply with exactly: OK"},
        {"role": "user", "content": "Say OK"}
    ])
    assert isinstance(resp, LLMResponse)
    assert len(resp.content) > 0
    assert resp.model == "deepseek-chat"
    assert resp.tokens.total > 0


def test_llm_structured_output():
    """LLM returns valid JSON when asked for structured output."""
    llm = DeepSeekLLM()
    resp = llm.chat_json(
        system="You are a classifier. Output only JSON.",
        messages=[{"role": "user", "content": "我要退货"}],
        json_schema={
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
        },
    )
    assert isinstance(resp, dict)
    assert "intent" in resp
    assert "confidence" in resp
    assert isinstance(resp["confidence"], (int, float))


def test_llm_handles_long_prompt():
    """LLM handles multi-turn conversation."""
    llm = DeepSeekLLM()
    resp = llm.chat([
        {"role": "user", "content": "手机屏幕有坏点，买了25天，能退吗？"}
    ])
    assert len(resp.content) > 0
