"""Evaluation test cases including normal cases and badcases."""
import json
from pathlib import Path

DEFAULT_TEST_CASES = [
    {
        "id": "normal_001",
        "query": "手机屏幕有坏点能退吗",
        "expected_intent": "after_sales",
        "expected_keywords": ["7天", "坏点", "质量"],
        "forbidden_content": ["全额退款", "赔偿"],
        "description": "屏幕坏点 -> 售后退款咨询",
    },
    {
        "id": "normal_002",
        "query": "我的包裹到哪了",
        "expected_intent": "logistics",
        "expected_keywords": ["物流", "配送"],
        "forbidden_content": [],
        "description": "物流查询",
    },
    {
        "id": "normal_003",
        "query": "手机开不了机了怎么办",
        "expected_intent": "tech_support",
        "expected_keywords": ["重启", "开机", "排查"],
        "forbidden_content": [],
        "description": "技术故障排查",
    },
    {
        "id": "badcase_001",
        "query": "这手机质量太差了，我要全额退款",
        "expected_intent": "after_sales",
        "forbidden_content": ["全额退款"],
        "description": "要求全额退款 -> 系统不应承诺全额退款",
    },
    {
        "id": "badcase_002",
        "query": "我25天前买的手机屏幕有坏点，能退款吗",
        "expected_intent": "after_sales",
        "expected_keywords": ["换货", "维修"],
        "forbidden_content": ["退款"],
        "description": "25天 -> 超过7天退货期，只能换货/维修",
    },
    {
        "id": "badcase_003",
        "query": "你们太过分了我要投诉你我要去曝光你们",
        "expected_intent": "after_sales",
        "description": "用户情绪激动 -> 应触发转人工",
    },
]


def load_test_cases(file_path: str | None = None) -> list[dict]:
    if file_path:
        path = Path(file_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return DEFAULT_TEST_CASES
