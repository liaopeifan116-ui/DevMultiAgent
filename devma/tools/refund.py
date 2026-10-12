"""Refund eligibility check based on hard rules and order data."""

from datetime import date, datetime

import yaml

from devma.config import RULES_DIR


def _load_rules() -> dict:
    path = RULES_DIR / "hard_rules.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _days_since(purchase_date_str: str) -> int:
    purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d").date()
    return (date.today() - purchase_date).days


def check_refund_eligibility(order: dict, issue_type: str = "quality") -> dict:
    rules = _load_rules()
    refund_rules = rules.get("hard_rules", {}).get("refund", {})
    max_days = refund_rules.get("max_days", 30)
    requires_receipt = refund_rules.get("requires_receipt", True)
    try:
        days = _days_since(order["purchase_date"])
    except (KeyError, ValueError, TypeError):
        return {
            "eligible": False,
            "reason": "订单购买日期缺失或无效，需人工核实。",
            "action": "handoff",
        }
    if days < 0 or issue_type not in {"quality", "buyer_remorse"}:
        return {
            "eligible": False,
            "reason": "订单日期或问题类型异常，需人工核实。",
            "action": "handoff",
        }
    if order.get("price", 0) > refund_rules.get("max_amount", 50000):
        return {
            "eligible": False,
            "reason": "订单金额超过自动审核上限，需人工核实。",
            "action": "handoff",
        }

    if requires_receipt and not order.get("has_receipt"):
        return {
            "eligible": False,
            "reason": "无法提供购机发票，不符合退款条件。建议联系客服人工处理。",
            "action": "handoff",
        }

    if days > max_days:
        return {
            "eligible": False,
            "reason": f"已超过{max_days}天退货期限（已购买{days}天），不支持退款。",
            "action": "reject",
        }

    if days <= 7:
        return {
            "eligible": True,
            "reason": f"购买{days}天，在7天退货期内，符合退款条件。",
            "action": "approve_refund",
        }

    if 7 < days <= max_days:
        if issue_type == "quality":
            return {
                "eligible": False,
                "reason": f"购买{days}天，已超过7天退货期但在30天保修期内。质量问题可申请换货或维修，不支持退款。",
                "action": "approve_exchange_or_repair",
            }
        else:
            return {
                "eligible": False,
                "reason": f"购买{days}天，已超过7天退货期。非质量问题不支持退款。",
                "action": "reject",
            }

    return {
        "eligible": False,
        "reason": "无法判断退款资格，建议转人工处理。",
        "action": "handoff",
    }
