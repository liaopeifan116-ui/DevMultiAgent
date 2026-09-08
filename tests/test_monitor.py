"""Tests for Monitor agent."""
from devma.agents.monitor import MonitorAgent


def test_monitor_check1_blocks_full_refund():
    monitor = MonitorAgent()
    result = monitor.check(
        solver_output="好的，我们为您办理全额退款。",
        user_message="手机屏幕有坏点能退吗",
        kb_context=["退款金额为实际支付金额"],
    )
    assert result["result"] == "REJECT"
    assert "Check 1" in result["reason"]


def test_monitor_check1_blocks_compensation():
    monitor = MonitorAgent()
    result = monitor.check(
        solver_output="好的，赔偿您50元作为补偿。",
        user_message="手机坏了",
        kb_context=[],
    )
    assert result["result"] == "REJECT"


def test_monitor_pass_valid_output():
    monitor = MonitorAgent()
    result = monitor.check(
        solver_output="您的订单购买25天，已超过7天退货期但在30天保修期内。屏幕坏点属于质量问题，您可以申请换货或维修。",
        user_message="手机屏幕有坏点能退吗",
        kb_context=["屏幕坏点属于质量问题，签收7天内可退货退款，7-30天内可换货或维修。"],
    )
    assert result["result"] == "PASS"


def test_monitor_check4_handoff():
    monitor = MonitorAgent()
    result = monitor.check(
        solver_output="好的，我来帮您处理。",
        user_message="你们太过分了，我要投诉你，我要去曝光你们！",
        kb_context=[],
    )
    assert result["checks"]["check4_escalation"]["handoff_recommended"] is True


def test_monitor_check1_no_false_positive():
    """Normal response shouldn't trigger Check 1."""
    monitor = MonitorAgent()
    result = monitor.check(
        solver_output="您好，根据您的订单信息，我来帮您查询退款政策。",
        user_message="能退款吗",
        kb_context=[],
    )
    assert result["checks"]["check1_keyword"]["hit"] is False


def test_monitor_all_checks_populated():
    """All 4 checks are present in the result."""
    monitor = MonitorAgent()
    result = monitor.check(
        solver_output="我来帮您处理换货申请。",
        user_message="手机屏幕有坏点",
        kb_context=["屏幕坏点7天内可退货"],
    )
    for check_name in ["check1_keyword", "check2_commitment", "check3_grounding", "check4_escalation"]:
        assert check_name in result["checks"]
