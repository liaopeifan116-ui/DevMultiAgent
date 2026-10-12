# 简历描述与代码对应

| 简历实现点 | 代码位置 | 可运行的功能 |
|---|---|---|
| 多意图拆分与动态编排 | `agents/router.py`、`orchestration/contracts.py`、`scheduler.py` | 预切分/模型结构化抽取/离线兜底，原文引用、实体和 DAG 校验；Send 并行及按依赖分批调度 |
| 多策略 Solver 与子图封装 | `agents/solver.py`、`orchestration/subgraphs.py` | ReAct 循环子图；受约束 Plan-and-Execute 子图；不调用 LLM 的物流工具节点 |
| 共享状态与协作通信 | `orchestration/state.py`、`graph.py` | task_id 局部回写及冲突检测；结构化前置事实复用；局部修正与下游失效；路由重规划；最多两次修正 |
| RAG 与分层质量控制 | `tools/kb_search.py`、`tools/refund.py`、`agents/monitor.py` | Markdown 切片、原始/扩展双路 Top-5 检索、证据留存复用；代码资格校验、规则和模型审核 |
| 检查点与暂停恢复 | `graph.open_graph`、`subgraphs.py`、`main.py` | SQLite；执行前确认与子图缺信息中断；进程重启后 resume |

功能测试在 `tests/test_dynamic_workflow.py` 中；离线测试不能用来证明真实模型的回答质量或生产性能。

尚未实现：真实退款/取消写接口及业务幂等事务、登录鉴权和订单归属检查、真实客服工单分配、线上部署与运营指标。
