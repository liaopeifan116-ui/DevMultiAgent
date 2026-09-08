# DevMultiAgent

DevMultiAgent 是一个面向电商售后场景的多 Agent 协同工作流。系统基于 LangGraph 编排 Router、三类 Solver 与 Monitor，将复合意图拆分、业务执行和质量检查分开处理，并通过共享状态传递子任务、工具结果与质检反馈。

## 核心设计

- **多意图路由**：Router 拆分复合请求，提取订单、商品和问题实体，并将任务分配给对应 Solver。
- **多策略执行**：技术排障采用 ReAct 风格的动态探索，售后处理采用 Plan-and-Solve 风格的分阶段执行，物流查询使用轻量 ToolUse。
- **共享状态协作**：AgentState 保存子意图、执行模式、工具调用、Solver 输出、Monitor 结论和重试状态。
- **知识检索**：使用 Query 补全、BM25 与查询词覆盖率召回政策、FAQ 和故障文档。
- **质量控制**：结合 YAML 规则、代码拦截和 LLM 审核检查违规承诺、知识依据及人工接管条件。

## 工作流程

```text
用户请求
   ↓
Router：意图拆分、实体提取、Solver 路由
   ↓
Solver：技术排障 / 售后处理 / 物流查询
   ↓
Monitor：规则检查、依据核验、修改或转人工
   ↓
最终回复
```

## 安装

需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
pip install -e .
```

复制环境变量示例并填写 DeepSeek API Key：

```powershell
Copy-Item .env.example .env
```

初始化示例订单数据：

```powershell
python data/seed_orders.py
```

## 运行

```powershell
python interactive.py
```

## 测试

```powershell
pytest
```

项目在未配置模型密钥时保留确定性兜底路径；涉及真实模型调用的测试需要配置 `DEEPSEEK_API_KEY`。

## 项目结构

```text
devma/
├── agents/          # Router、Solver、Monitor
├── orchestration/   # LangGraph、AgentState 与结构化协议
├── tools/           # 知识检索、订单查询、资格判断
├── rules/           # YAML 业务规则
└── evaluation/      # 检索评测
data/
├── kb/              # 示例知识库
└── seed_orders.py   # 示例订单数据生成脚本
docs/                # 设计取舍与评测记录
tests/               # 单元测试与流程测试
```
