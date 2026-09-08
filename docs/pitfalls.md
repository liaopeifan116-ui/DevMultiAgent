# 踩坑记录 — 真实开发中遇到的问题与决策

> 记录日期：2026-06-04 ~ 2026-06-05
> 触发背景：Codex 修改后的项目对话体验差，手动排查和修复

---

## 1. 关键词路由：省 token 的代价远超收益

**问题**：Router 用 `INTENT_KEYWORDS` 做关键词→意图映射，高置信度时跳过 LLM。

**踩到的坑**（同一机制触发 5+ 个 bug）：
- "我#1234屏幕坏了" → "坏"不在关键词列表 → intent=unknown → 掉进 general_chat，不查订单
- "我的redmi坏了" → "坏"被误删 → 零关键词命中 → 被多意图拆分过滤掉
- "AirPods到哪了" → 实体提取不认识 AirPods → 找不到订单
- "花屏了" → 不在关键词列表 → 依赖 LLM 兜底（兜对了但逻辑绕了一圈）
- "thinlpad" typo → 关键词匹配完全无用

**结论**：模型明明能自己判断意图。省一个 Router LLM 调用（~200ms）换来一堆边界 case 和不断增加的关键词维护负担 → 得不偿失。

**决策**：删掉 `INTENT_KEYWORDS` 和关键词快速通道。Router 只保留问候检测（真正不需要 LLM 的），其余全交给 DeepSeek。

**代码变化**：`router.py` 从 ~370 行降到 ~280 行。

---

## 2. `\b` 词边界在中文里是摆设

**问题**：`_split_message` 用 `\b还有\b` 来检测多意图分隔词。

```
re.split(r"[。；;]|\b还有\b|\b另外\b|\b以及\b|如果", text)
```

**根因**：`\b` 匹配 `\w` 和非 `\w` 的边界。中文汉字不属于 `\w`（Python regex 中 `\w` = `[a-zA-Z0-9_]`），所以 `\b还有\b` 在中文里永远匹配不到。

**后果**："我的redmi坏了，还有我的thinkpad多久到" 整句拆不开 → 两个独立问题被当成一个丢给单一 Solver → 回复四不像。

**修复**：去掉 `\b` → `还有|另外|以及|如果`。

---

## 3. 对话历史注入是跨轮污染源

**问题**：`router_node` 和 `solver_node` 把前 6 条消息拼成 `"对话历史: ... 当前消息: ..."` 塞给 LLM。

**后果**：前一轮问过 ThinkPad → 下一轮问华为和 AirPods → LLM 看到历史里的 ThinkPad → 在物流回复里编造 ThinkPad 信息。

**修复**：删掉三个节点的历史注入。LangGraph checkpoint 已保留消息状态，不需要手动拼接。

---

## 4. `recent_order` 粘性上下文

**问题**：交互模式在用户提到订单号后，把 `recent_order` 粘在所有后续请求上。

**后果**：用户先说 Redmi → `recent_order=#12346` → 下一条"能开机是怎么回事" → Router 把 `#12346` 注入实体 → 技术问题被强制关联到 Redmi 订单。

**修复**：删掉 `recent_order` 机制。商品→订单匹配靠 DB 感知的实体提取（`find_orders_by_text`）。

---

## 5. 硬编码实体列表 = 永远不够用

**问题**：`_extract_entities` 和 `_infer_product` 只认识 `["手机", "耳机", "屏幕", "包裹", "订单", "电脑"]`。

**后果**："AirPods到哪了"、"ThinkPad多久到"、"Samsung能保修吗" 全都找不到对应订单。

**修复**：加入 `get_all_product_tokens()` 从 SQLite 加载真实商品名，`find_orders_by_text()` 按 token 匹配分数排序。再加 `SequenceMatcher` 模糊匹配容忍 typo（"thinlpad" ≈ "ThinkPad"）。

---

## 6. Pydantic DTO 膨胀

**问题**：`SolverMetrics` 有 `plan_tokens`、`plan_latency_ms`、`exec_tokens`、`exec_latency_ms`，但 Solver 从不填 plan_* 字段。

`SolverOutput.transfer_request` 设计是为 Solver→Router 退回协议，但当前代码只用来存 `{"llm_used": bool}`。

**修复**：`plan_*` 删除，`transfer_request: dict` → `llm_used: bool`。

---

## 7. 死配置项

删掉的未使用配置：
- `CONFIDENCE_FALLBACK_THRESHOLD` — 关键词路由删除后无用
- `MAX_TRANSFER_COUNT` — Solver 转交机制未实现
- `MAX_TOOL_CALLS_PER_TYPE` — 工具调用限次未实现
- `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION` — MVP 用文件 BM25 而非 ChromaDB

---

## 数据快照（2026-06-05）

| 指标 | 数值 |
|------|------|
| 测试数 | 68（7 个测试文件） |
| Router 代码行数 | ~280（清理后） |
| 支持意图 | 5（after_sales / tech_support / logistics / general_chat / multi_intent） |
| 数据库订单 | 8 条，覆盖 6 种售后场景 |
| 知识库文档 | 3（售后 FAQ / 退款政策 / 隐私政策） |
| 核心工具 | 3（kb_search / order_query / refund_check） |
| Graph 节点 | 8（router / solver / clarify / general_chat / monitor / retry_solver / respond / handoff） |

---

*最后更新：2026-06-05*
