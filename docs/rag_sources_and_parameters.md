# RAG Sources and Parameter Notes

## Public Sources

The MVP seed documents summarize public after-sales rules instead of inventing
policy text from scratch.

- 国家市场监督管理总局《网络购买商品七日无理由退货暂行办法》
  https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2023/art_26ca8fe29e184edd899fa0a7a060d935.html
- 中国网信网转载国家工商行政管理总局令第 90 号
  https://www.cac.gov.cn/2017-01/12/c_1120294339.htm

The demo policy layer deliberately separates public legal summaries from
project-specific rules. For example, the public source supports seven-day
no-reason return and actual-paid-price refund logic, while the demo adds a
30-day repair/exchange window to make customer-service routing testable.

## Retrieval Parameters

| Parameter | Initial value | Why this value |
|---|---:|---|
| `top_k` | 5 | Small enough for concise Solver/Monitor context, large enough to include policy plus FAQ hits. |
| BM25 weight | 0.72 | Precise policy words such as `退款时效` and `订单号格式` should dominate in customer-service search. |
| Lexical-semantic overlap weight | 0.28 | Keeps fuzzy phrasing such as `屏幕有问题能退吗` from losing to exact but irrelevant snippets. |
| BM25 `k1` | 1.4 | Standard moderate term-frequency saturation; avoids one repeated term overwhelming a chunk. |
| BM25 `b` | 0.65 | Mild length normalization because Markdown sections vary in length but are already chunked. |
| score threshold | 0.05 | MVP seed corpus is tiny; keep recall high, then let Monitor reject unsupported facts. |

These are target starting values. After implementation runs, tune them using
trace evidence:

- Retrieval hit rate for known policy queries.
- Monitor Check 3 false positives caused by missing KB evidence.
- Average number of injected chunks per run.
- P95 latency of `kb_search`.

## Benchmark Evidence

The reproducible benchmark is implemented in:

- `devma/evaluation/rag_benchmark.py`
- `tests/test_rag_benchmark.py`

Run it with:

```powershell
python -m devma.evaluation.rag_benchmark
```

The latest generated outputs are:

- `docs/evaluation/rag_benchmark_results.json`
- `docs/evaluation/rag_benchmark_results.md`

Current comparison, generated on 2026-06-04:

| Config | Hit@K | MRR | Irrelevant FP | P95 latency |
|---|---:|---:|---:|---:|
| `baseline_bm25_only` | 0.6000 | 0.6000 | 0.2000 | 1.228 ms |
| `mvp_hybrid` | 0.6000 | 0.6000 | 0.2000 | 1.216 ms |

Measured delta:

| Metric | `mvp_hybrid` vs `baseline_bm25_only` |
|---|---:|
| Hit@K | 0.00% |
| MRR | 0.00% |
| Irrelevant false-positive rate | 0.00% |
| P95 latency | -0.98% |

Conclusion: the current hybrid parameters are a tested MVP baseline, not a
proven relevance improvement. They keep the retrieval implementation
parameterized and measurable, but the benchmark does not yet justify claiming
accuracy gains.

Observed failure cases:

- `privacy_phone`: the query about phone/email masking did not retrieve
  `privacy_policy.md`.
- `irrelevant_weather`: an unrelated weather query still retrieved an
  after-sales document, so the threshold is too permissive for out-of-domain
  queries.

Next tuning target:

- Improve privacy-field recall before raising the global threshold.
- Add a small out-of-domain rejection rule or calibrated threshold so unrelated
  queries do not enter Solver evidence.
- Re-run the benchmark after each parameter/content change and only claim
  improvement when Hit@K/MRR or false-positive rate actually changes.
