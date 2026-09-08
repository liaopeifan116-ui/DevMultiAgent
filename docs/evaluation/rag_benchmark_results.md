# RAG Benchmark Results

- Generated at: 2026-06-04T23:44:47
- Cases: 5

## Summary

| Config | Hit@K | MRR | Irrelevant FP | Avg Latency ms | P95 Latency ms |
|---|---:|---:|---:|---:|---:|
| baseline_bm25_only | 0.6000 | 0.6000 | 0.2000 | 1.158 | 1.228 |
| mvp_hybrid | 0.6000 | 0.6000 | 0.2000 | 1.139 | 1.216 |

## Comparisons

### mvp_hybrid_vs_baseline_bm25_only
- hit_rate_at_k_delta_pct: 0.0%
- mrr_delta_pct: 0.0%
- irrelevant_false_positive_rate_delta_pct: 0.0%
- p95_latency_ms_delta_pct: -0.98%

## Case Details

| Config | Case | Hit | Rank | Top Sources | Latency ms |
|---|---|---:|---:|---|---:|
| baseline_bm25_only | refund_deadline | True | 1 | refund_policy.md, refund_policy.md, after_sales_faq.md | 1.028 |
| baseline_bm25_only | screen_defect | True | 1 | after_sales_faq.md, privacy_policy.md, refund_policy.md | 1.228 |
| baseline_bm25_only | privacy_phone | False |  | after_sales_faq.md, after_sales_faq.md, refund_policy.md | 1.203 |
| baseline_bm25_only | invoice_refund | True | 1 | after_sales_faq.md, refund_policy.md, after_sales_faq.md | 1.214 |
| baseline_bm25_only | irrelevant_weather | False |  | after_sales_faq.md | 1.118 |
| mvp_hybrid | refund_deadline | True | 1 | refund_policy.md, refund_policy.md, after_sales_faq.md, after_sales_faq.md, after_sales_faq.md | 0.982 |
| mvp_hybrid | screen_defect | True | 1 | after_sales_faq.md, privacy_policy.md, refund_policy.md | 1.216 |
| mvp_hybrid | privacy_phone | False |  | after_sales_faq.md, after_sales_faq.md | 1.209 |
| mvp_hybrid | invoice_refund | True | 1 | after_sales_faq.md, refund_policy.md, after_sales_faq.md, refund_policy.md | 1.215 |
| mvp_hybrid | irrelevant_weather | False |  | after_sales_faq.md | 1.071 |
