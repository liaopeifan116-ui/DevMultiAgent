from devma.evaluation.rag_benchmark import (
    RetrievalConfig,
    evaluate_retrieval_configs,
    get_default_cases,
)


def test_rag_benchmark_compares_configs():
    cases = get_default_cases()
    baseline = RetrievalConfig(
        name="baseline",
        bm25_weight=1.0,
        overlap_weight=0.0,
        k1=1.2,
        b=0.75,
        threshold=0.05,
        top_k=3,
    )
    mvp = RetrievalConfig(
        name="mvp",
        bm25_weight=0.72,
        overlap_weight=0.28,
        k1=1.4,
        b=0.65,
        threshold=0.05,
        top_k=5,
    )

    report = evaluate_retrieval_configs([baseline, mvp], cases)

    assert report["case_count"] == len(cases)
    assert report["configs"]["baseline"]["hit_rate_at_k"] >= 0
    assert report["configs"]["mvp"]["mrr"] >= 0
    assert "hit_rate_at_k_delta_pct" in report["comparisons"]["mvp_vs_baseline"]
    assert len(report["cases"]) == len(cases) * 2
