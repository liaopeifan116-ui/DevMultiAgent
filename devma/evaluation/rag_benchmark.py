"""Reproducible RAG retrieval benchmark for MVP parameter choices."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from devma.config import PROJECT_ROOT
from devma.tools.kb_search import KBSearchTool, RetrievalParameters


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    bm25_weight: float
    overlap_weight: float
    k1: float
    b: float
    threshold: float
    top_k: int


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    query: str
    expected_sources: tuple[str, ...]
    note: str


def get_default_cases() -> list[RetrievalCase]:
    return [
        RetrievalCase(
            id="refund_deadline",
            query="超过7天还能退款吗",
            expected_sources=("refund_policy.md", "after_sales_faq.md"),
            note="政策时效类查询，应命中退款/售后规则。",
        ),
        RetrievalCase(
            id="screen_defect",
            query="手机屏幕坏点可以退货吗",
            expected_sources=("after_sales_faq.md", "refund_policy.md"),
            note="质量问题口语化查询，应命中售后 FAQ 或退款规则。",
        ),
        RetrievalCase(
            id="privacy_phone",
            query="手机号和邮箱会怎么脱敏",
            expected_sources=("privacy_policy.md",),
            note="字段级脱敏查询，应命中隐私规则。",
        ),
        RetrievalCase(
            id="invoice_refund",
            query="没有发票还能申请售后吗",
            expected_sources=("after_sales_faq.md", "refund_policy.md"),
            note="凭证/售后限制查询，应命中售后相关文档。",
        ),
        RetrievalCase(
            id="irrelevant_weather",
            query="今天上海天气怎么样",
            expected_sources=(),
            note="无关查询，用于观察误召回。",
        ),
    ]


def default_configs() -> list[RetrievalConfig]:
    return [
        RetrievalConfig(
            name="baseline_bm25_only",
            bm25_weight=1.0,
            overlap_weight=0.0,
            k1=1.2,
            b=0.75,
            threshold=0.05,
            top_k=3,
        ),
        RetrievalConfig(
            name="mvp_hybrid",
            bm25_weight=0.72,
            overlap_weight=0.28,
            k1=1.4,
            b=0.65,
            threshold=0.05,
            top_k=5,
        ),
    ]


def evaluate_retrieval_configs(configs: list[RetrievalConfig], cases: list[RetrievalCase]) -> dict:
    per_case: list[dict] = []
    summaries: dict[str, dict] = {}

    for config in configs:
        tool = KBSearchTool(
            RetrievalParameters(
                bm25_weight=config.bm25_weight,
                overlap_weight=config.overlap_weight,
                k1=config.k1,
                b=config.b,
            )
        )
        tool.seed()

        hits = []
        reciprocal_ranks = []
        false_positive_flags = []
        latencies_ms = []

        for case in cases:
            started = time.perf_counter()
            results = tool.search(case.query, top_k=config.top_k, threshold=config.threshold)
            latency_ms = (time.perf_counter() - started) * 1000
            latencies_ms.append(latency_ms)

            sources = [result["source"] for result in results]
            expected = set(case.expected_sources)
            rank = _first_expected_rank(sources, expected)
            hit = rank is not None if expected else len(results) == 0
            mrr = 1.0 / rank if rank else 0.0
            false_positive = not expected and bool(results)

            hits.append(1.0 if hit else 0.0)
            reciprocal_ranks.append(mrr)
            false_positive_flags.append(1.0 if false_positive else 0.0)
            per_case.append(
                {
                    "config": config.name,
                    "case_id": case.id,
                    "query": case.query,
                    "expected_sources": list(case.expected_sources),
                    "top_sources": sources,
                    "top_scores": [result["score"] for result in results],
                    "hit": hit,
                    "first_expected_rank": rank,
                    "latency_ms": round(latency_ms, 3),
                    "note": case.note,
                }
            )

        summaries[config.name] = {
            "parameters": asdict(config),
            "hit_rate_at_k": round(mean(hits), 4),
            "mrr": round(mean(reciprocal_ranks), 4),
            "irrelevant_false_positive_rate": round(mean(false_positive_flags), 4),
            "avg_latency_ms": round(mean(latencies_ms), 3),
            "p95_latency_ms": round(_percentile(latencies_ms, 95), 3),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(cases),
        "configs": summaries,
        "comparisons": _compare_to_first_config(configs, summaries),
        "cases": per_case,
    }


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rag_benchmark_results.json"
    md_path = output_dir / "rag_benchmark_results.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MVP RAG retrieval benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "evaluation",
        help="Directory for JSON and Markdown benchmark outputs.",
    )
    args = parser.parse_args()
    report = evaluate_retrieval_configs(default_configs(), get_default_cases())
    json_path, md_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _first_expected_rank(sources: list[str], expected_sources: set[str]) -> int | None:
    if not expected_sources:
        return None
    for index, source in enumerate(sources, start=1):
        if source in expected_sources:
            return index
    return None


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def _compare_to_first_config(configs: list[RetrievalConfig], summaries: dict[str, dict]) -> dict:
    if len(configs) < 2:
        return {}
    baseline_name = configs[0].name
    baseline = summaries[baseline_name]
    comparisons = {}
    for config in configs[1:]:
        current = summaries[config.name]
        comparisons[f"{config.name}_vs_{baseline_name}"] = {
            "hit_rate_at_k_delta_pct": _delta_pct(current["hit_rate_at_k"], baseline["hit_rate_at_k"]),
            "mrr_delta_pct": _delta_pct(current["mrr"], baseline["mrr"]),
            "irrelevant_false_positive_rate_delta_pct": _delta_pct(
                current["irrelevant_false_positive_rate"],
                baseline["irrelevant_false_positive_rate"],
            ),
            "p95_latency_ms_delta_pct": _delta_pct(current["p95_latency_ms"], baseline["p95_latency_ms"]),
        }
    return comparisons


def _delta_pct(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if current == 0 else 100.0
    return round(((current - baseline) / baseline) * 100, 2)


def _to_markdown(report: dict) -> str:
    lines = [
        "# RAG Benchmark Results",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Cases: {report['case_count']}",
        "",
        "## Summary",
        "",
        "| Config | Hit@K | MRR | Irrelevant FP | Avg Latency ms | P95 Latency ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["configs"].items():
        lines.append(
            "| {name} | {hit:.4f} | {mrr:.4f} | {fp:.4f} | {avg:.3f} | {p95:.3f} |".format(
                name=name,
                hit=summary["hit_rate_at_k"],
                mrr=summary["mrr"],
                fp=summary["irrelevant_false_positive_rate"],
                avg=summary["avg_latency_ms"],
                p95=summary["p95_latency_ms"],
            )
        )
    lines.extend(["", "## Comparisons", ""])
    for name, comparison in report["comparisons"].items():
        lines.append(f"### {name}")
        for key, value in comparison.items():
            lines.append(f"- {key}: {value}%")
        lines.append("")

    lines.extend(
        [
            "## Case Details",
            "",
            "| Config | Case | Hit | Rank | Top Sources | Latency ms |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for case in report["cases"]:
        rank = case["first_expected_rank"] or ""
        lines.append(
            "| {config} | {case_id} | {hit} | {rank} | {sources} | {latency:.3f} |".format(
                config=case["config"],
                case_id=case["case_id"],
                hit=str(case["hit"]),
                rank=rank,
                sources=", ".join(case["top_sources"]),
                latency=case["latency_ms"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
