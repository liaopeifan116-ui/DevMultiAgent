"""CLI: devma chat / eval / benchmark."""
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Fix Windows GBK encoding issues with Rich
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from devma.orchestration.graph import build_graph, save_evidence
from devma.tools.kb_search import get_kb_tool
from devma.config import EVIDENCE_DIR

app = typer.Typer(help="DevMultiAgent - Multi-Agent Customer Service System")
console = Console()


@app.command()
def chat(
    message: str = typer.Argument(..., help="User message in Chinese"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full trace"),
):
    """Run a single conversation through Router -> Solver -> Monitor."""
    kb = get_kb_tool()
    with console.status("[bold yellow]Loading knowledge base..."):
        n = kb.seed()
        console.print(f"[green]KB loaded: {n} chunks indexed")

    graph = build_graph()
    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 25}

    t_start = time.perf_counter()

    with console.status("[bold blue]Processing..."):
        initial = {"messages": [{"role": "user", "content": message}]}
        if verbose:
            for event in graph.stream(initial, config):
                node_name = list(event.keys())[0]
                console.print(f"  [dim]Node: {node_name}[/dim]")
        result = graph.invoke(initial, config)

    total_latency = round((time.perf_counter() - t_start) * 1000)
    final_output = result.get("final_output", "")

    console.print()
    console.print(Panel(Markdown(final_output), title="[bold green]Response", border_style="green"))

    metrics = result.get("metrics", {})
    table = Table(title="Run Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Intent", result.get("intent", "N/A"))
    table.add_row("Confidence", f"{result.get('confidence', 0):.2f}")
    table.add_row("Monitor Result", result.get("monitor_result", "N/A"))
    table.add_row("Handoff Needed", str(result.get("handoff_needed", False)))
    table.add_row("Total Latency", f"{total_latency}ms")
    table.add_row("LLM Used", str(metrics.get("llm_used", False)))
    table.add_row("Token Cost", str(metrics.get("solver_tokens", "N/A")))
    table.add_row("Tool Calls", str(len(result.get("tool_calls_made", []))))
    if result.get("monitor_result") in ("NEEDS_FIX", "REJECT"):
        table.add_row("Monitor Reason", result.get("monitor_reason", ""))

    console.print()
    console.print(table)

    evidence_path = save_evidence(result, run_id)
    console.print(f"\n[dim]Evidence saved: {evidence_path}[/dim]")


@app.command()
def eval(
    test_file: Optional[str] = typer.Option(None, "--tests", "-t", help="Path to test cases JSON"),
):
    """Run evaluation suite and generate report."""
    try:
        from devma.eval.test_cases import load_test_cases
        from devma.eval.judge import evaluate_run
    except ImportError as e:
        console.print(f"[red]Eval module not available: {e}[/red]")
        console.print("[yellow]The eval module (Task 6) has not been implemented yet.[/yellow]")
        raise typer.Exit(code=1)

    cases = load_test_cases(test_file)
    if not cases:
        console.print("[red]No test cases found.[/red]")
        return

    graph = build_graph()
    results = []

    with console.status(f"[bold blue]Running {len(cases)} test cases..."):
        for case in cases:
            run_id = str(uuid.uuid4())
            initial = {"messages": [{"role": "user", "content": case["query"]}]}
            result = graph.invoke(initial, {"configurable": {"thread_id": run_id}})
            eval_result = evaluate_run(case, result, run_id)
            results.append(eval_result)

    passed = sum(1 for r in results if r.get("overall") == "PASS")
    rejected = sum(1 for r in results if r.get("monitor_result") == "REJECT")

    console.print()
    console.print(Panel(
        f"Total: {len(results)} | Pass: [green]{passed}[/green] | Rejected: [red]{rejected}[/red]",
        title="Eval Summary",
    ))

    table = Table(title="Eval Results")
    table.add_column("Query", style="cyan")
    table.add_column("Monitor", style="yellow")
    table.add_column("Score", style="green")
    for r in results:
        table.add_row(r["query"][:40], r.get("monitor_result", ""), str(r.get("score", "")))
    console.print(table)

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = EVIDENCE_DIR / "reports" / f"eval_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "results": results}, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]Eval report: {report_path}[/dim]")


@app.command()
def benchmark():
    """Run benchmark metrics across test cases."""
    try:
        from devma.eval.test_cases import load_test_cases
    except ImportError as e:
        console.print(f"[red]Eval module not available: {e}[/red]")
        console.print("[yellow]The eval module (Task 6) has not been implemented yet.[/yellow]")
        console.print("[yellow]Falling back to built-in benchmark cases.[/yellow]")
        load_test_cases = None

    if load_test_cases is not None:
        cases = load_test_cases()
    else:
        cases = []

    if not cases:
        cases = [
            {"query": "手机屏幕有坏点能退吗", "expected_intent": "after_sales"},
            {"query": "我的包裹到哪了", "expected_intent": "logistics"},
            {"query": "手机开不了机怎么办", "expected_intent": "tech_support"},
        ]

    graph = build_graph()
    total_latency = 0
    total_tokens = 0
    intent_correct = 0
    rejections = 0
    handoffs = 0

    console.print("[bold]Benchmark Results[/bold]\n")

    for case in cases:
        run_id = str(uuid.uuid4())
        initial = {"messages": [{"role": "user", "content": case["query"]}]}
        t0 = time.perf_counter()
        result = graph.invoke(initial, {"configurable": {"thread_id": run_id}})
        elapsed = (time.perf_counter() - t0) * 1000
        total_latency += elapsed
        total_tokens += result.get("metrics", {}).get("solver_tokens", 0)

        if result.get("intent") == case.get("expected_intent"):
            intent_correct += 1
        if result.get("monitor_result") == "REJECT":
            rejections += 1
        if result.get("handoff_needed"):
            handoffs += 1

        console.print(f"  [cyan]{case['query'][:40]}[/cyan] -> intent={result.get('intent')}, "
                      f"monitor={result.get('monitor_result')}, latency={elapsed:.0f}ms")

    n = len(cases)
    console.print()
    console.print(f"  Intent Accuracy: [bold]{intent_correct}/{n} = {intent_correct/n*100:.1f}%[/bold]")
    console.print(f"  Avg Latency: [bold]{total_latency/n:.0f}ms[/bold]")
    console.print(f"  Avg Token Cost: [bold]{total_tokens/n:.0f} tokens[/bold]")
    console.print(f"  Rejection Rate: [bold red]{rejections}/{n} = {rejections/n*100:.1f}%[/bold red]")
    console.print(f"  Handoff Rate: [bold yellow]{handoffs}/{n} = {handoffs/n*100:.1f}%[/bold yellow]")


def main():
    app()


if __name__ == "__main__":
    main()
