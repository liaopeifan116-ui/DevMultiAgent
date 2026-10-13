"""CLI: devma chat / eval / benchmark."""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Fix Windows GBK encoding issues with Rich
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import typer
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from devma.config import EVIDENCE_DIR
from devma.orchestration.graph import build_graph, open_graph, save_evidence
from devma.tools.kb_search import get_kb_tool

app = typer.Typer(help="DevMultiAgent - Multi-Agent Customer Service System")
console = Console()


def _run_once(graph, initial, config, verbose=False):
    """Streaming is execution, not a preview: never invoke a second time."""
    if not verbose:
        return graph.invoke(initial, config)
    for event in graph.stream(initial, config, stream_mode="updates"):
        console.print(f"  [dim]Node: {', '.join(event.keys())}[/dim]")
    snapshot = graph.get_state(config)
    result = dict(snapshot.values)
    interrupts = [i for task in snapshot.tasks for i in task.interrupts]
    if interrupts:
        result["__interrupt__"] = interrupts
    return result


def _print_interrupts(result, thread_id):
    if not result.get("__interrupt__"):
        return False
    console.print(f"[yellow]Paused. Thread: {thread_id}[/yellow]")
    for item in result["__interrupt__"]:
        console.print(
            json.dumps(
                {"interrupt_id": item.id, "request": item.value},
                ensure_ascii=False,
                indent=2,
            )
        )
    console.print("Resume with: devma resume THREAD_ID --answer '{\"approved\":true}'")
    return True


@app.command()
def chat(
    message: str = typer.Argument(..., help="User message in Chinese"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full trace"),
    thread_id: Optional[str] = typer.Option(
        None, "--thread-id", help="Reuse a persisted conversation"
    ),
    pause: bool = typer.Option(
        False, "--pause", help="Pause before executing any business tools"
    ),
    pause_on_missing_info: bool = typer.Option(
        False, "--pause-on-missing-info", help="Pause to ask for order information"
    ),
):
    """Run a turn; reuse --thread-id to continue pending business tasks."""
    kb = get_kb_tool()
    with console.status("[bold yellow]Loading knowledge base..."):
        n = kb.seed()
        console.print(f"[green]KB loaded: {n} chunks indexed")

    run_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}

    t_start = time.perf_counter()

    with open_graph() as graph, console.status("[bold blue]Processing..."):
        snapshot = graph.get_state(config)
        if snapshot.next:
            console.print(
                "[red]This thread has unfinished work. Use resume/status instead of starting a new turn.[/red]"
            )
            raise typer.Exit(code=1)
        initial = {
            "messages": [{"role": "user", "content": message}],
            "pause_before_execution": pause,
            "pause_on_missing_info": pause_on_missing_info,
        }
        result = _run_once(graph, initial, config, verbose)
    console.print(f"[dim]Thread: {run_id}[/dim]")
    if _print_interrupts(result, run_id):
        return

    total_latency = round((time.perf_counter() - t_start) * 1000)
    final_output = result.get("final_output", "")

    console.print()
    console.print(
        Panel(
            Markdown(final_output), title="[bold green]Response", border_style="green"
        )
    )

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
    table.add_row("Reply Tokens (partial)", str(metrics.get("solver_tokens", "N/A")))
    table.add_row("Tool Calls", str(len(result.get("tool_calls_made", []))))
    if result.get("monitor_result") in ("NEEDS_FIX", "REJECT"):
        table.add_row("Monitor Reason", result.get("monitor_reason", ""))

    console.print()
    console.print(table)

    evidence_path = save_evidence(result, run_id)
    console.print(f"\n[dim]Evidence saved: {evidence_path}[/dim]")


@app.command()
def resume(
    thread_id: str,
    answer: Optional[str] = typer.Option(
        None, "--answer", help="JSON approval/order or interrupt-id map"
    ),
    answer_file: Optional[Path] = typer.Option(
        None,
        "--answer-file",
        exists=True,
        dir_okay=False,
        help="UTF-8 JSON file, avoids shell quoting",
    ),
):
    """Resume a SQLite-checkpointed run, including after a process restart."""
    try:
        if (answer is None) == (answer_file is None):
            raise typer.BadParameter("provide exactly one of --answer or --answer-file")
        supplied = json.loads(
            answer_file.read_text(encoding="utf-8") if answer_file else answer
        )
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("answer must be valid JSON") from exc
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    with open_graph() as graph:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            console.print("[red]No unfinished run for this thread.[/red]")
            raise typer.Exit(code=1)
        pending_interrupts = any(task.interrupts for task in snapshot.tasks)
        result = graph.invoke(
            Command(resume=supplied) if pending_interrupts else None, config
        )
    if not _print_interrupts(result, thread_id):
        console.print(Panel(Markdown(result.get("final_output", "")), title="Response"))
        console.print(f"[dim]Trace: {save_evidence(result, thread_id)}[/dim]")


@app.command()
def status(thread_id: str):
    """Inspect task statuses without executing model/tool calls."""
    with open_graph() as graph:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        console.print(
            json.dumps(
                {
                    "next": snapshot.next,
                    "turn_number": snapshot.values.get("turn_number", 0),
                    "tasks": [
                        {
                            "task_id": t["task_id"],
                            "solver": t["target_solver"],
                            "depends_on": t["depends_on"],
                            "status": t.get("status", "pending"),
                            "revision": t.get("revision", 1),
                            "missing_fields": t.get("missing_fields", []),
                            "clarification_id": t.get("clarification_id", ""),
                            "question": t.get("question", ""),
                        }
                        for t in snapshot.values.get("task_ledger", {}).values()
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


@app.command()
def eval(
    test_file: Optional[str] = typer.Option(
        None, "--tests", "-t", help="Path to test cases JSON"
    ),
):
    """Run evaluation suite and generate report."""
    try:
        from devma.eval.judge import evaluate_run
        from devma.eval.test_cases import load_test_cases
    except ImportError as e:
        console.print(f"[red]Eval module not available: {e}[/red]")
        console.print(
            "[yellow]The eval module (Task 6) has not been implemented yet.[/yellow]"
        )
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
    console.print(
        Panel(
            f"Total: {len(results)} | Pass: [green]{passed}[/green] | Rejected: [red]{rejected}[/red]",
            title="Eval Summary",
        )
    )

    table = Table(title="Eval Results")
    table.add_column("Query", style="cyan")
    table.add_column("Monitor", style="yellow")
    table.add_column("Score", style="green")
    for r in results:
        table.add_row(
            r["query"][:40], r.get("monitor_result", ""), str(r.get("score", ""))
        )
    console.print(table)

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = EVIDENCE_DIR / "reports" / f"eval_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": ts, "results": results}, f, ensure_ascii=False, indent=2
        )
    console.print(f"\n[dim]Eval report: {report_path}[/dim]")


@app.command()
def benchmark():
    """Run benchmark metrics across test cases."""
    try:
        from devma.eval.test_cases import load_test_cases
    except ImportError as e:
        console.print(f"[red]Eval module not available: {e}[/red]")
        console.print(
            "[yellow]The eval module (Task 6) has not been implemented yet.[/yellow]"
        )
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

        console.print(
            f"  [cyan]{case['query'][:40]}[/cyan] -> intent={result.get('intent')}, "
            f"monitor={result.get('monitor_result')}, latency={elapsed:.0f}ms"
        )

    n = len(cases)
    console.print()
    console.print(
        f"  Intent Accuracy: [bold]{intent_correct}/{n} = {intent_correct / n * 100:.1f}%[/bold]"
    )
    console.print(f"  Avg Latency: [bold]{total_latency / n:.0f}ms[/bold]")
    console.print(f"  Avg Token Cost: [bold]{total_tokens / n:.0f} tokens[/bold]")
    console.print(
        f"  Rejection Rate: [bold red]{rejections}/{n} = {rejections / n * 100:.1f}%[/bold red]"
    )
    console.print(
        f"  Handoff Rate: [bold yellow]{handoffs}/{n} = {handoffs / n * 100:.1f}%[/bold yellow]"
    )


def main():
    app()


if __name__ == "__main__":
    main()
