"""Interactive multi-turn chat loop for DevMultiAgent."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from devma.orchestration.graph import open_graph
from devma.tools.kb_search import get_kb_tool

console = Console()


def update_user_context(context: dict, text: str) -> list[tuple[str, str]]:
    """Extract lightweight product -> order mappings from user-provided facts."""
    captured: list[tuple[str, str]] = []
    pairs = re.findall(
        r"(手机|电脑|耳机|订单|包裹)[^，。；;]*?订单号(?:是|为)?\s*#?(\d{4,})", text
    )
    for product, order_id in pairs:
        normalized = f"#{order_id}"
        context.setdefault("product_orders", {})[product] = normalized
        captured.append((product, normalized))

    pending_product = ""
    for clause in re.split(r"[，。；;]", text):
        product_match = re.search(r"(手机|电脑|耳机|包裹)", clause)
        order_match = re.search(r"订单号(?:是|为)?\s*#?(\d{4,})", clause)
        if product_match:
            pending_product = product_match.group(1)
        if order_match:
            normalized = f"#{order_match.group(1)}"
            product = pending_product or "订单"
            context.setdefault("product_orders", {})[product] = normalized
            if (product, normalized) not in captured:
                captured.append((product, normalized))
            pending_product = ""
    return captured


def main() -> None:
    console.print("[bold blue]Initializing DevMultiAgent...[/bold blue]")
    kb = get_kb_tool()
    with console.status("[bold yellow]Loading knowledge base..."):
        n = kb.seed()
        console.print(f"[green]KB loaded: {n} chunks indexed")

    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default=str(uuid.uuid4()))
    args = parser.parse_args()
    with open_graph() as graph:
        _chat_loop(graph, args.thread_id)


def _chat_loop(graph, session_id):
    console.print(
        f"[dim]Thread: {session_id}; use --thread-id to reopen this conversation.[/dim]"
    )
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 60}
    user_context = graph.get_state(config).values.get(
        "user_context", {"product_orders": {}}
    )
    last_metrics: dict = {}

    console.print(
        "[bold green]Ready! Type your message (or /quit to exit, /stats to show last metrics)[/bold green]\n"
    )

    while True:
        try:
            user_input = console.input("[bold cyan]You> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if user_input.lower() == "/stats":
            show_stats(last_metrics)
            continue

        snapshot = graph.get_state(config)
        if user_input.startswith("/resume "):
            try:
                answer = json.loads(user_input[len("/resume ") :])
            except json.JSONDecodeError:
                console.print("[red]Please provide valid JSON.[/red]")
                continue
            if not snapshot.next:
                console.print("[yellow]No pending work to resume.[/yellow]")
                continue
            initial = (
                Command(resume=answer)
                if any(t.interrupts for t in snapshot.tasks)
                else None
            )
        else:
            if snapshot.next:
                console.print(
                    "[yellow]This conversation is paused. Use /resume JSON first.[/yellow]"
                )
                continue
            update_user_context(user_context, user_input)
            initial = {
                "messages": [{"role": "user", "content": user_input}],
                "user_context": user_context,
            }

        started_at = time.perf_counter()
        with console.status("[bold blue]Processing..."):
            result = graph.invoke(initial, config)
        latency = round((time.perf_counter() - started_at) * 1000)
        if result.get("__interrupt__"):
            for item in result["__interrupt__"]:
                console.print(
                    json.dumps(
                        {"interrupt_id": item.id, "request": item.value},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            console.print(
                '[yellow]Use /resume {"order_id":"#1234"} or /resume {"approved":true}.[/yellow]'
            )
            continue

        console.print()
        console.print(
            Panel(
                Markdown(result.get("final_output", "")),
                title="[bold green]Response",
                border_style="green",
            )
        )

        metrics = result.get("metrics", {})
        last_metrics = {
            "intent": result.get("intent"),
            "confidence": result.get("confidence", 0),
            "monitor_result": result.get("monitor_result"),
            "latency": latency,
            "llm_used": metrics.get("llm_used", False),
            "tokens": metrics.get("solver_tokens"),
            "context": user_context,
        }
        print_run_summary(last_metrics)


def show_stats(metrics: dict) -> None:
    if not metrics:
        console.print("[dim]No stats yet. Send a message first.[/dim]")
        return
    console.print(
        f"[dim]Last: intent={metrics.get('intent')}, "
        f"confidence={metrics.get('confidence', 0):.2f}, "
        f"monitor={metrics.get('monitor_result')}, "
        f"latency={metrics.get('latency')}ms, "
        f"llm_used={metrics.get('llm_used')}, "
        f"tokens={metrics.get('tokens')}, "
        f"context={metrics.get('context')}[/dim]"
    )


def print_run_summary(metrics: dict) -> None:
    console.print(
        f"[dim]Intent: {metrics['intent']} | "
        f"Confidence: {metrics['confidence']:.2f} | "
        f"Monitor: {metrics['monitor_result']} | "
        f"Latency: {metrics['latency']}ms | "
        f"LLM Used: {metrics['llm_used']} | "
        f"Tokens: {metrics['tokens']}[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
