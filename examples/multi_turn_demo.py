"""Offline multi-turn demo; all databases are temporary, never user data."""

import json
import os
import tempfile
from pathlib import Path

os.environ["DEVMA_OFFLINE"] = "1"

from data.seed_orders import seed_orders
from devma.orchestration.graph import open_graph
from devma.tools import order_query


def main():
    messages = [
        "手机#1234到哪了；电脑#5678到哪了；耳机能退吗",
        "为什么需要订单号",
        "订单号是12346，另外手机开不了机怎么办",
        "手机重启试过了，还是开不了机",
    ]
    config = {"configurable": {"thread_id": "multi-turn-demo"}, "recursion_limit": 60}
    original_order_db = order_query.ORDER_DB_PATH
    try:
        with tempfile.TemporaryDirectory(prefix="devma-multi-turn-") as directory:
            order_query.ORDER_DB_PATH = str(Path(directory) / "orders.db")
            seed_orders(order_query.ORDER_DB_PATH)
            checkpoint = str(Path(directory) / "checkpoints.db")
            for text in messages:
                # Reopen SQLite on every turn to also demonstrate persistence.
                with open_graph(checkpoint) as graph:
                    result = graph.invoke(
                        {"messages": [{"role": "user", "content": text}]}, config
                    )
                print(
                    json.dumps(
                        {
                            "user": text,
                            "reply": result["final_output"],
                            "turn": result["turn_number"],
                            "tasks": [
                                {
                                    "id": key,
                                    "goal": entry["span"],
                                    "status": entry["status"],
                                    "revision": entry["revision"],
                                }
                                for key, entry in result["task_ledger"].items()
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
    finally:
        order_query.ORDER_DB_PATH = original_order_db


if __name__ == "__main__":
    main()
