"""Run once: create SQLite database and seed with mock orders."""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "orders.db")


def seed_orders(db_path: str = DB_PATH) -> int:
    """Explicit demo initializer; importing this module never overwrites a DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            user_name TEXT NOT NULL,
            product TEXT NOT NULL,
            price REAL NOT NULL,
            purchase_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            warranty_days INTEGER NOT NULL DEFAULT 365,
            has_receipt INTEGER NOT NULL DEFAULT 1,
            notes TEXT
        )
    """)

    orders = [
        (
            "#12345",
            "张先生",
            "Xiaomi 14 Pro 512GB 黑色",
            4299.00,
            "2026-04-23",
            "completed",
            365,
            1,
            "屏幕有坏点，用户反馈质量问题",
        ),
        (
            "#12346",
            "张先生",
            "Redmi Buds 5 Pro",
            399.00,
            "2026-05-01",
            "completed",
            365,
            1,
            None,
        ),
        (
            "#12347",
            "李女士",
            "iPhone 16 Pro Max 256GB 金色",
            9999.00,
            "2026-03-15",
            "completed",
            365,
            1,
            "已过30天退货期",
        ),
        (
            "#12348",
            "王先生",
            "华为 Mate 70 Pro 512GB",
            6999.00,
            "2026-05-10",
            "completed",
            365,
            1,
            "购买仅8天",
        ),
        (
            "#12349",
            "赵女士",
            "Samsung Galaxy S25 256GB",
            5499.00,
            "2026-04-01",
            "completed",
            365,
            0,
            "发票已丢失",
        ),
        (
            "#12350",
            "张先生",
            "AirPods Pro 3",
            1799.00,
            "2025-12-01",
            "completed",
            365,
            1,
            "购买超过5个月，超退货期",
        ),
        (
            "#1234",
            "刘先生",
            "Xiaomi 15 Pro 256GB 白色",
            4999.00,
            "2026-06-01",
            "completed",
            365,
            1,
            "手机，购买仅3天，在7天退货期内——用于演示退款资格",
        ),
        (
            "#5678",
            "陈女士",
            "ThinkPad X1 Carbon Gen 12",
            10999.00,
            "2026-05-20",
            "shipped",
            365,
            1,
            "笔记本电脑，已发货在途，物流查询演示——预计6月8日到达",
        ),
    ]

    cur.executemany(
        "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        orders,
    )

    conn.commit()
    conn.close()
    return len(orders)


if __name__ == "__main__":
    print(f"Seeded {seed_orders()} demo orders into {DB_PATH}")
