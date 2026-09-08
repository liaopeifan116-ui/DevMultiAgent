"""SQLite order lookup tool."""
import re
import sqlite3
from difflib import SequenceMatcher
from devma.config import ORDER_DB_PATH


def query_order(order_id: str) -> dict | None:
    conn = sqlite3.connect(ORDER_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def query_orders_by_user(user_name: str) -> list[dict]:
    conn = sqlite3.connect(ORDER_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "SELECT * FROM orders WHERE user_name = ? ORDER BY purchase_date DESC",
        (user_name,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def query_orders_by_product(product_keyword: str) -> list[dict]:
    """Find recent orders whose product name or notes mention a product keyword."""
    conn = sqlite3.connect(ORDER_DB_PATH)
    conn.row_factory = sqlite3.Row
    pattern = f"%{product_keyword}%"
    cur = conn.execute(
        """
        SELECT * FROM orders
        WHERE product LIKE ? OR notes LIKE ?
        ORDER BY purchase_date DESC
        """,
        (pattern, pattern),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def find_orders_by_text(user_text: str) -> list[dict]:
    """Search orders by matching product name tokens against user text.

    Each order is scored by how many of its product-name tokens appear in
    the user's message. Results are sorted best-match first. When the user
    says "Xiaomi 15 Pro", the order containing all three tokens ranks above
    one that only matches "Xiaomi".
    """
    conn = sqlite3.connect(ORDER_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT id, product FROM orders")
    all_orders = [dict(r) for r in cur.fetchall()]
    conn.close()

    text_lower = user_text.lower()
    # Extract individual words (alphanumeric sequences) from user text for fuzzy matching
    user_words = re.findall(r"[a-zA-Z0-9]+", text_lower)

    scored: list[tuple[int, str]] = []
    for order in all_orders:
        product = order["product"]
        tokens = [t for t in product.split() if len(t) >= 2]
        if not tokens:
            continue
        hits = 0
        for t in tokens:
            t_lower = t.lower()
            if t_lower in text_lower:
                hits += 1
            elif len(t) >= 4:
                # Fuzzy match against each word in user text (typo tolerance)
                for word in user_words:
                    if len(word) >= 3:
                        ratio = SequenceMatcher(None, t_lower, word).ratio()
                        if ratio >= 0.65:
                            hits += 1
                            break
        if hits > 0:
            scored.append((hits, order["id"]))

    # Sort by score descending, take top matches
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []

    # Return all orders that tie with the top score, plus any that scored
    # at least 2 (multi-token matches) — these are likely intended targets
    top_score = scored[0][0]
    if top_score >= 2:
        threshold = 2
    else:
        threshold = 1

    matched_ids = [oid for score, oid in scored if score >= threshold]
    matched: list[dict] = []
    for oid in matched_ids:
        full = query_order(oid)
        if full and full not in matched:
            matched.append(full)
    return matched


def get_all_product_tokens() -> list[str]:
    """Return all significant tokens from product names for entity extraction."""
    conn = sqlite3.connect(ORDER_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT product FROM orders")
    products = [dict(r)["product"] for r in cur.fetchall()]
    conn.close()

    tokens: list[str] = []
    for p in products:
        for token in p.split():
            if len(token) >= 2:
                tokens.append(token)
    return list(dict.fromkeys(tokens))  # deduplicate, preserve order
