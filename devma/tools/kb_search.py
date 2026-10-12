"""Lexical hybrid retrieval: BM25 plus term coverage and dual-query rank fusion.

The MVP keeps vector retrieval as an extension point, but the tested path is
fast local retrieval over seed Markdown files. This avoids slow embedding model
startup while still supporting precise policy queries such as "退款时效".
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from threading import RLock

from devma.config import DATA_DIR


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    content: str
    source: str
    tokens: list[str]
    heading: str = ""


@dataclass(frozen=True)
class RetrievalParameters:
    bm25_weight: float = 0.72
    overlap_weight: float = 0.28
    k1: float = 1.4
    b: float = 0.65

    def __post_init__(self):
        if (
            self.bm25_weight < 0
            or self.overlap_weight < 0
            or not math.isclose(self.bm25_weight + self.overlap_weight, 1.0)
        ):
            raise ValueError("retrieval weights must be non-negative and sum to one")
        if self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("invalid BM25 k1/b")


class KBSearchTool:
    def __init__(self, parameters: RetrievalParameters | None = None):
        self.parameters = parameters or RetrievalParameters()
        self._chunks: list[DocumentChunk] = []
        self._idf: dict[str, float] = {}
        self._seed_lock = RLock()

    def seed(self) -> int:
        with self._seed_lock:
            self._chunks = self._load_seed_chunks()
            self._idf = self._build_idf(self._chunks)
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.05) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        with self._seed_lock:
            if not self._chunks:
                self.seed()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for chunk in self._chunks:
            bm25 = self._bm25(query_tokens, chunk)
            coverage = _lexical_semantic_overlap(query, chunk.content)
            score = (self.parameters.bm25_weight * bm25) + (
                self.parameters.overlap_weight * coverage
            )
            if score >= threshold:
                retrieval = (
                    "hybrid"
                    if bm25 > 0 and coverage > 0
                    else "bm25"
                    if bm25 > 0
                    else "term_coverage"
                )
                scored.append((score, retrieval, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "score": round(score, 6),
                "retrieval": retrieval,
                "heading": chunk.heading,
                "content_hash": hashlib.sha256(chunk.content.encode()).hexdigest(),
                "bm25_score": round(self._bm25(query_tokens, chunk), 6),
                "term_coverage": round(
                    _lexical_semantic_overlap(query, chunk.content), 6
                ),
            }
            for score, retrieval, chunk in scored[:top_k]
        ]

    def rewrite_query(self, query: str, context: str = "") -> str:
        if not context:
            return query
        terms = _tokenize(f"{query} {context}")
        seen: set[str] = set()
        compact = []
        for term in terms:
            if term not in seen:
                compact.append(term)
                seen.add(term)
        return " ".join(compact[:10]) or query

    def search_with_rewrite(
        self, query: str, context: str = "", top_k: int = 5
    ) -> list[dict]:
        rewritten = self.rewrite_query(query, context)
        # Recall BOTH branches, then deduplicate/fuse. Rewriting never replaces
        # the original identifier-bearing query, even if the rewrite returns hits.
        branches = {"original": self.search(query, top_k=max(top_k * 2, 10))}
        if rewritten != query:
            branches["rewritten"] = self.search(rewritten, top_k=max(top_k * 2, 10))
        merged = {}
        for branch, hits in branches.items():
            for rank, hit in enumerate(hits, 1):
                record = merged.setdefault(
                    hit["id"], {**hit, "fusion_score": 0.0, "query_branches": []}
                )
                record["fusion_score"] += 1 / (60 + rank)
                record["query_branches"].append(branch)
        return sorted(merged.values(), key=lambda r: (-r["fusion_score"], r["id"]))[
            :top_k
        ]

    def _load_seed_chunks(self) -> list[DocumentChunk]:
        docs_dir = DATA_DIR / "kb" / "seed_docs"
        chunks: list[DocumentChunk] = []
        if not docs_dir.exists():
            return chunks

        for md_file in sorted(docs_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            sections = [
                s.strip() for s in re.split(r"\n(?=#{1,6}\s)", content) if s.strip()
            ]
            headings: dict[int, str] = {}
            for index, section in enumerate(sections):
                header = re.match(r"^(#{1,6})\s+(.+)", section)
                if header:
                    depth = len(header.group(1))
                    headings = {k: v for k, v in headings.items() if k < depth}
                    headings[depth] = header.group(2)
                heading = " / ".join(headings.values())
                if len(section) < 20:
                    continue
                for part_index, part in enumerate(_split_long_section(section)):
                    chunk_id = f"{md_file.stem}_{index}" + (
                        f"_{part_index}" if part_index else ""
                    )
                    chunks.append(
                        DocumentChunk(
                            chunk_id, part, md_file.name, _tokenize(part), heading
                        )
                    )
        return chunks

    def _build_idf(self, chunks: list[DocumentChunk]) -> dict[str, float]:
        doc_count = max(len(chunks), 1)
        freq: dict[str, int] = {}
        for chunk in chunks:
            for token in set(chunk.tokens):
                freq[token] = freq.get(token, 0) + 1
        return {
            token: math.log((doc_count - count + 0.5) / (count + 0.5) + 1.0)
            for token, count in freq.items()
        }

    def _bm25(self, query_tokens: list[str], chunk: DocumentChunk) -> float:
        if not chunk.tokens:
            return 0.0
        avg_len = sum(len(item.tokens) for item in self._chunks) / max(
            len(self._chunks), 1
        )
        k1 = self.parameters.k1
        b = self.parameters.b
        score = 0.0
        for token in query_tokens:
            tf = chunk.tokens.count(token)
            if tf == 0:
                continue
            idf = self._idf.get(token, 0.0)
            denom = tf + k1 * (1 - b + b * len(chunk.tokens) / max(avg_len, 1))
            score += idf * (tf * (k1 + 1)) / denom
        return score / max(len(query_tokens), 1)


_tool: KBSearchTool | None = None
_tool_lock = RLock()


def _split_long_section(
    text: str, max_chars: int = 1200, overlap: int = 120
) -> list[str]:
    """Heading-first, paragraph-preferred splitting. Bounds are characters, not tokens."""
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return parts


def _tokenize(text: str) -> list[str]:
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    ascii_terms = re.findall(r"[A-Za-z0-9#_+-]{2,}", text.lower())
    tokens: list[str] = []
    for term in chinese_terms:
        tokens.append(term)
        for size in (2, 3, 4):
            tokens.extend(
                term[i : i + size] for i in range(max(len(term) - size + 1, 0))
            )
    tokens.extend(ascii_terms)
    return [token for token in tokens if token.strip()]


def _lexical_semantic_overlap(query: str, content: str) -> float:
    query_tokens = set(_tokenize(query))
    content_tokens = set(_tokenize(content))
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


def get_kb_tool() -> KBSearchTool:
    global _tool
    with _tool_lock:
        if _tool is None:
            _tool = KBSearchTool()
    return _tool


def search_kb(query: str, context: str = "", top_k: int = 5) -> list[dict]:
    tool = get_kb_tool()
    if context:
        return tool.search_with_rewrite(query, context, top_k=top_k)
    return tool.search(query, top_k=top_k)
