"""Hybrid knowledge retrieval: keyword BM25 plus lightweight reranking.

The MVP keeps vector retrieval as an extension point, but the tested path is
fast local retrieval over seed Markdown files. This avoids slow embedding model
startup while still supporting precise policy queries such as "退款时效".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from devma.config import DATA_DIR


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    content: str
    source: str
    tokens: list[str]


@dataclass(frozen=True)
class RetrievalParameters:
    bm25_weight: float = 0.72
    overlap_weight: float = 0.28
    k1: float = 1.4
    b: float = 0.65


class KBSearchTool:
    def __init__(self, parameters: RetrievalParameters | None = None):
        self.parameters = parameters or RetrievalParameters()
        self._chunks: list[DocumentChunk] = []
        self._idf: dict[str, float] = {}

    def seed(self) -> int:
        self._chunks = self._load_seed_chunks()
        self._idf = self._build_idf(self._chunks)
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.05) -> list[dict]:
        if not self._chunks:
            self.seed()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for chunk in self._chunks:
            bm25 = self._bm25(query_tokens, chunk)
            semantic = _lexical_semantic_overlap(query, chunk.content)
            score = (self.parameters.bm25_weight * bm25) + (self.parameters.overlap_weight * semantic)
            if score >= threshold:
                retrieval = "hybrid" if bm25 > 0 and semantic > 0 else "bm25" if bm25 > 0 else "semantic"
                scored.append((score, retrieval, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "score": round(min(score, 1.0), 3),
                "retrieval": retrieval,
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

    def search_with_rewrite(self, query: str, context: str = "", top_k: int = 5) -> list[dict]:
        rewritten = self.rewrite_query(query, context)
        results = self.search(rewritten, top_k=top_k)
        if not results:
            results = self.search(query, top_k=top_k)
        return results

    def _load_seed_chunks(self) -> list[DocumentChunk]:
        docs_dir = DATA_DIR / "kb" / "seed_docs"
        chunks: list[DocumentChunk] = []
        if not docs_dir.exists():
            return chunks

        for md_file in sorted(docs_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            sections = [s.strip() for s in re.split(r"\n(?=#{1,3}\s)|\n##\s", content) if s.strip()]
            for index, section in enumerate(sections):
                if len(section) < 20:
                    continue
                chunk_id = f"{md_file.stem}_{index}"
                chunks.append(DocumentChunk(chunk_id, section, md_file.name, _tokenize(section)))
        return chunks

    def _build_idf(self, chunks: list[DocumentChunk]) -> dict[str, float]:
        doc_count = max(len(chunks), 1)
        freq: dict[str, int] = {}
        for chunk in chunks:
            for token in set(chunk.tokens):
                freq[token] = freq.get(token, 0) + 1
        return {token: math.log((doc_count - count + 0.5) / (count + 0.5) + 1.0) for token, count in freq.items()}

    def _bm25(self, query_tokens: list[str], chunk: DocumentChunk) -> float:
        if not chunk.tokens:
            return 0.0
        avg_len = sum(len(item.tokens) for item in self._chunks) / max(len(self._chunks), 1)
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


def _tokenize(text: str) -> list[str]:
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    ascii_terms = re.findall(r"[A-Za-z0-9#_+-]{2,}", text.lower())
    tokens: list[str] = []
    for term in chinese_terms:
        tokens.append(term)
        for size in (2, 3, 4):
            tokens.extend(term[i : i + size] for i in range(max(len(term) - size + 1, 0)))
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
    if _tool is None:
        _tool = KBSearchTool()
    return _tool


def search_kb(query: str, context: str = "", top_k: int = 5) -> list[dict]:
    tool = get_kb_tool()
    if context:
        return tool.search_with_rewrite(query, context, top_k=top_k)
    return tool.search(query, top_k=top_k)
