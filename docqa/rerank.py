"""Reranking des passages candidats.

- "lexical" : score de recouvrement pondéré par IDF + bonus de proximité des termes (hors ligne, CI).
- sinon : cross-encoder sentence-transformers (ex. "BAAI/bge-reranker-v2-m3").
"""

from __future__ import annotations

import math

from .ingest import Chunk
from .retrieval import Hit, tokenize


class LexicalReranker:
    name = "lexical"

    def __init__(self, chunks: list[Chunk]):
        n = len(chunks)
        df: dict[str, int] = {}
        for c in chunks:
            for t in set(tokenize(c.text)):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n + 1) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, text: str) -> float:
        q = set(tokenize(query))
        toks = tokenize(text)
        if not q or not toks:
            return 0.0
        covered = q.intersection(toks)
        coverage = sum(self.idf.get(t, 1.0) for t in covered) / sum(self.idf.get(t, 1.0) for t in q)
        # proximité : plus petite fenêtre contenant les termes couverts
        positions = [i for i, t in enumerate(toks) if t in covered]
        span = (positions[-1] - positions[0] + 1) if len(positions) > 1 else 1
        proximity = len(covered) / span if covered else 0.0
        return coverage + 0.3 * proximity

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        rescored = [(c, self.score(query, c.text)) for c, _ in hits]
        return sorted(rescored, key=lambda h: h[1], reverse=True)[:k]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        scores = self.model.predict([(query, c.text) for c, _ in hits])
        rescored = [(c, float(s)) for (c, _), s in zip(hits, scores)]
        return sorted(rescored, key=lambda h: h[1], reverse=True)[:k]


def build_reranker(model: str, chunks: list[Chunk]):
    return LexicalReranker(chunks) if model == "lexical" else CrossEncoderReranker(model)
