"""Pipeline RAG configurable en trois variantes, pour pouvoir les comparer.

- naive         : recherche dense seule
- hybrid        : BM25 + dense fusionnés (RRF)
- hybrid_rerank : hybride + reranking des candidats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .config import settings
from .ingest import Chunk, load_corpus
from .llm import extract_citations, generate
from .rerank import build_reranker
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever

VARIANTS = ("naive", "hybrid", "hybrid_rerank")


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[str]
    passages: list[dict] = field(default_factory=list)
    variant: str = "hybrid_rerank"
    route: str = "rag"


class RagPipeline:
    def __init__(self, chunks: list[Chunk], variant: str = "hybrid_rerank"):
        if variant not in VARIANTS:
            raise ValueError(f"Variante inconnue : {variant}. Choix : {VARIANTS}")
        self.variant = variant
        self.chunks = chunks
        dense = DenseRetriever(chunks, settings.embedding_model)
        if variant == "naive":
            self.retriever = dense
        else:
            self.retriever = HybridRetriever([BM25Retriever(chunks), dense])
        self.reranker = build_reranker(settings.reranker_model, chunks) if variant == "hybrid_rerank" else None

    def retrieve(self, question: str, k: int | None = None):
        k = k or settings.top_k
        if self.reranker:
            candidates = self.retriever.search(question, settings.candidates)
            return self.reranker.rerank(question, candidates, k)
        return self.retriever.search(question, k)

    def ask(self, question: str) -> RagAnswer:
        hits = self.retrieve(question)
        answer = generate(question, hits)
        return RagAnswer(
            question=question,
            answer=answer,
            citations=extract_citations(answer),
            passages=[
                {"id": c.id, "doc": c.doc, "page": c.page, "score": round(s, 4), "text": c.text} for c, s in hits
            ],
            variant=self.variant,
        )


@lru_cache(maxsize=4)
def get_pipeline(variant: str = "hybrid_rerank") -> RagPipeline:
    return RagPipeline(load_corpus(settings.corpus_dir, settings.chunk_chars), variant)
