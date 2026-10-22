"""Recherche : BM25 (lexicale), dense (embeddings) et hybride (fusion RRF)."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

STOPWORDS = set(
    "le la les un une des de du d l et en au aux a à est sont pour par sur dans que qui quel quelle quels "
    "quelles ce cette ces il elle on nous vous ils elles se sa son ses leur leurs ou plus combien comment "
    "the of and to in is what how many".split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(text)) if t not in STOPWORDS]


Hit = tuple[Chunk, float]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # le nom du document (ex. "rapport_2024") est indexé avec le texte : utile pour les questions datées
        self.bm25 = BM25Okapi([tokenize(c.doc.replace("_", " ") + " " + c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever:
    """Embeddings denses.

    - "lsa" : TF-IDF + SVD (analyse sémantique latente), 100 % hors ligne, utilisé en CI.
    - sinon : nom d'un modèle sentence-transformers (ex. "intfloat/multilingual-e5-small").
    """

    name = "dense"

    def __init__(self, chunks: list[Chunk], model: str = "lsa"):
        self.chunks = chunks
        self.model_name = model
        texts = [c.text for c in chunks]
        if model == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)
            tfidf = self.vec.fit_transform(texts)
            n_comp = max(2, min(128, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
            self.emb = self._norm(self.svd.fit_transform(tfidf))
            self._encode = lambda q: self._norm(self.svd.transform(self.vec.transform([q])))[0]
        else:
            from sentence_transformers import SentenceTransformer

            st = SentenceTransformer(model)
            self.emb = st.encode(texts, normalize_embeddings=True)
            self._encode = lambda q: st.encode([q], normalize_embeddings=True)[0]

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.emb @ self._encode(query)
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


class HybridRetriever:
    """Fusion des classements BM25 et dense par Reciprocal Rank Fusion (RRF)."""

    name = "hybrid"

    def __init__(self, retrievers: list, rrf_k: int = 60):
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (chunk, _) in enumerate(r.search(query, k * 3)):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[chunk.id] = chunk
        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(by_id[cid], score) for cid, score in best]
