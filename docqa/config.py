"""Configuration centralisée, pilotée par variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    corpus_dir: Path = field(default_factory=lambda: Path(os.getenv("DOCQA_CORPUS", ROOT / "data" / "sample")))
    sql_csv: Path = field(
        default_factory=lambda: Path(os.getenv("DOCQA_SQL_CSV", ROOT / "data" / "sample" / "financials.csv"))
    )
    chunk_chars: int = int(os.getenv("DOCQA_CHUNK_CHARS", "500"))
    top_k: int = int(os.getenv("DOCQA_TOP_K", "4"))
    candidates: int = int(os.getenv("DOCQA_CANDIDATES", "12"))
    # Embeddings : "lsa" (hors ligne, par défaut) ou un modèle sentence-transformers
    embedding_model: str = os.getenv("DOCQA_EMBEDDINGS", "lsa")
    # Reranker : "lexical" (hors ligne, par défaut) ou un cross-encoder sentence-transformers
    reranker_model: str = os.getenv("DOCQA_RERANKER", "lexical")
    # LLM compatible OpenAI (OpenAI, Mistral, Groq, Ollama...). Sans clé : mode extractif.
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")


settings = Settings()
