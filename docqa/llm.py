"""Génération de la réponse.

- Avec LLM_API_KEY : appel d'une API compatible OpenAI (OpenAI, Mistral, Groq, Ollama...).
- Sans clé : mode extractif, qui renvoie la phrase la plus pertinente avec sa citation.
  Il sert de référence (baseline) et permet à la CI de tourner sans secret.
"""

from __future__ import annotations

import re

import httpx

from .config import settings
from .retrieval import Hit, tokenize

CITE_RE = re.compile(r"\[([\w\-.]+#p\d+-c\d+)\]")

SYSTEM_PROMPT = """Tu es un assistant d'analyse documentaire. Réponds UNIQUEMENT à partir des passages fournis.
Après chaque affirmation, cite le passage utilisé avec son identifiant entre crochets, ex. [doc#p2-c0].
Si les passages ne contiennent pas la réponse, réponds exactement : "Je ne trouve pas cette information dans les documents."
Réponds en français, en 1 à 3 phrases."""

NOT_FOUND = "Je ne trouve pas cette information dans les documents."


def extract_citations(answer: str) -> list[str]:
    return list(dict.fromkeys(CITE_RE.findall(answer)))


def _format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{c.id}] {c.text}" for c, _ in hits)


def generate_llm(question: str, hits: list[Hit]) -> str:
    resp = httpx.post(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        json={
            "model": settings.llm_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Passages :\n{_format_context(hits)}\n\nQuestion : {question}"},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_extractive(question: str, hits: list[Hit], min_overlap: int = 2) -> str:
    q = set(tokenize(question))
    best, best_score, best_id = None, 0.0, None
    for rank, (chunk, _) in enumerate(hits):
        doc_tokens = set(tokenize(chunk.doc.replace("_", " ")))
        for sent in re.split(r"(?<=[.!?;])\s+", chunk.text):
            sent_tokens = set(tokenize(sent))
            overlap = len(q & sent_tokens) + 0.5 * len((q & doc_tokens) - sent_tokens)
            score = overlap - 0.1 * rank  # léger avantage aux passages mieux classés
            if overlap >= min_overlap and score > best_score:
                best, best_score, best_id = sent.strip(), score, chunk.id
    return f"{best} [{best_id}]" if best else NOT_FOUND


def generate(question: str, hits: list[Hit]) -> str:
    if settings.llm_api_key:
        return generate_llm(question, hits)
    return generate_extractive(question, hits)
