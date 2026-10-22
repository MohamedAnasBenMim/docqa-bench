"""Métriques d'évaluation.

- exactitude          : la réponse contient tous les faits attendus
- recall@k            : au moins un passage de la bonne page est retrouvé
- précision citations : part des citations qui pointent vers une page attendue
- citation correcte   : la réponse cite au moins une page attendue, et aucune autre
- fidélité            : tous les nombres de la réponse figurent dans les passages cités
- hallucination       : réponse donnée à une question sans réponse dans les documents
"""

from __future__ import annotations

import re

from docqa.llm import NOT_FOUND
from docqa.retrieval import normalize


def norm(s: str) -> str:
    s = normalize(s).replace(",", ".").replace(" ", " ").replace("\xa0", " ")
    return re.sub(r"(?<=\d) (?=\d{3}\b)", "", s)  # 12 000 -> 12000


def page_of(citation: str) -> str:
    return citation.split("-c")[0]


def is_correct(answer: str, facts: list[str]) -> bool:
    a = norm(answer)
    return all(norm(f) in a for f in facts)


def recall_at_k(passages: list[dict], gold: list[str]) -> bool:
    return any(page_of(p.get("id", "")) in gold for p in passages)


def citation_precision(citations: list[str], gold: list[str]) -> float | None:
    if not citations:
        return None
    return sum(1 for c in citations if page_of(c) in gold or c in gold) / len(citations)


def citation_correct(citations: list[str], gold: list[str]) -> bool:
    if not citations:
        return False
    return all(page_of(c) in gold or c in gold for c in citations)


def faithful(answer: str, citations: list[str], passages: list[dict]) -> bool:
    cited_text = " ".join(norm(p.get("text", "")) for p in passages if p.get("id") in citations)
    numbers = re.findall(r"\d+(?:\.\d+)?", norm(re.sub(r"\[[^\]]*\]", "", answer)))
    return all(n in cited_text for n in numbers)


def abstained(answer: str) -> bool:
    return norm(NOT_FOUND)[:30] in norm(answer)
