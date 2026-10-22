"""Agent routeur : choisit entre recherche documentaire (RAG) et calcul sur données (SQL).

Règle : une question chiffrée qui porte sur un indicateur présent dans la table
(et sur plusieurs années, une moyenne, un total ou une évolution) part en SQL ;
tout le reste part en RAG. Si le SQL ne sait pas répondre, on retombe sur le RAG.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .rag import RagAnswer, get_pipeline
from .retrieval import normalize
from .sql_tool import SqlTool

AGG_WORDS = [
    "evolution",
    "croissance",
    "progression",
    "variation",
    "moyenne",
    "moyen",
    "total",
    "cumul",
    "somme",
    "entre",
]


def needs_sql(question: str, tool: SqlTool) -> bool:
    words = set(re.findall(r"[a-z]+", normalize(question)))
    return tool.detect_metric(question) is not None and any(w in words for w in AGG_WORDS)


@lru_cache(maxsize=1)
def get_sql_tool() -> SqlTool:
    return SqlTool()


def answer(question: str, variant: str = "hybrid_rerank") -> RagAnswer:
    tool = get_sql_tool()
    if needs_sql(question, tool):
        try:
            res = tool.ask(question)
            if res["answer"]:
                return RagAnswer(
                    question=question,
                    answer=res["answer"],
                    citations=["sql:financials"],
                    passages=[{"sql": res["sql"], "rows": res["rows"]}],
                    variant=variant,
                    route="sql",
                )
        except Exception:  # noqa: BLE001 — repli volontaire sur le RAG
            pass
    return get_pipeline(variant).ask(question)
