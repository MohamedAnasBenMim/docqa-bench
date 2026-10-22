"""API FastAPI + petite interface web."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import answer
from .rag import VARIANTS

app = FastAPI(title="DocQA-Bench", version=__version__)
UI = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    variant: str = "hybrid_rerank"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.variant not in VARIANTS:
        raise HTTPException(400, f"variant doit être dans {VARIANTS}")
    res = answer(req.question, req.variant)
    return res.__dict__


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return UI
