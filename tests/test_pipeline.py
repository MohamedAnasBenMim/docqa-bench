import pytest
from fastapi.testclient import TestClient

from docqa.agent import answer, get_sql_tool, needs_sql
from docqa.api import app
from docqa.config import settings
from docqa.ingest import load_corpus
from docqa.rag import RagPipeline
from docqa.sql_tool import SqlTool


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(settings.corpus_dir, settings.chunk_chars)


def test_chunks_keep_document_and_page(chunks):
    assert chunks
    assert all("#p" in c.id and c.page >= 1 for c in chunks)
    assert not any(c.text.startswith("—") for c in chunks), "les titres de page ne doivent pas fuiter"


@pytest.mark.parametrize("variant", ["naive", "hybrid", "hybrid_rerank"])
def test_retrieval_finds_right_page(chunks, variant):
    hits = RagPipeline(chunks, variant).retrieve("Qui préside le conseil d'administration ?")
    assert any(c.id.startswith("nordelec_gouvernance_risques#p1") for c, _ in hits)


def test_answer_has_citation():
    res = answer("Combien d'ingénieurs compte le centre de R&D de Grenoble fin 2025 ?")
    assert res.route == "rag"
    assert "210" in res.answer
    assert res.citations and res.citations[0].startswith("nordelec_rapport_2025#p3")


def test_router_sends_aggregations_to_sql():
    tool = get_sql_tool()
    assert needs_sql("Évolution du chiffre d'affaires entre 2023 et 2025 ?", tool)
    assert not needs_sql("Combien d'ingénieurs compte le centre de Grenoble ?", tool)


def test_sql_variation():
    res = SqlTool().ask("Quelle est l'évolution du chiffre d'affaires entre 2023 et 2025 ?")
    assert res["rows"][0]["variation_pct"] == 20.4


@pytest.mark.parametrize(
    "sql", ["DROP TABLE financials", "SELECT * FROM sqlite_master", "SELECT 1; DELETE FROM financials"]
)
def test_sql_guardrails(sql):
    with pytest.raises(ValueError):
        SqlTool.validate(sql)


def test_api():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "Quel est le principal facteur de risque ?"})
    assert r.status_code == 200 and r.json()["citations"]
    assert client.post("/ask", json={"question": "test", "variant": "inconnue"}).status_code == 400
