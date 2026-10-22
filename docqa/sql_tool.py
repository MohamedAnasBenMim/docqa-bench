"""Mode text-to-SQL pour les questions chiffrées.

Les données tabulaires (CSV) sont chargées dans une base SQLite en mémoire.
- Avec LLM : le modèle écrit la requête SQL, qui est validée (SELECT uniquement, table autorisée).
- Sans LLM : traduction par règles (indicateur + années), suffisante pour la démo et la CI.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path

import httpx

from .config import settings
from .retrieval import normalize

TABLE = "financials"

# synonymes français -> colonne
METRICS = {
    "revenue_meur": ["chiffre d'affaires", "chiffre d affaires", "ca ", "revenus", "ventes"],
    "ebitda_meur": ["ebitda"],
    "net_income_meur": ["resultat net", "benefice"],
    "net_debt_meur": ["dette nette", "endettement"],
    "employees": ["effectif", "collaborateurs", "salaries", "employes"],
    "rnd_pct_revenue": ["r&d", "recherche et developpement"],
    "dividend_eur": ["dividende"],
}
UNITS = {
    "revenue_meur": "M€",
    "ebitda_meur": "M€",
    "net_income_meur": "M€",
    "net_debt_meur": "M€",
    "employees": "collaborateurs",
    "rnd_pct_revenue": "% du CA",
    "dividend_eur": "€ par action",
}


class SqlTool:
    def __init__(self, csv_path: Path | None = None):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        path = csv_path or settings.sql_csv
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
        self.columns = cols
        self.conn.execute(f"CREATE TABLE {TABLE} ({', '.join(c + ' REAL' for c in cols)})")
        self.conn.executemany(
            f"INSERT INTO {TABLE} VALUES ({', '.join('?' for _ in cols)})",
            [[float(r[c]) for c in cols] for r in rows],
        )
        self.years = sorted(int(r["year"]) for r in rows)

    # ---------- détection ----------
    @staticmethod
    def detect_metric(question: str) -> str | None:
        q = normalize(question) + " "
        for col, words in METRICS.items():
            if any(normalize(w) in q for w in words):
                return col
        return None

    # ---------- génération SQL ----------
    def to_sql_rules(self, question: str) -> str | None:
        metric = self.detect_metric(question)
        if not metric:
            return None
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", question)]
        q = normalize(question)
        if len(years) >= 2 and any(w in q for w in ["evolution", "croissance", "progression", "variation", "entre"]):
            y1, y2 = min(years), max(years)
            return (
                f"SELECT ROUND((b.{metric} - a.{metric}) * 100.0 / a.{metric}, 1) AS variation_pct, "
                f"a.{metric} AS debut, b.{metric} AS fin FROM {TABLE} a, {TABLE} b "
                f"WHERE a.year = {y1} AND b.year = {y2}"
            )
        if any(w in q for w in ["moyenne", "moyen"]):
            return f"SELECT ROUND(AVG({metric}), 2) AS moyenne FROM {TABLE}"
        if any(w in q for w in ["total", "cumul", "somme"]):
            return f"SELECT SUM({metric}) AS total FROM {TABLE}"
        year = years[0] if years else self.years[-1]
        return f"SELECT {metric} FROM {TABLE} WHERE year = {year}"

    def to_sql_llm(self, question: str) -> str:
        prompt = (
            f"Table SQLite `{TABLE}` avec les colonnes : {', '.join(self.columns)} "
            f"(montants en millions d'euros, années {self.years[0]}-{self.years[-1]}).\n"
            f"Écris UNE requête SELECT qui répond à : {question}\nRéponds uniquement avec le SQL."
        )
        resp = httpx.post(
            f"{settings.llm_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={"model": settings.llm_model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        sql = resp.json()["choices"][0]["message"]["content"]
        return re.sub(r"^```(sql)?|```$", "", sql.strip(), flags=re.MULTILINE).strip()

    @staticmethod
    def validate(sql: str) -> None:
        s = sql.strip().rstrip(";").lower()
        if not s.startswith("select") or ";" in s:
            raise ValueError("Seules les requêtes SELECT uniques sont autorisées")
        if re.search(r"\b(insert|update|delete|drop|alter|attach|pragma|create)\b", s):
            raise ValueError("Mot-clé SQL interdit")
        tables = set(re.findall(r"\b(?:from|join)\s+(\w+)", s))
        if tables - {TABLE}:
            raise ValueError(f"Table non autorisée : {tables - {TABLE}}")

    # ---------- exécution ----------
    def ask(self, question: str) -> dict:
        sql = self.to_sql_llm(question) if settings.llm_api_key else self.to_sql_rules(question)
        if not sql:
            return {"answer": None, "sql": None}
        self.validate(sql)
        cur = self.conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        metric = self.detect_metric(question)
        unit = UNITS.get(metric, "")
        if rows and len(cols) == 1:
            value = rows[0][0]
            value = int(value) if float(value).is_integer() else value
            answer = f"{value} {unit}".strip()
        elif rows and "variation_pct" in cols:
            v, a, b = rows[0]
            answer = f"{v:+} % (de {a:g} à {b:g} {unit})".strip()
        else:
            answer = str([dict(zip(cols, r)) for r in rows])
        return {"answer": f"{answer} [sql:{TABLE}]", "sql": sql, "rows": [dict(zip(cols, r)) for r in rows]}
