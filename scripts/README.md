# Passer au corpus réel

1. Crée un dossier `data/real/` et places-y 3 à 5 rapports annuels (PDF) d'entreprises françaises cotées.
   Sources publiques : site « Investisseurs » de chaque entreprise, ou la base des documents réglementés de l'AMF.
   Choisis des documents dont la licence permet l'usage (et ne les commite pas s'ils sont volumineux :
   `data/real/` est ignoré par git, documente juste les liens de téléchargement ici).
2. Crée `data/real/financials.csv` avec les indicateurs clés (même format que `data/sample/financials.csv`).
3. Rédige `eval/questions_real.jsonl` (~100 questions) au format :
   `{"id": "r001", "type": "rag", "question": "...", "facts": ["..."], "gold": ["nom_du_pdf#p12"]}`
   Inclure ~15 % de questions sans réponse (`"type": "unanswerable"`) pour mesurer les hallucinations.
4. Lance :

```bash
export DOCQA_CORPUS=data/real DOCQA_SQL_CSV=data/real/financials.csv
python -m eval.run_eval --questions eval/questions_real.jsonl --out eval/results_real
```

5. Reporte le tableau de `eval/results_real/results.md` dans le README (section « Corpus réel »).
