"""Chargement des documents (PDF, Markdown, texte) et découpage en passages citables.

Chaque passage garde son document et sa page : c'est ce qui permet de citer la source exacte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PAGE_RE = re.compile(r"^##\s*Page\s+(\d+)[^\n]*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str  # ex. "nordelec_rapport_2025#p2-c0"
    doc: str  # nom du document (sans extension)
    page: int
    text: str


def _pages_from_markdown(text: str) -> list[tuple[int, str]]:
    matches = list(PAGE_RE.finditer(text))
    if not matches:
        return [(1, text)]
    pages = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        pages.append((int(m.group(1)), text[m.end() : end]))
    return pages


def _pages_from_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _split(text: str, max_chars: int) -> list[str]:
    """Découpe par paragraphes, puis par phrases si un paragraphe est trop long."""
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para or para.startswith("#"):
            continue
        if len(para) <= max_chars:
            parts.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if buf and len(buf) + len(sent) + 1 > max_chars:
                parts.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            parts.append(buf)
    return parts


def load_corpus(corpus_dir: Path, max_chars: int = 500) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.name.lower() == "readme.md":
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = _pages_from_pdf(path)
        elif suffix in {".md", ".txt"}:
            pages = _pages_from_markdown(path.read_text(encoding="utf-8"))
        else:
            continue
        for page, text in pages:
            for k, part in enumerate(_split(text, max_chars)):
                chunks.append(Chunk(id=f"{path.stem}#p{page}-c{k}", doc=path.stem, page=page, text=part))
    if not chunks:
        raise ValueError(f"Aucun document exploitable dans {corpus_dir}")
    return chunks
