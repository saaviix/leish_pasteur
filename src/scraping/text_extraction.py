"""
text_extraction.py
==================
Extraction de texte brut depuis :
  - des fichiers PDF locaux (via pdfplumber)
  - des URLs (pages HTML via BeautifulSoup, ou PDF telecharge)

Aucune invention : si une source echoue, on log l'erreur et on renvoie "".
"""

import io
from pathlib import Path

import requests

try:
    import pdfplumber
except ImportError:  # message clair si dependance manquante
    pdfplumber = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SandflyResearchBot/1.0)"}
TIMEOUT = 30


def extract_pdf(path: Path) -> str:
    """Texte d'un PDF local. '' si echec (ex. PDF scanne = image)."""
    if pdfplumber is None:
        print("[ERREUR PDF] pdfplumber non installe (pip install pdfplumber)")
        return ""
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"[ERREUR PDF] {path.name}: {e}")
        return ""


def extract_pdf_bytes(content: bytes, label: str) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"[ERREUR PDF] {label}: {e}")
        return ""


def extract_url(url: str) -> str:
    """Texte d'une URL (HTML ou PDF). '' si echec."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERREUR URL] {url}: {e}")
        return ""

    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return extract_pdf_bytes(r.content, url)

    if BeautifulSoup is None:
        print("[ERREUR URL] beautifulsoup4 non installe (pip install beautifulsoup4)")
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())
