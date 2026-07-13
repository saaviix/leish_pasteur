"""
text_extraction.py
----------------------
Fonctions pour recuperer le texte brut d'un PDF local ou d'une URL
(page web classique OU URL pointant directement vers un PDF, comme
souvent sur PubMed/PMC).
Aucune generation de texte : uniquement extraction telle quelle depuis
la source. Si l'extraction echoue, on renvoie une chaine vide et on le
signale dans les logs (jamais de contenu invente).
"""
import requests
import pdfplumber
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (research literature scraper)"}
TIMEOUT = 25


def extract_text_from_pdf_file(path: str) -> str:
    """Extrait tout le texte d'un PDF local, page par page."""
    texte = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                texte.append(t)
    except Exception as e:
        print(f"  [ERREUR PDF] {path} : {e}")
        return ""
    return "\n".join(texte)


def _is_pdf_response(resp) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return "pdf" in ctype or resp.url.lower().endswith(".pdf")


def extract_text_from_url(url: str) -> str:
    """
    Recupere le texte d'une URL :
      - si c'est un PDF (Content-Type ou extension .pdf) -> extraction PDF
      - sinon -> on suppose du HTML, on extrait le texte visible (BeautifulSoup)
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERREUR URL] {url} : {e}")
        return ""

    if _is_pdf_response(resp):
        tmp_path = "/tmp/_scraper_tmp.pdf"
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        return extract_text_from_pdf_file(tmp_path)

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")
