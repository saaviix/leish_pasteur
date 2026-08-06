"""
matching.py
===========
Cherche les mentions de Phlebotomus sergenti dans un texte et, autour de chaque
mention (fenetre WINDOW caracteres), repere quelles communes marocaines (du
referentiel) et quelles annees sont citees.

Anti-hallucination : on ne renvoie que des extraits LITTERAUX du texte source.
"""

import re
import unicodedata

WINDOW = 400  # caracteres de part et d'autre d'une mention (ajustable)

SERGENTI_PATTERNS = [
    r"phlebotomus\s+sergenti",
    r"ph\.?\s*sergenti",
    r"p\.?\s*sergenti",
    r"phl[ée]botome\s+sergenti",
]

YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def find_sergenti_spans(text: str):
    """Retourne les (start, end) de chaque mention de sergenti."""
    low = strip_accents(text)
    spans = []
    for pat in SERGENTI_PATTERNS:
        for m in re.finditer(pat, low):
            spans.append((m.start(), m.end()))
    return sorted(set(spans))


def match_communes(text: str, commune_norms: dict):
    """
    text          : texte brut d'une source
    commune_norms : {commune_norm: commune_original} du referentiel

    Retourne une liste de dicts, un par (commune trouvee pres d'une mention) :
      {commune, annees, extrait}
    """
    if not text or len(text) < 30:
        return []

    spans = find_sergenti_spans(text)
    if not spans:
        return []

    low = strip_accents(text)
    results = {}
    for (s, e) in spans:
        w_start = max(0, s - WINDOW)
        w_end = min(len(text), e + WINDOW)
        window_low = low[w_start:w_end]
        window_raw = text[w_start:w_end].replace("\n", " ").strip()
        years = sorted(set(YEAR_RE.findall(window_raw)))

        for c_norm, c_orig in commune_norms.items():
            if len(c_norm) < 4:
                continue  # trop court -> trop de faux positifs
            if c_norm in window_low:
                if c_orig not in results:
                    results[c_orig] = {"commune": c_orig, "annees": set(), "extrait": window_raw}
                results[c_orig]["annees"].update(years)

    out = []
    for r in results.values():
        r["annees"] = ";".join(sorted(r["annees"]))
        r["extrait"] = r["extrait"][:700]
        out.append(r)
    return out
