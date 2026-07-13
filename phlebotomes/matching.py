"""
matching.py
--------------
Recherche, dans un texte brut, les mentions de "Phlebotomus sergenti"
et regarde quelles communes marocaines sont citees a proximite, ainsi
que les annees (dates) mentionnees dans le meme voisinage.

Aucune inference/hallucination : on ne renvoie que des passages
reellement presents dans le texte source, avec leur position.
"""
import re
import unicodedata

# Variantes courantes rencontrees dans la litterature (FR/EN/latin abrege)
VECTOR_PATTERN = re.compile(
    r"(Ph(?:l[ée]botome|l[ée]botomus)?\.?\s*sergenti|P\.\s*sergenti)",
    re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

WINDOW = 400  # caracteres avant/apres chaque mention du vecteur


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def build_flexible_pattern(name: str) -> re.Pattern:
    """
    Construit un pattern regex tolerant aux accents/tirets/apostrophes
    pour un nom de commune, ex: "Béni Mellal" matchera aussi
    "Beni Mellal", "BENI-MELLAL", "Béni  Mellal", etc.
    """
    ACCENT_MAP = {
        "a": "[aàâ]", "e": "[eéèêë]", "i": "[iîï]",
        "o": "[oôö]", "u": "[uùûü]", "c": "[cç]",
    }
    parts = []
    for ch in _strip_accents(name).lower():
        if ch.isalnum():
            parts.append(ACCENT_MAP.get(ch, re.escape(ch)))
        elif ch.isspace() or ch in "-'":
            parts.append(r"[\s\-']+")
        # autre ponctuation : ignoree
    pattern = r"\b" + "".join(parts) + r"\b"
    return re.compile(pattern, re.IGNORECASE)


def find_vector_mentions(texte: str):
    """Renvoie la liste des positions (debut, fin) ou 'sergenti' apparait."""
    return [(m.start(), m.end()) for m in VECTOR_PATTERN.finditer(texte)]


def extract_years_in_window(texte: str, start: int, end: int):
    fenetre = texte[max(0, start - WINDOW): end + WINDOW]
    return sorted(set(m.group(0) for m in YEAR_PATTERN.finditer(fenetre)))


def get_snippet(texte: str, start: int, end: int, taille=220) -> str:
    """Extrait un court passage autour de la mention, pour verification humaine."""
    s = max(0, start - taille // 2)
    e = min(len(texte), end + taille // 2)
    snippet = texte[s:e].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet


def communes_in_window(texte: str, start: int, end: int, communes_patterns):
    """
    Cherche quelles communes (liste de (commune, pattern)) apparaissent
    dans la fenetre de texte autour d'une mention du vecteur.
    Renvoie la liste des noms de communes trouves dans cette fenetre.
    """
    fenetre = texte[max(0, start - WINDOW): end + WINDOW]
    trouvees = []
    for commune, pattern in communes_patterns:
        if pattern.search(fenetre):
            trouvees.append(commune)
    return trouvees
