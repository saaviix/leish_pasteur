"""
main.py — Scraper de presence de Phlebotomus sergenti par commune (Maroc)
----------------------------------------------------------------------------
CE QUE FAIT CE SCRIPT :
  1. Lit tous les PDF du dossier ./articles
  2. Telecharge et lit toutes les URLs listees dans ./urls.txt (une par ligne,
     PubMed/PMC/pages web/PDF direct - les deux formats sont geres)
  3. Cherche les mentions de "Phlebotomus sergenti" dans chaque texte
  4. Pour chaque mention, regarde quelles communes marocaines (referentiel
     communes_maroc_final.csv) sont citees a proximite (~400 caracteres)
  5. Recupere les annees mentionnees dans le meme voisinage (dates possibles
     de presence rapportee)
  6. Exporte un CSV avec UNE LIGNE PAR COMMUNE (les 1503 communes du
     referentiel), qu'il y ait ou non une mention trouvee.

IMPORTANT (pas d'hallucination) :
  - Si aucune mention n'est trouvee pour une commune, la ligne reste vide
    sur les champs Sergenti_mentionne / Dates / Sources, avec la remarque
    "Aucune mention trouvee dans les sources fournies".
  - Les extraits (colonne Extrait_verification) sont copies TELS QUELS du
    texte source, jamais reformules ou completes, pour que tu puisses
    verifier toi-meme chaque ligne "Oui".
"""
import os
import glob
import pandas as pd

from text_extraction import extract_text_from_pdf_file, extract_text_from_url
from matching import (
    build_flexible_pattern,
    find_vector_mentions,
    extract_years_in_window,
    get_snippet,
    communes_in_window,
)

ARTICLES_DIR = "./articles"
URLS_FILE = "./urls.txt"
COMMUNES_REF = "./communes_maroc_final.csv"
OUTPUT_CSV = "./output_phlebotomes_sergenti.csv"


def charger_sources():
    """Renvoie une liste de (nom_source, texte_brut)."""
    sources = []

    pdfs = glob.glob(os.path.join(ARTICLES_DIR, "*.pdf"))
    print(f"[INFO] {len(pdfs)} PDF trouves dans {ARTICLES_DIR}")
    for path in pdfs:
        print(f"  -> lecture {path}")
        texte = extract_text_from_pdf_file(path)
        sources.append((os.path.basename(path), texte))

    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, encoding="utf-8") as f:
            urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print(f"[INFO] {len(urls)} URLs trouvees dans {URLS_FILE}")
        for url in urls:
            print(f"  -> telechargement {url}")
            texte = extract_text_from_url(url)
            sources.append((url, texte))
    else:
        print(f"[INFO] Pas de fichier {URLS_FILE} trouve, on continue sans URL.")

    return sources


def analyser_sources(sources, communes_df):
    """
    Cherche dans chaque source les mentions du vecteur + communes associees.
    Renvoie un dict {commune_id: [ {source, dates, extrait}, ... ]}
    """
    communes_patterns = [
        (row.commune, build_flexible_pattern(row.commune))
        for row in communes_df.itertuples()
    ]

    resultats = {row.id: [] for row in communes_df.itertuples()}
    id_by_commune = {row.commune: row.id for row in communes_df.itertuples()}

    for nom_source, texte in sources:
        if not texte:
            print(f"  [ATTENTION] texte vide pour {nom_source}, ignore.")
            continue
        mentions = find_vector_mentions(texte)
        print(f"  {nom_source} : {len(mentions)} mention(s) de sergenti")
        for start, end in mentions:
            communes_trouvees = communes_in_window(texte, start, end, communes_patterns)
            if not communes_trouvees:
                continue  # mention du vecteur sans commune identifiable a proximite
            annees = extract_years_in_window(texte, start, end)
            extrait = get_snippet(texte, start, end)
            for commune in communes_trouvees:
                cid = id_by_commune[commune]
                resultats[cid].append({
                    "source": nom_source,
                    "dates": ", ".join(annees) if annees else "",
                    "extrait": extrait,
                })
    return resultats


def construire_csv(communes_df, resultats):
    lignes = []
    for row in communes_df.itertuples():
        trouvailles = resultats.get(row.id, [])
        if trouvailles:
            sources = "; ".join(sorted(set(t["source"] for t in trouvailles)))
            dates = "; ".join(sorted(set(d for t in trouvailles for d in t["dates"].split(", ") if d)))
            extraits = " || ".join(t["extrait"] for t in trouvailles[:3])  # 3 extraits max pour rester lisible
            lignes.append({
                "commune": row.commune,
                "province": row.province,
                "region": row.region,
                "sergenti_mentionne": "Oui",
                "dates_mentionnees": dates,
                "sources": sources,
                "extrait_verification": extraits,
                "remarque": "",
            })
        else:
            lignes.append({
                "commune": row.commune,
                "province": row.province,
                "region": row.region,
                "sergenti_mentionne": "Non",
                "dates_mentionnees": "",
                "sources": "",
                "extrait_verification": "",
                "remarque": "Aucune mention trouvee dans les sources fournies",
            })
    return pd.DataFrame(lignes)


def main():
    communes_df = pd.read_csv(COMMUNES_REF)
    sources = charger_sources()
    if not sources:
        print("[STOP] Aucune source (ni PDF ni URL) a analyser. "
              "Mets des PDF dans ./articles ou des URLs dans ./urls.txt")
        return
    resultats = analyser_sources(sources, communes_df)
    df_out = construire_csv(communes_df, resultats)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    n_oui = (df_out["sergenti_mentionne"] == "Oui").sum()
    print(f"\n[TERMINE] {n_oui} commune(s) avec au moins une mention trouvee, "
          f"sur {len(df_out)} communes au total.")
    print(f"Resultat : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
