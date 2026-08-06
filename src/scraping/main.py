"""
main.py
=======
Scraper de presence de Phlebotomus sergenti par commune (Maroc).

Pipeline :
  1. Charge le referentiel des communes (data/raw/communes_maroc_final.csv).
  2. Extrait le texte de chaque PDF local (data/external/articles/) et de
     chaque URL listee dans urls.txt.
  3. Cherche les mentions de P. sergenti et les communes citees a proximite.
  4. Produit UNE LIGNE PAR COMMUNE (mention trouvee ou non).

Sortie :
  outputs/processed/sergenti_par_commune_scraped.csv

Usage :
  python src/scraping/main.py

Garanties anti-hallucination : voir README (extraits litteraux, aucune
invention, sources en echec signalees dans les logs).
"""

import sys
import unicodedata
from pathlib import Path

import pandas as pd

# rendre les imports robustes quel que soit le cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching import match_communes
from text_extraction import extract_pdf, extract_url

ROOT = Path(__file__).resolve().parents[2]
COMMUNES_CSV = ROOT / "data" / "raw" / "communes_maroc_final.csv"
ARTICLES_DIR = ROOT / "data" / "external"
URLS_FILE = Path(__file__).resolve().parent / "urls.txt"
OUT_DIR = ROOT / "outputs" / "processed"
OUT_CSV = OUT_DIR / "sergenti_par_commune_scraped.csv"


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def load_commune_refs():
    df = pd.read_csv(COMMUNES_CSV)
    # dedoublonner sur "commune" pour avoir un index unique
    df = df.drop_duplicates(subset="commune", keep="first").reset_index(drop=True)
    print(f"[INFO] referentiel communes : {len(df)} uniques apres dedoublonnage")
    # {commune_norm: commune_original}
    norms = {}
    for _, row in df.iterrows():
        c = str(row["commune"]).strip()
        norms[strip_accents(c)] = c
    meta = df.set_index("commune")[["province", "region"]].to_dict("index")
    return df, norms, meta


def load_urls():
    if not URLS_FILE.exists():
        return []
    urls = []
    for line in URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip().strip('",')
        if line and line.startswith("http"):
            urls.append(line)
    return urls


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    communes_df, commune_norms, meta = load_commune_refs()

    # accumulateur : commune -> {annees, sources, extrait}
    hits = {}

    def record(source_label, matches):
        for m in matches:
            c = m["commune"]
            if c not in hits:
                hits[c] = {"annees": set(), "sources": set(), "extrait": ""}
            if m["annees"]:
                hits[c]["annees"].update(m["annees"].split(";"))
            hits[c]["sources"].add(source_label)
            if not hits[c]["extrait"]:
                hits[c]["extrait"] = m["extrait"]

    # ---- 1. PDF locaux ----
    if ARTICLES_DIR.exists():
        pdfs = [p for p in ARTICLES_DIR.glob("**/*.pdf") if not p.name.startswith("._")]
        print(f"[INFO] {len(pdfs)} PDF a analyser dans {ARTICLES_DIR}")
        for p in pdfs:
            text = extract_pdf(p)
            if text:
                record(p.name, match_communes(text, commune_norms))
    else:
        print(f"[INFO] dossier articles absent : {ARTICLES_DIR}")

    # ---- 2. URLs ----
    urls = load_urls()
    print(f"[INFO] {len(urls)} URL a analyser")
    for url in urls:
        text = extract_url(url)
        if text:
            record(url, match_communes(text, commune_norms))

    # ---- 3. une ligne par commune ----
    rows = []
    for _, row in communes_df.iterrows():
        c = str(row["commune"]).strip()
        h = hits.get(c)
        if h:
            rows.append({
                "commune": c,
                "province": row["province"],
                "region": row["region"],
                "sergenti_mentionne": "Oui",
                "annees": ";".join(sorted(h["annees"])),
                "sources": " | ".join(sorted(h["sources"])),
                "extrait_verification": h["extrait"],
                "remarque": "",
            })
        else:
            rows.append({
                "commune": c,
                "province": row["province"],
                "region": row["region"],
                "sergenti_mentionne": "Non",
                "annees": "",
                "sources": "",
                "extrait_verification": "",
                "remarque": "Aucune mention trouvee dans les sources fournies",
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    n_oui = int((out["sergenti_mentionne"] == "Oui").sum())
    print("=" * 60)
    print(f"Communes avec mention P. sergenti : {n_oui} / {len(out)}")
    print(f"Ecrit : {OUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
