"""
fix_mun_pop.py
===============
Recupere les donnees de population par commune, qui vivaient dans
data/raw/mun_pop.csv sous un nom trompeur : ce fichier est en realite un
classeur Excel (.xlsx) a 4 feuilles (IDs, mun_pop, marital_status, Sheet7),
illisible tel quel par pandas.read_csv() et jusqu'ici inutilise par tout le
pipeline.

  1. Lit la feuille 'mun_pop' du classeur source (data/raw/mun_pop_source.xlsx,
     renomme depuis l'ancien mun_pop.csv -- rien n'est perdu, les feuilles
     marital_status/Sheet7 restent disponibles pour un futur travail
     socio-economique, hors perimetre de cette refonte).
  2. Reconcilie le nom de commune contre le referentiel
     communes_maroc_final.csv (meme logique que clean_lct.py : exact d'abord,
     puis flou global -- il n'y a pas de province dans mun_pop pour restreindre
     la recherche comme pour les cas LCT -- avec un seuil strict pour eviter
     les faux positifs).
  3. Ecrit un vrai CSV, exploitable pour calculer des TAUX d'incidence
     (cas / 100k habitants) plutot que des comptages bruts.

Entrees :
  data/raw/mun_pop_source.xlsx (feuille 'mun_pop')
  data/raw/communes_maroc_final.csv

Sortie :
  data/raw/mun_pop.csv

Usage :
  python src/data_prep/fix_mun_pop.py
"""

import pandas as pd

import config
import geo_matching

FUZZY_CUTOFF = 0.90  # meme raisonnement que clean_lct.py : recherche globale
                      # (pas de province pour restreindre) -> seuil strict


def main() -> None:
    if not config.MUN_POP_SOURCE_XLSX.exists():
        raise FileNotFoundError(
            f"{config.MUN_POP_SOURCE_XLSX} introuvable. "
            f"C'est le classeur Excel source (ex-mun_pop.csv)."
        )

    pop = pd.read_excel(config.MUN_POP_SOURCE_XLSX, sheet_name="mun_pop")
    communes = pd.read_csv(config.COMMUNES_CSV)
    communes_idx = geo_matching.build_commune_index(communes)
    uniqueness = geo_matching.build_global_uniqueness(communes_idx)

    keys = communes_idx["commune_key"].tolist()
    names = communes_idx["commune"].tolist()
    ids = communes_idx["id"].tolist()
    provs = communes_idx["province"].tolist()
    regions = communes_idx["region"].tolist()

    pop["_key"] = pop["mun"].map(config.norm_key)

    results = []
    for key, raw_name in zip(pop["_key"], pop["mun"]):
        if key in keys:
            i = keys.index(key)
            results.append({"commune_id": ids[i], "commune": names[i], "province": provs[i],
                             "region": regions[i], "match_method": "exact", "match_score": 1.0})
            continue
        idx, score = geo_matching.fuzzy_match(key, keys, cutoff=FUZZY_CUTOFF)
        if idx is not None and uniqueness.get(keys[idx], 0) == 1:
            results.append({"commune_id": ids[idx], "commune": names[idx], "province": provs[idx],
                             "region": regions[idx], "match_method": "fuzzy", "match_score": round(score, 3)})
        else:
            results.append({"commune_id": None, "commune": None, "province": None,
                             "region": None, "match_method": "unmatched", "match_score": 0.0})

    match_df = pd.DataFrame(results)
    pop_cols = [c for c in pop.columns if c.startswith("pop_")]
    out = pd.concat([match_df, pop[["mun"] + pop_cols].rename(columns={"mun": "mun_raw"})], axis=1)

    n_exact = int((out["match_method"] == "exact").sum())
    n_fuzzy = int((out["match_method"] == "fuzzy").sum())
    n_unmatched = int((out["match_method"] == "unmatched").sum())
    print(f"[INFO] reconciliation population : {n_exact} exactes, {n_fuzzy} floues "
          f"(seuil={FUZZY_CUTOFF}), {n_unmatched} non reconciliees sur {len(out)} lignes "
          f"({100 * (n_exact + n_fuzzy) / len(out):.1f}% de taux de jointure)")

    matched = out[out["match_method"] != "unmatched"].copy()
    dup_ids = matched["commune_id"].duplicated(keep=False)
    if dup_ids.any():
        print(f"[WARN] {int(dup_ids.sum())} lignes population pointent vers le meme commune_id "
              f"(doublons dans la source ou collision de matching) -- gardees telles quelles, "
              f"a dedupliquer en aval selon le besoin")

    # ---- fix mega-villes (trouve par audit) : 6 communes du referentiel
    # (Casablanca, Rabat, Fes, Marrakech, Sale, Tanger) n'avaient AUCUNE ligne
    # de population car le classeur source les decoupe en arrondissements
    # jamais recomposes. Casablanca a un marqueur fiable (colonne 'city',
    # 21 lignes taggees "Casablanca"). Pour les 5 autres, pas de marqueur --
    # les arrondissements ont ete identifies par nom (litterature
    # administrative marocaine) et RETENUS UNIQUEMENT si (a) aucune collision
    # avec une commune distincte deja matchee dans le referentiel, ET (b) la
    # somme obtenue est coherente (±10%) avec la population reelle connue de
    # la ville (RGPH 2014) -- sinon la liste est probablement incomplete ou
    # ambigue et NON appliquee (mieux vaut une donnee manquante qu'une donnee
    # fausse). Rabat (somme 503 375 vs ~577 827 reel, -13%, liste incomplete
    # -- ex. "Agdal" seul est ambigu, existe ailleurs au Maroc) et Fes
    # ("M�dina" ambigu, peut appartenir a n'importe laquelle des grandes
    # villes) ne passent pas ce test -> restent documentees, non fabriquees.
    CITY_TAG_GROUPS = {"Casablanca": None}  # via colonne 'city' de la source
    CITY_NAME_GROUPS = {
        # verifie : 930 226 vs ~947 952 reel (-1.9%)
        "Tanger-Assilah": {"commune": "Tanger", "names": ["Bni Makada", "Mghogha", "Souani"]},
        # verifie : 996 643 vs ~928 850 reel (+7.3%)
        "Marrakech": {"commune": "Marrakech",
                      "names": ["Ménara", "Gueliz", "Sidi Youssef Ben Ali", "Marrakech-Médina", "Annakhil"]},
        # verifie : 938 863 vs ~890 403 reel (+5.4%)
        "Salé": {"commune": "Salé",
                 "names": ["Tabriquet", "Hssaine", "Layayda", "Bab Lamrissa", "Bettana"]},
    }

    def _add_city_row(city_label, commune_name, rows_df, method_label):
        id_match = communes[communes["commune"] == commune_name]
        if len(id_match) != 1:
            print(f"[WARN] {commune_name} : {len(id_match)} correspondance(s) dans le referentiel "
                  f"(attendu 1) -> fix saute")
            return None
        cid = id_match.iloc[0]["id"]
        if (matched["commune_id"] == cid).any():
            print(f"[INFO] {commune_name} a deja une population -> fix saute")
            return None
        s = {c: rows_df[c].sum() for c in pop_cols}
        print(f"[INFO] {commune_name} ajoutee : population = somme de {len(rows_df)} arrondissements "
              f"({method_label}) = {s['pop_total']:.0f} hab.")
        return pd.DataFrame([{
            "commune_id": int(cid), "commune": commune_name,
            "province": id_match.iloc[0]["province"], "region": id_match.iloc[0]["region"],
            "match_method": "aggregated_arrondissements", "match_score": 1.0,
            "mun_raw": f"somme de {len(rows_df)} arrondissements ({method_label})",
            **s,
        }])

    if "Casablanca" in pop["city"].values:
        new_row = _add_city_row("Casablanca", "Casablanca", pop[pop["city"] == "Casablanca"],
                                 "colonne city=Casablanca")
        if new_row is not None:
            matched = pd.concat([matched, new_row], ignore_index=True)

    for _, grp in CITY_NAME_GROUPS.items():
        rows_df = pop[pop["mun"].isin(grp["names"])]
        if len(rows_df) != len(grp["names"]):
            found = rows_df["mun"].tolist()
            print(f"[WARN] {grp['commune']} : {len(grp['names']) - len(rows_df)} arrondissement(s) "
                  f"introuvable(s) dans la source (attendu {grp['names']}, trouve {found}) -> fix saute")
            continue
        new_row = _add_city_row(grp["commune"], grp["commune"], rows_df,
                                 f"noms verifies, ecart <10% vs RGPH 2014: {', '.join(grp['names'])}")
        if new_row is not None:
            matched = pd.concat([matched, new_row], ignore_index=True)

    print("[INFO] Rabat/Fes : liste d'arrondissements trouvee mais somme incoherente (Rabat -13%) ou "
          "noms ambigus (Fes: 'Medina' collision possible avec d'autres villes) -> restent SANS "
          "population, documente comme limite connue plutot que fabrique")

    matched.to_csv(config.MUN_POP_CSV, index=False, encoding="utf-8")
    print(f"[OK] {config.MUN_POP_CSV}  ({len(matched)}/{len(out)} lignes)")

    unmatched_path = config.PROCESSED / "mun_pop_unmatched.csv"
    config.ensure_dirs()
    out[out["match_method"] == "unmatched"][["mun_raw"] + pop_cols].to_csv(unmatched_path, index=False, encoding="utf-8")
    print(f"[OK] non reconciliees (pour revue) : {unmatched_path}")


if __name__ == "__main__":
    main()
