"""
clean_lct.py
============
Nettoyage et standardisation des donnees de cas de leishmaniose cutanee (LCT).

Objectifs :
  1. Charger data/raw/leish_LCT.csv (cas 2009-2020).
  2. Normaliser les libelles texte (region, province, commune) : trim, casse,
     harmonisation des accents/tirets (ex. "DrâaTafilalet" vs "Drâa-Tafilalet",
     "OURIKA" vs "Ourika").
  3. Reconcilier region/province/COMMUNE avec le referentiel unique
     communes_maroc_final.csv (source de verite geographique) : correspondance
     exacte puis floue (geo_matching.py), restreinte a la province deja
     identifiee. Avant la Phase 2 de la refonte, seules region/province
     etaient reconciliees -- la commune ne l'etait pas du tout, ce qui
     laissait 58% des lignes (15 080/26 016) sans correspondance commune
     silencieusement invisible dans les sorties.
  4. Dedoublonner les lignes strictement identiques (non detecte auparavant).
  5. Produire un RAPPORT de completude (donnees manquantes) : c'est la base de
     l'inference bayesienne pour les communes/provinces sans donnee.

Sorties :
  - outputs/processed/lct_clean.csv            (cas nettoyes)
  - outputs/processed/lct_missing_report.csv   (rapport de donnees manquantes)

Usage :
  python src/data_prep/clean_lct.py
"""

import difflib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import config
import geo_matching

norm_key = config.norm_key

ROOT = config.ROOT
IN_LCT = config.LCT_CSV
IN_COMMUNES = config.COMMUNES_CSV
OUT_DIR = config.PROCESSED
OUT_CLEAN = OUT_DIR / "lct_clean.csv"
OUT_MISSING = OUT_DIR / "lct_missing_report.csv"

FUZZY_CUTOFF = 0.82
# seuil plus strict pour le repli inter-province (stage 2) : sur un espace de
# recherche non contraint par la province, les prefixes tres frequents dans
# la toponymie marocaine ("Ait", "Oulad", "Sidi", "Ain", "Bni"...) produisent
# des faux positifs a score modere (ex. "Oulad Tahar" ~ "Oulad Amghar" a 0.87,
# "Skoura" ~ "Meskoura" a 0.857) -- verifie empiriquement sur ce jeu de donnees.
FUZZY_CUTOFF_CROSS_PROVINCE = 0.90


def load_and_standardize(lct_path: Path, communes_path: Path):
    lct = pd.read_csv(lct_path)
    communes = pd.read_csv(communes_path)

    n_raw = len(lct)
    lct = lct.drop_duplicates(keep="first").reset_index(drop=True)
    n_dedup = n_raw - len(lct)

    # --- 2 dernieres lignes Commune/Province vides (session 2026-08-12) :
    # Province/Commune absentes de la source, mais Localite contenait un
    # indice exploitable par l'utilisateur (pas par un matching textuel --
    # niveau quartier, hors du referentiel commune). Decisions explicites :
    # "Hay Rmail LOUED" identifiee comme Sidi Kacem (rattachee ici, resolue
    # ensuite normalement par le matching standard) ; "Pave lamir N 5" non
    # identifiable -> ligne supprimee plutot que laissee non reconciliee.
    hay_rmail = lct["Localite"] == "Hay Rmail LOUED"
    if hay_rmail.any():
        lct.loc[hay_rmail, "Commune"] = "Sidi Kacem"
        lct.loc[hay_rmail, "Province"] = "Sidi Kacem"
    pave_lamir = lct["Localite"] == "Pave lamir N° 5"
    if pave_lamir.any():
        lct = lct[~pave_lamir].reset_index(drop=True)

    # --- recuperation du mois depuis Date_Diagnostic (2016/2018/2019) ---
    # Mois_Diagnostic = NaN a 100% pour 2016/2018/2019, mais Date_Diagnostic
    # contient en realite le mois pour ces memes lignes -- le champ a ete mal
    # cartographie a la source (rapports annuels au format different pour ces
    # 3 annees), pas reellement absent. Format melange selon l'annee/la ligne :
    # numero brut (2016 : 100%), date ISO 'YYYY-MM-DD...' ou 'JJ/MM/AAAA'
    # (2018/2019, majorite), ou nom de mois francais ('Fevrier', 'JANVIER',
    # parfois avec accent corrompu en mojibake 'F�vrier') (2018/2019, reste).
    # Les 3 formes sont extraites plutot que de n'en recuperer qu'une seule
    # (une premiere version de ce fix ne gerait que le numero brut et ratait
    # ~40% des cas recuperables de 2018/2019).
    MONTHS_FR = {
        "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
        "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
    }

    def parse_month(raw) -> float:
        if pd.isna(raw):
            return np.nan
        s = str(raw).strip()
        n = pd.to_numeric(s, errors="coerce")
        if pd.notna(n) and 1 <= n <= 12:
            return float(n)
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return float(dt.month)
        key = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()
        key = "".join(ch for ch in key if ch.isalpha())
        if key in MONTHS_FR:
            return float(MONTHS_FR[key])
        # mojibake (accent remplace par un caractere illisible) -> le
        # caractere est simplement absent apres le strip ascii ci-dessus
        # (ex. "F�vrier" -> "fvrier") ; correspondance approchee tolerante
        # a 1-2 caracteres d'ecart plutot qu'une liste de variantes figee.
        close = difflib.get_close_matches(key, MONTHS_FR.keys(), n=1, cutoff=0.75)
        return float(MONTHS_FR[close[0]]) if close else np.nan

    date_as_month = lct["Date_Diagnostic"].apply(parse_month)
    recoverable = lct["Mois_Diagnostic"].isna() & date_as_month.notna()
    n_recovered = int(recoverable.sum())
    if n_recovered:
        lct.loc[recoverable, "Mois_Diagnostic"] = date_as_month[recoverable]
        # Date_Diagnostic pour ces lignes ne represente pas une vraie date
        # complete (ou a ete consommee comme mois) -- vide pour ne pas induire
        # en erreur un futur usage de la colonne.
        lct.loc[recoverable, "Date_Diagnostic"] = np.nan
        lct["mois_recupere_de_date_diagnostic"] = recoverable
        years_fixed = sorted(lct.loc[recoverable, "Annee_Source"].unique().tolist())
        print(f"[INFO] mois recupere depuis Date_Diagnostic (mauvaise colonne/format a la source) pour "
              f"{n_recovered} lignes, annees {years_fixed}")
    else:
        lct["mois_recupere_de_date_diagnostic"] = False

    # --- referentiel geographique (province -> region canonique) ---
    communes_idx = geo_matching.build_commune_index(communes)
    ref = (
        communes_idx.groupby("prov_key")
        .agg(province_ref=("province", "first"), region_ref=("region", "first"))
        .reset_index()
    )
    by_province = geo_matching.candidates_by_province(communes_idx)
    manual_overrides = geo_matching.resolve_manual_overrides(communes_idx)
    province_capital_fallback = geo_matching.resolve_province_capital_fallback(communes_idx)

    # --- standardisation des colonnes LCT ---
    lct["Region"] = lct["Region"].map(config.canonical_region)
    lct["prov_key"] = lct["Province"].map(norm_key)
    for col in ["Commune", "Secteur", "Localite"]:
        if col in lct.columns:
            lct[col] = lct[col].astype(str).str.strip()
            mask_upper = lct[col].str.isupper() & (lct[col].str.len() > 1)
            lct.loc[mask_upper, col] = lct.loc[mask_upper, col].str.title()

    # --- reconciliation region/province via le referentiel ---
    lct = lct.merge(ref, on="prov_key", how="left")
    lct["Region_std"] = lct["region_ref"].fillna(lct["Region"])
    lct["Province_std"] = lct["province_ref"].fillna(lct["Province"])
    lct["province_matched"] = lct["region_ref"].notna()

    # --- reconciliation commune, stage 1 : exacte puis floue, restreinte a la province declaree ---
    match_results = [
        geo_matching.match_commune_in_province(row.Commune, row.prov_key, by_province, cutoff=FUZZY_CUTOFF,
                                                manual_overrides=manual_overrides,
                                                province_capital_fallback=province_capital_fallback)
        if row.province_matched else
        {"commune_std": None, "commune_id": None, "match_method": "province_unmatched", "match_score": 0.0}
        for row in lct[["Commune", "prov_key", "province_matched"]].itertuples()
    ]
    match_df = pd.DataFrame(match_results)

    # --- stage 2 : repli inter-province pour ce qui reste non reconcilie,
    # UNIQUEMENT si le nom de commune est unique au niveau national (evite
    # les faux positifs sur des noms generiques presents dans >1 province,
    # ex. "Ras El Ma" existe dans 10 provinces differentes) ---
    uniqueness = geo_matching.build_global_uniqueness(communes_idx)
    match_df["province_conflict"] = False
    still_unmatched = match_df["match_method"] == "unmatched"
    if still_unmatched.any():
        fallback = lct.loc[still_unmatched, "Commune"].apply(
            lambda name: geo_matching.match_commune_cross_province(
                name, communes_idx, uniqueness, cutoff=FUZZY_CUTOFF_CROSS_PROVINCE
            )
        )
        fallback_df = pd.DataFrame(fallback.tolist(), index=fallback.index)
        for col in ["commune_std", "commune_id", "match_method", "match_score"]:
            match_df.loc[still_unmatched, col] = fallback_df[col].values
        # tout succes en stage 2 est par construction dans une province differente
        # de celle declaree (stage 1 avait deja echoue dans la province declaree)
        cross_ok = fallback_df["match_method"].isin(["exact_cross_province", "fuzzy_cross_province"])
        match_df.loc[still_unmatched, "province_conflict"] = cross_ok.values

    # --- stage 3 : repli chef-lieu SANS liste blanche de noms, decision
    # explicite de l'utilisateur (session 2026-08-12) pour clore l'ecart
    # restant apres 4 vagues de correction manuelle (~1400 cas). Precision
    # individuelle sacrifiee pour couverture -- trace separement
    # ("province_capital_catchall", score 0.3) de la version verifiee nom par
    # nom ("province_capital_fallback", score 0.5) pour rester auditable. Ne
    # touche jamais les lignes "province_unmatched" (aucune province connue,
    # rien sur quoi se rabattre) ni les rares "province_conflict" du stage 2
    # (deja resolues dans une autre province, laissees telles quelles).
    still_unmatched_3 = match_df["match_method"] == "unmatched"
    if still_unmatched_3.any():
        catchall = lct.loc[still_unmatched_3, "prov_key"].apply(
            lambda pk: geo_matching.catchall_province_capital(pk, by_province)
        )
        for idx in catchall.index:
            res = catchall.loc[idx]
            if res is not None:
                for col in ["commune_std", "commune_id", "match_method", "match_score"]:
                    match_df.loc[idx, col] = res[col]

    lct["Commune_std"] = match_df["commune_std"]
    lct["commune_id"] = match_df["commune_id"]
    lct["commune_match_method"] = match_df["match_method"]
    lct["commune_match_score"] = match_df["match_score"]
    lct["province_conflict"] = match_df["province_conflict"]
    lct["commune_matched"] = lct["commune_match_method"].isin(
        ["manual_override", "exact", "fuzzy", "prefix_in_province", "abbrev_in_province",
         "restored_prefix_in_province", "province_capital_fallback", "province_capital_catchall",
         "exact_cross_province", "fuzzy_cross_province",
         "prefix_cross_province", "abbrev_cross_province"]
    )

    n_manual = int((lct["commune_match_method"] == "manual_override").sum())
    n_exact = int((lct["commune_match_method"] == "exact").sum())
    n_fuzzy = int((lct["commune_match_method"] == "fuzzy").sum())
    n_prefix = int((lct["commune_match_method"] == "prefix_in_province").sum())
    n_abbrev = int((lct["commune_match_method"] == "abbrev_in_province").sum())
    n_cross = int(lct["commune_match_method"].isin(
        ["exact_cross_province", "fuzzy_cross_province",
         "prefix_cross_province", "abbrev_cross_province"]
    ).sum())
    n_capital = int(lct["commune_match_method"].eq("province_capital_fallback").sum())
    n_catchall = int(lct["commune_match_method"].eq("province_capital_catchall").sum())
    n_unmatched = int((~lct["commune_matched"]).sum())
    n_conflict = int(lct["province_conflict"].sum())
    print(f"[INFO] deduplication : {n_dedup} lignes strictement dupliquees supprimees ({n_raw} -> {len(lct)})")
    print(f"[INFO] reconciliation commune : {n_manual} corrections manuelles verifiees, {n_exact} exactes "
          f"(province declaree), {n_fuzzy} floues (province declaree, seuil={FUZZY_CUTOFF}), "
          f"{n_prefix} par prefixe de mot unique (ex. 'Boumalne' -> 'Boumalne Dades'), "
          f"{n_abbrev} par expansion d'abreviation (ex. 'Z.Cheikh' -> 'Zaouiat Cheikh'), "
          f"{n_cross} repli inter-province (nom unique au national, "
          f"dont {n_conflict} avec province declaree differente -- a documenter), "
          f"{n_capital} chef-lieu verifie nom par nom, {n_catchall} chef-lieu sans verification "
          f"individuelle (stage 3, decision explicite utilisateur), "
          f"{n_unmatched} non reconciliees sur {len(lct)} lignes "
          f"({100 * lct['commune_matched'].sum() / len(lct):.1f}% de taux de jointure commune)")

    return lct, communes


def completeness_report(lct: pd.DataFrame, communes: pd.DataFrame) -> pd.DataFrame:
    """Rapport : pour chaque province du referentiel, a-t-on des cas ? quelles annees ?"""
    communes["prov_key"] = communes["province"].map(norm_key)
    ref = (
        communes.groupby("prov_key")
        .agg(
            province=("province", "first"),
            region=("region", "first"),
            n_communes=("commune", "count"),
        )
        .reset_index()
    )

    cases = (
        lct.groupby("prov_key")
        .agg(
            n_cases=("Province", "size"),
            n_cases_commune_matched=("commune_matched", "sum"),
            years=("Annee_Source", lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))),
        )
        .reset_index()
    )
    rep = ref.merge(cases, on="prov_key", how="left")
    rep["n_cases"] = rep["n_cases"].fillna(0).astype(int)
    rep["n_cases_commune_matched"] = rep["n_cases_commune_matched"].fillna(0).astype(int)
    rep["has_data"] = rep["n_cases"] > 0
    rep["years"] = rep["years"].apply(lambda v: v if isinstance(v, list) else [])
    rep["n_years_covered"] = rep["years"].apply(len)
    rep["status"] = rep["has_data"].map({True: "avec_donnees", False: "MANQUANT_gap"})
    rep = rep.sort_values(["has_data", "region", "province"]).reset_index(drop=True)
    return rep[["region", "province", "n_communes", "n_cases", "n_cases_commune_matched",
                "n_years_covered", "years", "status"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IN_LCT.exists():
        raise FileNotFoundError(f"Introuvable : {IN_LCT}")
    if not IN_COMMUNES.exists():
        raise FileNotFoundError(f"Introuvable : {IN_COMMUNES}")

    lct, communes = load_and_standardize(IN_LCT, IN_COMMUNES)
    lct.to_csv(OUT_CLEAN, index=False, encoding="utf-8")

    rep = completeness_report(lct, communes)
    rep.to_csv(OUT_MISSING, index=False, encoding="utf-8")

    # resume console
    n_prov_total = len(rep)
    if "has_data" in rep.columns:
        n_prov_gap = int((~rep["has_data"]).sum())
    else:
        n_prov_gap = int(rep["status"].eq("MANQUANT_gap").sum())
    n_unmatched = int((~lct["province_matched"]).sum())
    n_commune_matched = int(lct["commune_matched"].sum())
    print("=" * 60)
    print("NETTOYAGE LCT termine")
    print("=" * 60)
    print(f"Lignes de cas          : {len(lct)}")
    print(f"Provinces (referentiel): {n_prov_total}")
    print(f"Provinces SANS donnee  : {n_prov_gap}  (-> a inferer par le modele bayesien)")
    print(f"Lignes province non reconnues au referentiel : {n_unmatched}")
    print(f"Lignes commune reconciliee au referentiel    : {n_commune_matched}/{len(lct)} "
          f"({100 * n_commune_matched / len(lct):.1f}%)")
    print("-" * 60)
    print(f"Ecrit : {OUT_CLEAN}")
    print(f"Ecrit : {OUT_MISSING}")


if __name__ == "__main__":
    main()
