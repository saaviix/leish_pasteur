"""
underreporting.py
==================
Estimation de la sous-declaration des cas LCT, en deux volets honnetes plutot
qu'un chiffre unique invente :

1. CAPTURE-RECAPTURE A 2 SOURCES (Chapman) entre l'evidence epidemiologique
   (y_epi : cas rapportes) et l'evidence entomologique (y_ento : capture du
   vecteur) au niveau province. C'est la methode classique pour estimer un
   total "vrai" au-dela de ce qui est detecte -- mais elle suppose les deux
   sources INDEPENDANTES. Ce script verifie explicitement cette hypothese
   (elle est violee ici, voir sortie) plutot que de l'appliquer aveuglement.

2. A defaut (le point 1 s'avere structurellement inapplicable), utilise le
   modele bayesien d'occupation deja calibre (bayesian_occupancy.py, BYM2) :
   les provinces "gap" (aucune evidence, ni epi ni ento) avec psi_mean > 0.5
   sont des provinces ou le modele infere une transmission probable malgre
   zero cas rapporte. Le modele GBM deja entraine (gbm_spatial_temporal.py)
   sert ensuite a chiffrer un ordre de grandeur du volume de cas que ces
   provinces auraient si elles se comportaient comme des provinces
   climatiquement similaires deja confirmees -- un vrai calcul base sur les
   modeles existants, pas un multiplicateur invente.

Ce que ce script NE PRETEND PAS faire : estimer combien de cas INDIVIDUELS
sont manques DANS les provinces deja declarantes (ca demanderait des donnees
de terrain -- enquete de detection active sur un echantillon sentinelle, ou
liaison individuelle entre deux systemes de surveillance independants -- que
ce projet n'a pas). C'est documente explicitement en sortie plutot que
comble par une hypothese non verifiable.

Entrees :
  outputs/processed/province_table.csv
  outputs/posterior/occupancy_trace.nc
  outputs/posterior/psergenti_posterior_presence.csv
  outputs/processed/commune_panel.csv
  outputs/processed/gbm_model.joblib

Sortie :
  outputs/processed/underreporting_provinces.csv

Usage :
  python src/analysis/underreporting.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402
import model_io  # noqa: E402

PSI_THRESHOLD = 0.5


def chapman_capture_recapture(prov: pd.DataFrame) -> dict:
    prov = prov.copy()
    prov["y_ento"] = ((prov["y_ento_hard"] == 1) | (prov["y_ento_soft"] == 1)).astype(int)

    tab = pd.crosstab(prov["y_epi"], prov["y_ento"])
    n1 = int((prov["y_epi"] == 1).sum())        # detecte par surveillance epi
    n2 = int((prov["y_ento"] == 1).sum())       # detecte par etudes entomologiques
    m = int(((prov["y_epi"] == 1) & (prov["y_ento"] == 1)).sum())  # detecte par les deux

    # Chapman (1951), version corrigee du biais du Lincoln-Petersen classique
    n_hat = ((n1 + 1) * (n2 + 1) / (m + 1)) - 1
    var = ((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)) / (((m + 1) ** 2) * (m + 2))
    se = np.sqrt(max(var, 0.0))

    nested = (n2 == m)  # source 2 entierement incluse dans source 1 -> pas d'info independante

    return {
        "table": tab, "n1_epi": n1, "n2_ento": n2, "m_both": m,
        "N_hat_chapman": n_hat, "se": se, "sources_nested": nested,
    }


def gap_provinces_from_bayes() -> pd.DataFrame:
    path = config.POSTERIOR / "psergenti_posterior_presence.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable.\nLance d'abord : python src/models/bayesian_occupancy.py")
    post = pd.read_csv(path)
    return post[(post["evidence_type"] == "no_data_gap") & (post["psi_mean"] > PSI_THRESHOLD)].copy()


def estimate_hidden_burden(gap_provinces: pd.DataFrame) -> pd.DataFrame:
    """Pour les provinces gap probablement positives, utilise le GBM deja
    entraine pour chiffrer le volume de cas qu'elles auraient si elles
    suivaient le meme rapport climat/vecteur -> cas que les provinces
    confirmees comparables (extrapolation du modele, pas une observation)."""
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists() or not gap_provinces.__len__():
        return pd.DataFrame()

    saved = model_io.load_gbm(config.PROCESSED)
    panel = pd.read_csv(panel_path)

    recent = panel[(panel["annee"] == 2020) & panel["province"].isin(gap_provinces["province"])].copy()
    if not len(recent):
        return pd.DataFrame()

    # memes features que gbm_spatial_temporal.py -- lags/rolling absents pour
    # ces provinces (zero historique de cas) -> mis a 0, coherent avec un
    # "aucun antecedent de cas connu" plutot qu'une fuite d'information.
    for c in saved.get("raw_feature_cols", []):
        if c not in recent.columns and c != "province":
            recent[c] = 0.0

    recent["y_pred_cas_estime"] = model_io.predict_gbm_saved(saved, recent)

    by_prov = (
        recent.groupby("province")
        .agg(cas_estimes_2020=("y_pred_cas_estime", "sum"), n_communes=("commune_id", "nunique"))
        .reset_index()
    )
    return by_prov


def main() -> None:
    config.ensure_dirs()
    prov = pd.read_csv(config.PROVINCE_TABLE)

    print("=" * 90)
    print("1. CAPTURE-RECAPTURE A 2 SOURCES (Chapman) -- epi (cas rapportes) x ento (vecteur capture)")
    print("=" * 90)
    cr = chapman_capture_recapture(prov)
    print(cr["table"])
    print(f"\nn1 (detecte par epi)          = {cr['n1_epi']}")
    print(f"n2 (detecte par ento)          = {cr['n2_ento']}")
    print(f"m  (detecte par les deux)      = {cr['m_both']}")
    print(f"N_hat (Chapman)                = {cr['N_hat_chapman']:.1f} +/- {cr['se']:.1f}")

    if cr["sources_nested"]:
        print("\n[RESULTAT IMPORTANT] n2 == m : toutes les provinces avec evidence")
        print("entomologique ont AUSSI des cas rapportes -- l'etude du vecteur n'a")
        print("jamais ete menee dans une province sans cas deja connus. Les deux")
        print("sources ne sont PAS independantes (hypothese de Chapman violee) :")
        print("N_hat degenere et redonne simplement n1. La capture-recapture classique")
        print("ne peut donc RIEN dire ici sur les provinces totalement non-detectees --")
        print("c'est exactement pour ca que le modele bayesien spatial (BYM2) est")
        print("necessaire (partie 2 ci-dessous), pas un raccourci.")

    print(f"\n{'='*90}")
    print(f"2. PROVINCES 'GAP' PROBABLEMENT POSITIVES (psi_mean > {PSI_THRESHOLD}, modele bayesien BYM2)")
    print(f"{'='*90}")
    gap = gap_provinces_from_bayes()
    print(f"\n{len(gap)} provinces sans aucune evidence (ni epi ni ento) mais avec proba de "
          f"presence inferee > {PSI_THRESHOLD} :")
    print(gap[["province", "region", "psi_mean", "psi_sd"]].to_string(index=False))

    burden = estimate_hidden_burden(gap)
    if len(burden):
        print(f"\nEstimation du volume de cas 2020 (modele GBM deja entraine, extrapole "
              f"aux communes de ces provinces) :")
        print(burden.to_string(index=False))
        total = burden["cas_estimes_2020"].sum()
        print(f"\nTotal estime (ordre de grandeur, PAS une observation) : {total:.0f} cas/an "
              f"non captes par la surveillance passive dans ces {len(burden)} provinces")
        print("Probablement un plancher : les features cases_lag*/roll* sont forcees a 0 (aucun")
        print("historique connu), or ce sont les variables les plus importantes du GBM -> le")
        print("modele est structurellement conservateur ici, le vrai volume est possiblement plus haut.")
    else:
        print("\n[INFO] estimation GBM non disponible (modele/panel introuvable)")

    print(f"\n{'='*90}")
    print("3. DETECTION COMPLETENESS DE LA SURVEILLANCE EPI (parametre p_epi du modele bayesien)")
    print(f"{'='*90}")
    trace_path = config.POSTERIOR / "occupancy_trace.nc"
    if trace_path.exists():
        import arviz as az
        idata = az.from_netcdf(str(trace_path))
        s = az.summary(idata, var_names=["p_epi", "p_soft"])
        print(s)
        p_epi_mean = float(s.loc["p_epi", "mean"])
        print(f"\np_epi = P(la surveillance epi rapporte >=1 cas | le vecteur est reellement present) = {p_epi_mean:.3f}")
        print("Attention : c'est une proba de DETECTION DU SITE (la province a-t-elle au moins")
        print("un cas rapporte), PAS une fraction de cas individuels captes. Ne pas l'utiliser")
        print("comme multiplicateur direct sur un nombre de cas.")
    else:
        print(f"[INFO] {trace_path} introuvable, section sautee")

    print(f"\n{'='*90}")
    print("4. CE QUI N'EST PAS ESTIME ICI (limite honnete)")
    print(f"{'='*90}")
    print("La sous-declaration INDIVIDUELLE dans les provinces qui rapportent DEJA des cas")
    print("(combien de cas reels supplementaires par rapport aux cas rapportes) ne peut pas")
    print("etre estimee de facon rigoureuse avec les donnees actuelles -- il faudrait soit :")
    print("  - une enquete de detection active sur un echantillon sentinelle de communes,")
    print("  - ou deux systemes de surveillance independants avec liaison au niveau individuel")
    print("    (capture-recapture classique appliquee aux CAS, pas aux provinces).")
    print("Aucun multiplicateur de litterature n'est applique ici sans donnee pour le justifier.")

    out_path = config.PROCESSED / "underreporting_provinces.csv"
    gap_out = gap.merge(burden, on="province", how="left") if len(burden) else gap
    gap_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
