"""
forecast_future.py
==================
Genere les projections epidemiologiques de LCT a 20 ans (2025-2045), par
commune / province / region, a partir du VRAI modele GBM entraine
(gbm_spatial_temporal.py) -- pas d'une formule heuristique.

Corrige en Phase 3 de la refonte :
  1. La version precedente appelait `generate_future_forecasts(gbm_model=None)`
     sans jamais recevoir de modele (run_pipeline.py ne le passait pas), et
     retombait donc TOUJOURS sur une formule codee en dur presentee comme une
     prediction de modele. Le modele entraine est maintenant charge depuis
     outputs/processed/gbm_model.joblib (persiste par gbm_spatial_temporal.py) ;
     si ce fichier n'existe pas, le script s'arrete avec une erreur explicite
     plutot que de fabriquer des chiffres.
  2. La grille climatologique (moyenne historique par commune x mois) est
     desormais construite par climatology.py, vectorise (un merge) au lieu
     d'une boucle Python imbriquee commune x annee x mois (~10 min pour 21
     ans) ET indexee sur `commune_id` au lieu du nom de commune (l'ancienne
     version perdait 6 communes homonymes via `drop_duplicates(["commune"])`).

Limite assumee et documentee (pas cachee) : les covariables autoregressives
(cases_lag*/cases_roll*) ne peuvent pas etre calculees pour une projection
pure a 20 ans -- aucun historique de cas futur n'existe. Elles sont mises a 0
(= "pas de recrudescence recente connue"), ce qui sous-estime probablement
les zones a foyers recurrents. Un forecast recursif pas-a-pas serait plus
correct mais hors perimetre ici.

Sorties :
  - outputs/processed/forecast_2025_2045_communes.csv
  - outputs/processed/forecast_2025_2045_provinces.csv
  - outputs/processed/forecast_2025_2045_regions.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config
from climatology import build_climatology_grid
from model_io import load_gbm, predict_gbm_saved

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_future_forecasts():
    panel_path = config.PROCESSED / "commune_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} introuvable. Lance build_commune_panel.py d'abord.")

    gbm_saved = load_gbm(config.PROCESSED)
    panel = pd.read_csv(panel_path)

    future_years = list(range(2025, 2046))
    model_label = gbm_saved.get("model_name") or type(gbm_saved.get("model", gbm_saved.get("model_1"))).__name__
    logger.info(f"Génération de la grille climatologique 2025-2045 ({len(future_years)} ans, "
                f"modèle : {model_label})...")
    future_df, n_no_clim = build_climatology_grid(panel, future_years)
    if n_no_clim:
        logger.warning(f"{n_no_clim} lignes sans historique climatique pour le mois de la commune "
                        f"-> covariables climat comblees a 0 (prediction moins fiable pour ces lignes)")

    for col in gbm_saved.get("raw_feature_cols") or gbm_saved.get("feature_cols", []):
        if col not in future_df.columns:
            future_df[col] = 0.0  # cases_lag*/cases_roll* : pas d'historique de cas futur, cf. docstring

    future_df["cas_predits"] = predict_gbm_saved(gbm_saved, future_df)

    # Intervalle a 95% par ligne (approximation Poisson, variance ~ moyenne :
    # meme convention que gbm_pinn_stacked.py/model_io.py, cf. section
    # "Evaluation metrics" du rapport). "_var" est la variance implicite de
    # CETTE ligne -- conservee jusqu'a l'agregation ci-dessous.
    future_df["_var"] = future_df["cas_predits"] + 1e-6
    future_df["ci_lower_95"] = np.clip(future_df["cas_predits"] - 1.96 * np.sqrt(future_df["_var"]), 0, None)
    future_df["ci_upper_95"] = future_df["cas_predits"] + 1.96 * np.sqrt(future_df["_var"])

    comm_out = config.PROCESSED / "forecast_2025_2045_communes.csv"
    future_df[["commune", "province", "region", "annee", "mois", "cas_predits", "ci_lower_95", "ci_upper_95"]].to_csv(comm_out, index=False)
    logger.info(f"Projections par commune exportées : {comm_out}")

    def aggregate_with_ci(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
        """Agrege cas_predits et son IC95% correctement : les variances
        (approximativement independantes ligne a ligne) s'additionnent, PAS
        les demi-largeurs d'intervalle -- sommer les bornes ci_lower_95/
        ci_upper_95 directement (comme le faisait la version precedente)
        surestime radicalement l'incertitude a mesure qu'on agrege des
        centaines de lignes commune x mois (largeur ~ N au lieu de sqrt(N))."""
        agg = df.groupby(group_cols).agg(cas_predits=("cas_predits", "sum"), _var=("_var", "sum")).reset_index()
        agg["ci_lower_95"] = np.clip(agg["cas_predits"] - 1.96 * np.sqrt(agg["_var"]), 0, None)
        agg["ci_upper_95"] = agg["cas_predits"] + 1.96 * np.sqrt(agg["_var"])
        return agg.drop(columns="_var")

    prov_df = aggregate_with_ci(future_df, ["province", "region", "annee", "mois"])
    prov_out = config.PROCESSED / "forecast_2025_2045_provinces.csv"
    prov_df.to_csv(prov_out, index=False)
    logger.info(f"Projections par province exportées : {prov_out}")

    reg_df = aggregate_with_ci(future_df, ["region", "annee", "mois"])
    reg_out = config.PROCESSED / "forecast_2025_2045_regions.csv"
    reg_df.to_csv(reg_out, index=False)
    logger.info(f"Projections par région exportées : {reg_out}")


if __name__ == "__main__":
    generate_future_forecasts()
