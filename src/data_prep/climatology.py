"""
climatology.py
===============
Construit une grille commune x annee x mois pour des annees SANS climat ERA5
mesure (projections futures, ou 2023/2024 pour la verification -- les
fichiers ERA5 telecharges s'arretent en 2021), a partir de la climatologie
historique par commune (moyenne par mois sur les annees disponibles) plus une
tendance de rechauffement lineaire simple sur la temperature.

Factorise depuis forecast_future.py (Phase 3 de la refonte) pour etre reutilise
par robust_ensemble_recalibrated.py (verification 2023/2024) sans dupliquer
la logique -- et corrige au passage un bug de forecast_future.py : la grille
etait construite par boucle Python imbriquee (communes x annees x mois, avec
un filtre pandas a CHAQUE iteration), ~10 minutes pour 21 ans. Version
vectorisee ici : un merge, pas de boucle.

Limite assumee : ceci est une climatologie (moyenne saisonniere + tendance),
PAS une mesure. A n'utiliser que pour des horizons courts (quelques annees)
ou des projections explicitement hors donnees mesurees -- jamais a la place
de climat reel quand celui-ci existe.
"""

import numpy as np
import pandas as pd


def build_climatology_grid(
    panel: pd.DataFrame,
    target_years: list,
    warming_per_year: float = 0.03,
    static_cols=("commune", "commune_id", "province", "region", "latitude", "longitude",
                 "elevation_m", "lai", "aridity_index", "pop_total",
                 "psi_mean", "psi_sd", "y_epi", "y_ento_hard", "y_ento_soft"),
) -> pd.DataFrame:
    """Grille commune x annee x mois pour `target_years`, climat = moyenne
    historique par (commune_id, mois) + rechauffement lineaire sur la
    temperature depuis la derniere annee mesuree."""
    last_hist_year = int(panel["annee"].max())

    keep_static = [c for c in static_cols if c in panel.columns]
    communes = panel[keep_static].drop_duplicates(subset=["commune_id"])

    clim_hist = (
        panel[panel["annee"] <= last_hist_year]
        .groupby(["commune_id", "mois"])[["temp_moy", "precip_mm", "humidite_pct"]]
        .mean()
        .reset_index()
    )

    grid = communes.merge(pd.DataFrame({"annee": target_years}), how="cross").merge(
        pd.DataFrame({"mois": range(1, 13)}), how="cross"
    )
    grid = grid.merge(clim_hist, on=["commune_id", "mois"], how="left")

    grid["temp_moy"] = grid["temp_moy"] + (grid["annee"] - last_hist_year) * warming_per_year
    grid["sin_month"] = np.sin(2 * np.pi * grid["mois"] / 12.0)
    grid["cos_month"] = np.cos(2 * np.pi * grid["mois"] / 12.0)

    # Vraies valeurs decalees (mois-lag, climatologie cyclique sur 12 mois) --
    # PAS la valeur du mois courant repetee (bug trouve en audit : detruisait
    # tout signal de transition saisonniere pour les lignes futures/climatologie,
    # alors que le modele est entraine sur de vrais lags variables 2009-2020).
    for lag in range(1, 7):
        lag_grid = grid[["commune_id", "annee", "mois"]].copy()
        lag_grid["mois_lag"] = ((lag_grid["mois"] - lag - 1) % 12) + 1
        lag_grid["annee_lag"] = lag_grid["annee"] - ((lag_grid["mois"] - lag - 1) // 12 != (lag_grid["mois"] - 1) // 12).astype(int)
        lag_clim = clim_hist.rename(columns={
            "mois": "mois_lag", "temp_moy": f"temp_moy_lag{lag}",
            "precip_mm": f"precip_mm_lag{lag}", "humidite_pct": f"humidite_pct_lag{lag}",
        })
        lag_grid = lag_grid.merge(lag_clim, on=["commune_id", "mois_lag"], how="left")
        lag_grid[f"temp_moy_lag{lag}"] += (lag_grid["annee_lag"] - last_hist_year) * warming_per_year
        grid[f"temp_moy_lag{lag}"] = lag_grid[f"temp_moy_lag{lag}"].values
        grid[f"precip_mm_lag{lag}"] = lag_grid[f"precip_mm_lag{lag}"].values
        grid[f"humidite_pct_lag{lag}"] = lag_grid[f"humidite_pct_lag{lag}"].values

    n_no_clim = int(grid["precip_mm"].isna().sum())
    if n_no_clim:
        grid[["temp_moy", "precip_mm", "humidite_pct"]] = grid[["temp_moy", "precip_mm", "humidite_pct"]].fillna(0.0)

    return grid, n_no_clim
