"""
forecast_future_recursive.py
=============================
Projection future du modele officiel (GBM+PINN stacke), reconstruite pas a
pas (mois par mois, janvier 2021 a decembre 2045) plutot qu'en un seul appel
vectorise sur une grille climatologique statique.

Corrige la limite documentee explicitement dans forecast_future.py :
  "Limite assumee et documentee (pas cachee) : les covariables autoregressives
  (cases_lag*/cases_roll*) ne peuvent pas etre calculees pour une projection
  pure a 20 ans -- aucun historique de cas futur n'existe. Elles sont mises a
  0 [...] Un forecast recursif pas-a-pas serait plus correct mais hors
  perimetre ici."
Ce script implemente ce forecast recursif : a chaque mois, les covariables
cases_lag*/cases_roll*/neighbor_cases_* sont recalculees a partir des VRAIES
donnees historiques (jusqu'a decembre 2020) puis, au-dela, des PREDICTIONS
du modele lui-meme aux mois precedents.

Optimisation : contrairement a une implementation naive qui rappellerait les
fonctions de featurisation d'entrainement (add_pinn_physics_features,
add_neighbor_features) sur tout l'historique cumule a CHAQUE mois -- ce qui
recalcule inutilement les memes valeurs passees des centaines de fois et
devient de plus en plus lent a mesure que l'historique grandit (teste :
~35s/mois des le premier mois, donc plusieurs heures sur 300 mois) -- ce
script ne recalcule que ce qui est necessaire pour le MOIS COURANT a chaque
etape, via des jointures ciblees sur un historique indexe, et pousse le
mecanisme PINN "etat accumule" (E_H/I_H/C_vraie) directement plutot que de
rappeler le pipeline d'entrainement complet.

Sorties (memes noms de colonnes que forecast_future.py, remplace ses sorties) :
  - outputs/processed/forecast_2025_2045_communes.csv
  - outputs/processed/forecast_2025_2045_provinces.csv
  - outputs/processed/forecast_2025_2045_regions.csv

Usage :
  python src/analysis/forecast_future_recursive.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
sys.path.insert(0, str(ROOT / "src" / "models"))
import config  # noqa: E402
from climatology import build_climatology_grid  # noqa: E402
from model_io import load_gbm, predict_gbm_saved  # noqa: E402
import pinn_seirv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

N_NEIGHBORS = 5
LAGS = [1, 2, 3, 12, 18, 24]
ROLLS = {"cases_roll3": 3, "cases_roll6": 6, "cases_roll9": 9, "cases_roll12": 12, "cases_roll18": 18}


def load_pinn():
    path = config.PROCESSED / "pinn_seirv_weights.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = pinn_seirv.SEIRVPINN(n_provinces=len(ckpt["provinces"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["provinces"]


def build_neighbor_edges(static: pd.DataFrame, k: int = N_NEIGHBORS) -> pd.DataFrame:
    """Meme logique que gbm_pinn_stacked.add_neighbor_features, mais calculee
    UNE SEULE FOIS (les coordonnees sont statiques) plutot qu'a chaque mois."""
    edges = []
    for _, g in static.groupby("province"):
        names = g["commune"].to_numpy()
        lat = g["latitude"].to_numpy()
        lon = g["longitude"].to_numpy()
        n = len(names)
        if n < 2:
            continue
        for i in range(n):
            d = np.sqrt((lat - lat[i]) ** 2 + (lon - lon[i]) ** 2)
            d[i] = np.inf
            nn_idx = np.argsort(d)[:min(k, n - 1)]
            edges.extend((names[i], names[j]) for j in nn_idx)
    return pd.DataFrame(edges, columns=["commune", "neighbor"])


def compute_pinn_features(model, prov_to_idx: dict, df: pd.DataFrame, t_ref_year: int) -> pd.DataFrame:
    """Passe avant complete du PINN pour les lignes de df (un seul mois a la
    fois) -- necessite temp_moy, precip_mm, humidite_pct, latitude, longitude,
    province, annee, mois, cases_lag1/cases_roll3/cases_roll6 (RAW, pas
    log1p) deja presents dans df."""
    with torch.no_grad():
        temp_t = torch.tensor(df["temp_moy"].fillna(df["temp_moy"].median()).values, dtype=torch.float32).unsqueeze(1)
        precip_t = torch.tensor(df["precip_mm"].fillna(df["precip_mm"].median()).values, dtype=torch.float32).unsqueeze(1)
        humid_t = torch.tensor(df["humidite_pct"].fillna(df["humidite_pct"].median()).values, dtype=torch.float32).unsqueeze(1)
        lat_t = torch.tensor(df["latitude"].values, dtype=torch.float32).unsqueeze(1)
        lon_t = torch.tensor(df["longitude"].values, dtype=torch.float32).unsqueeze(1)

        emergence = model.emergence_fn(torch.cat([temp_t, precip_t], dim=1)).numpy().flatten()
        muV = model.mortality_fn(temp_t).numpy().flatten()
        sigV = model.eip_fn(temp_t).numpy().flatten()

        t_months = (df["annee"].values - t_ref_year) * 12 + (df["mois"].values - 1)
        t_t = torch.tensor(t_months.astype(np.float32)).unsqueeze(1)
        hist_vals = np.stack([
            np.log1p(df["cases_lag1"].fillna(0.0).values.astype(np.float32)),
            np.log1p(df["cases_roll3"].fillna(0.0).values.astype(np.float32)),
            np.log1p(df["cases_roll6"].fillna(0.0).values.astype(np.float32)),
        ], axis=1)
        hist_t = torch.tensor(hist_vals, dtype=torch.float32)
        prov_idx = torch.tensor(
            df["province"].map(prov_to_idx).fillna(0).astype(int).values, dtype=torch.long
        ).unsqueeze(1)

        out = model(t_t, lat_t, lon_t, temp_t, precip_t, humid_t, hist_t, prov_idx)
        cas_rapportes = (model.rho * out["obs_rate"]).detach().numpy().flatten()

    feats = pd.DataFrame(index=df.index)
    feats["pinn_emergence"] = emergence
    feats["pinn_mortalite_vecteur"] = muV
    feats["pinn_capacite_vectorielle"] = emergence / muV
    feats["pinn_incubation_extrinseque"] = 1.0 / sigV
    feats["pinn_E_H"] = out["E_H"].numpy().flatten()
    feats["pinn_I_H"] = out["I_H"].numpy().flatten()
    feats["pinn_C_vraie"] = out["obs_rate"].numpy().flatten()
    feats["pinn_cas_rapportes"] = cas_rapportes
    return feats


def run() -> None:
    t_start = time.time()
    panel_path = config.PROCESSED / "commune_panel.csv"
    panel = pd.read_csv(panel_path)

    gbm_saved = load_gbm(config.PROCESSED)
    pinn_model, pinn_provinces = load_pinn()
    prov_to_idx = {p: i for i, p in enumerate(pinn_provinces)}
    logger.info(f"Modeles charges : {gbm_saved.get('model_name')} + PINN ({len(pinn_provinces)} provinces)")

    static_cols = ["commune_id", "commune", "province", "region", "latitude", "longitude",
                    "elevation_m", "lai", "aridity_index", "psi_mean", "psi_sd",
                    "y_epi", "y_ento_hard", "y_ento_soft"]
    static_cols = [c for c in static_cols if c in panel.columns]
    static = panel[static_cols].drop_duplicates(subset=["commune_id"]).reset_index(drop=True)
    n_communes = len(static)

    edges = build_neighbor_edges(static)
    logger.info(f"{n_communes} communes, {len(edges)} arcs de voisinage (k={N_NEIGHBORS}, meme province)")

    t_ref_year = int(panel["annee"].min())

    # historique n_cas reel : long format (commune_id, annee, mois) -> n_cas,
    # jusqu'a decembre 2020 (dernier mois reel connu au niveau commune)
    hist = panel.loc[panel["annee"] <= 2020, ["commune_id", "annee", "mois", "n_cas"]].copy()

    # climat 2021 (mesure reel, deja dans panel) + climatologie 2022-2045
    real_2021_clim = panel.loc[panel["annee"] == 2021,
                                ["commune_id", "annee", "mois", "temp_moy", "precip_mm", "humidite_pct"]].copy()
    logger.info("Construction de la grille climatologique 2022-2045...")
    future_grid, n_no_clim = build_climatology_grid(panel, list(range(2022, 2046)))
    if n_no_clim:
        logger.warning(f"{n_no_clim} lignes climatologiques sans historique -> comblees a 0")
    future_clim = future_grid[["commune_id", "annee", "mois", "temp_moy", "precip_mm", "humidite_pct"]].copy()
    all_clim = pd.concat([real_2021_clim, future_clim], ignore_index=True)

    future_months = [(y, m) for y in range(2021, 2046) for m in range(1, 13)]

    results = []
    for i, (year, month) in enumerate(future_months):
        rows = static.copy()
        rows["annee"] = year
        rows["mois"] = month
        clim_month = all_clim[(all_clim["annee"] == year) & (all_clim["mois"] == month)][
            ["commune_id", "temp_moy", "precip_mm", "humidite_pct"]
        ]
        rows = rows.merge(clim_month, on="commune_id", how="left")
        rows["sin_month"] = np.sin(2 * np.pi * month / 12.0)
        rows["cos_month"] = np.cos(2 * np.pi * month / 12.0)

        # --- lags/rolling de cas, via jointures ciblees sur l'historique (pas
        # de recalcul de groupby/shift sur tout l'historique cumule) ---
        for lag in LAGS:
            ly, lm = year, month - lag
            while lm <= 0:
                lm += 12
                ly -= 1
            sub = hist[(hist["annee"] == ly) & (hist["mois"] == lm)][["commune_id", "n_cas"]]
            rows = rows.merge(sub.rename(columns={"n_cas": f"cases_lag{lag}"}), on="commune_id", how="left")

        for col, window in ROLLS.items():
            months_needed = []
            ly, lm = year, month
            for _ in range(window):
                lm -= 1
                if lm <= 0:
                    lm += 12
                    ly -= 1
                months_needed.append((ly, lm))
            sub = hist.merge(pd.DataFrame(months_needed, columns=["annee", "mois"]),
                              on=["annee", "mois"], how="inner")
            roll_mean = sub.groupby("commune_id")["n_cas"].mean().rename(col)
            rows = rows.merge(roll_mean, on="commune_id", how="left")

        # --- features PINN (etat mecaniste + fonctions climat) ---
        pinn_feats = compute_pinn_features(pinn_model, prov_to_idx, rows, t_ref_year)
        rows = pd.concat([rows, pinn_feats], axis=1)

        # --- features de voisinage (moyenne chez les k plus proches, meme province) ---
        src = rows[["commune", "cases_lag1", "cases_roll3", "psi_mean"]].rename(
            columns={"cases_lag1": "_n_lag1", "cases_roll3": "_n_roll3", "psi_mean": "_n_psi"}
        )
        merged = edges.merge(src.rename(columns={"commune": "neighbor"}), on="neighbor")
        nbr_agg = merged.groupby("commune").agg(
            neighbor_cases_lag1=("_n_lag1", "mean"),
            neighbor_cases_roll3=("_n_roll3", "mean"),
            neighbor_psi_mean=("_n_psi", "mean"),
        )
        rows = rows.merge(nbr_agg, on="commune", how="left")

        pred = predict_gbm_saved(gbm_saved, rows)
        rows["cas_predits"] = pred
        var = rows["cas_predits"] + 1e-6
        rows["ci_lower_95"] = np.clip(rows["cas_predits"] - 1.96 * np.sqrt(var), 0, None)
        rows["ci_upper_95"] = rows["cas_predits"] + 1.96 * np.sqrt(var)

        results.append(rows[["commune", "province", "region", "annee", "mois",
                              "cas_predits", "ci_lower_95", "ci_upper_95"]].copy())

        # reinjecter les predictions comme n_cas connu pour les mois suivants
        feedback = rows[["commune_id", "annee", "mois"]].copy()
        feedback["n_cas"] = rows["cas_predits"].values
        hist = pd.concat([hist, feedback], ignore_index=True)

        if (i + 1) % 24 == 0 or (year, month) == future_months[-1]:
            elapsed = time.time() - t_start
            logger.info(f"  {year}-{month:02d} termine ({i + 1}/{len(future_months)}), "
                        f"total predit ce mois = {rows['cas_predits'].sum():.1f} cas, "
                        f"{elapsed:.0f}s ecoulees")

    all_pred = pd.concat(results, ignore_index=True)
    future_pred = all_pred[all_pred["annee"] >= 2025].copy()

    comm_out = config.PROCESSED / "forecast_2025_2045_communes.csv"
    future_pred.to_csv(comm_out, index=False)
    logger.info(f"Projections par commune exportees : {comm_out} ({len(future_pred)} lignes)")

    future_pred["_var"] = ((future_pred["ci_upper_95"] - future_pred["ci_lower_95"]) / (2 * 1.96)) ** 2

    def aggregate_with_ci(df, group_cols):
        agg = df.groupby(group_cols).agg(cas_predits=("cas_predits", "sum"), _var=("_var", "sum")).reset_index()
        agg["ci_lower_95"] = np.clip(agg["cas_predits"] - 1.96 * np.sqrt(agg["_var"]), 0, None)
        agg["ci_upper_95"] = agg["cas_predits"] + 1.96 * np.sqrt(agg["_var"])
        return agg.drop(columns="_var")

    prov_df = aggregate_with_ci(future_pred, ["province", "region", "annee", "mois"])
    prov_out = config.PROCESSED / "forecast_2025_2045_provinces.csv"
    prov_df.to_csv(prov_out, index=False)
    logger.info(f"Projections par province exportees : {prov_out}")

    reg_df = aggregate_with_ci(future_pred, ["region", "annee", "mois"])
    reg_out = config.PROCESSED / "forecast_2025_2045_regions.csv"
    reg_df.to_csv(reg_out, index=False)
    logger.info(f"Projections par region exportees : {reg_out}")

    nat_annual = future_pred.groupby("annee")["cas_predits"].sum()
    logger.info(f"Total national predit par annee (2025-2045) :\n{nat_annual.round(1).to_string()}")
    logger.info(f"Termine en {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    run()
