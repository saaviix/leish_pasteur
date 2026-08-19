"""
occupancy_gp.py
================
Variante du modele d'occupation bayesien pour Phlebotomus sergenti : au lieu
du lissage spatial ICAR discret (voisinage Delaunay, cf. bayesian_occupancy.py),
la correlation spatiale entre provinces est modelisee par un noyau de
Gaussian Process a decroissance exponentielle sur la distance haversine reelle
(km) entre centroides -- un champ spatial continu plutot qu'un graphe discret.

    logit(psi_i) = a0 + X_i . a_clim + f_spatial(coords_i),   f_spatial ~ GP(0, K_exp)
    K_exp(d) = eta^2 * exp(-d / rho)

`rho` (portee spatiale, km) et `eta` (ecart-type du champ) sont estimes,
donnant une distance interpretable a laquelle la correlation retombe a ~37%.

A comparer avec bayesian_occupancy.py (ICAR) : les deux approches sont des
choix de modelisation differents pour le meme probleme (peu/pas de donnees
entomologiques dans la moitie des provinces) -- ce script produit un second
jeu d'estimations psi pour validation croisee des deux methodes plutot que de
remplacer l'ICAR.

Migre et adapte depuis l'ancien script exploratoire
`main src/occupancy_model_spatial_gp.py` (chemins ad hoc /home/claude/...
remplaces par config.py + les sorties du pipeline).

Entrees :
  outputs/processed/zone_bioclim_province.csv  (bioclimatic_zoning.py)
  data/raw/phlebotomus_sergenti_par_province.csv
  outputs/posterior/psergenti_posterior_presence.csv  (optionnel, pour comparaison avec l'ICAR)

Sorties :
  outputs/posterior/occupancy_gp_posterior.csv
  outputs/figures/occupancy_gp_vs_icar.png  (si l'ICAR a deja tourne)

Usage :
  python src/models/occupancy_gp.py
"""

import os
import sys
from pathlib import Path

# meme raisonnement que bayesian_occupancy.py : pas de g++ sur cette machine
# -> cxx="" desactive la compilation C, PyTensor bascule sur son linker
# numba (JIT). A definir avant le premier import de pytensor/pymc.
os.environ["PYTENSOR_FLAGS"] = "cxx="

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

FEAT_COLS = ["temp_moy_an", "precip_totale_an", "humidite_moy_an", "elevation_m"]


def detection_strength(status):
    if pd.isna(status):
        return np.nan
    s = str(status).lower()
    if s.startswith("oui"):
        return 2
    if "indirecte" in s or "ancienne" in s:
        return 1
    return 0


def n_sources(row):
    src = row.get("Source", None)
    if pd.isna(src) or str(src).strip() == "-":
        return 0
    return len([s for s in str(src).split(";") if s.strip()])


def haversine_matrix(lon, lat):
    R = 6371
    lon, lat = np.radians(lon), np.radians(lat)
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_data() -> pd.DataFrame:
    zone_path = config.PROCESSED / "zone_bioclim_province.csv"
    if not zone_path.exists():
        raise FileNotFoundError(
            f"{zone_path} introuvable.\nLance d'abord : python src/models/bioclimatic_zoning.py"
        )
    zone = pd.read_csv(zone_path)

    vec = pd.read_csv(config.SERGENTI_CSV)
    vec.columns = [c.strip() for c in vec.columns]
    vec["detection"] = vec["Statut_P_sergenti"].apply(detection_strength)
    vec["effort"] = vec.apply(n_sources, axis=1)

    data = zone.merge(vec[["Province", "detection", "effort"]],
                       left_on="province", right_on="Province", how="left")
    data["a_ete_etudiee"] = data["detection"].notna()

    for col in FEAT_COLS:
        data[f"{col}_z"] = (data[col] - data[col].mean()) / data[col].std()

    return data


def main() -> None:
    config.ensure_dirs()
    data = load_data()
    n = len(data)

    D = haversine_matrix(data["longitude"].values, data["latitude"].values)
    print(f"{n} provinces, distances de {D[D > 0].min():.0f} a {D.max():.0f} km")

    obs_mask = data["a_ete_etudiee"].values
    idx_obs = np.where(obs_mask)[0]
    y_strong = (data.loc[obs_mask, "detection"] == 2).astype(int).values
    effort_obs = data.loc[obs_mask, "effort"].clip(lower=0).values
    X_psi = data[[f"{c}_z" for c in FEAT_COLS]].values

    # ---- logit(psi) = a0 + X.a_clim + f_spatial(coords), f_spatial ~ GP(0, K_exp) ----
    # parametrisation non centree (f = L @ eta) pour un echantillonnage NUTS efficace
    with pm.Model():
        a0 = pm.Normal("a0", 0, 2)
        a_clim = pm.Normal("a_clim", 0, 1, shape=len(FEAT_COLS))

        rho = pm.HalfNormal("rho_km", 150)   # portee spatiale (km)
        eta = pm.HalfNormal("eta", 1.5)      # ecart-type du processus spatial

        K = eta**2 * pm.math.exp(-D / rho) + np.eye(n) * 1e-6
        L = pt.linalg.cholesky(K)
        f_raw = pm.Normal("f_raw", 0, 1, shape=n)
        f_spatial = pm.Deterministic("f_spatial", L @ f_raw)

        logit_psi = a0 + pm.math.dot(X_psi, a_clim) + f_spatial
        psi = pm.Deterministic("psi", pm.math.sigmoid(logit_psi))

        b0 = pm.Normal("b0", -1, 1)
        b_effort = pm.HalfNormal("b_effort", 1)
        p_detect_given_present = pm.math.sigmoid(b0 + b_effort * effort_obs)
        p_obs_strong = psi[idx_obs] * p_detect_given_present
        pm.Bernoulli("y_strong", p=p_obs_strong, observed=y_strong)

        idata = pm.sample(1500, tune=1500, chains=4, cores=1, target_accept=0.95,
                           progressbar=True, random_seed=0)

    summary = az.summary(idata, var_names=["a0", "a_clim", "rho_km", "eta", "b0", "b_effort"])
    print("\n" + summary.to_string())
    max_rhat = pd.to_numeric(summary["r_hat"], errors="coerce").max()
    print(f"\nMax r_hat: {max_rhat:.4f} (viser < 1.01)")
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")

    psi_samples = idata.posterior["psi"].values.reshape(-1, n)
    data["psi_gp_mean"] = psi_samples.mean(axis=0)
    data["psi_gp_q05"] = np.percentile(psi_samples, 5, axis=0)
    data["psi_gp_q95"] = np.percentile(psi_samples, 95, axis=0)

    rho_est = idata.posterior["rho_km"].values.mean()
    print(f"\nPortee spatiale estimee (rho) = {rho_est:.0f} km -- distance a laquelle "
          f"la correlation entre 2 provinces tombe a ~37% (1/e)")

    out_cols = ["region", "province", "a_ete_etudiee", "detection", "effort",
                "psi_gp_mean", "psi_gp_q05", "psi_gp_q95"]
    result = data[out_cols].sort_values("psi_gp_mean", ascending=False)
    out_path = config.POSTERIOR / "occupancy_gp_posterior.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[OK] {out_path}")

    # ---- comparaison avec l'ICAR (bayesian_occupancy.py), si deja execute ----
    if not config.POSTERIOR_CSV.exists():
        print(f"\n[INFO] {config.POSTERIOR_CSV} absent -> pas de comparaison avec l'ICAR "
              f"(lance bayesian_occupancy.py pour l'obtenir)")
        return

    icar = pd.read_csv(config.POSTERIOR_CSV)[["province", "psi_mean"]].rename(
        columns={"psi_mean": "psi_icar"}
    )
    compare = data[["province", "a_ete_etudiee", "psi_gp_mean"]].merge(icar, on="province", how="left")

    print("\nProvinces jamais etudiees -- comparaison ICAR vs GP spatial :")
    print(
        compare[~compare["a_ete_etudiee"]]
        .sort_values("psi_gp_mean", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        ax = axes[0]
        ax.scatter(compare["psi_icar"], compare["psi_gp_mean"],
                   c=compare["a_ete_etudiee"].map({True: "green", False: "blue"}), alpha=0.6)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("psi (ICAR)")
        ax.set_ylabel("psi (GP spatial)")
        ax.set_title("ICAR vs GP spatial\n(vert=documentee, bleu=jamais etudiee)")

        ax = axes[1]
        sc = ax.scatter(data["longitude"], data["latitude"], c=data["psi_gp_mean"],
                         cmap="RdYlGn", s=60, edgecolors="k", linewidths=0.3, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="psi (GP spatial)")
        ax.set_title("psi GP spatial, par province")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

        plt.tight_layout()
        fig_path = config.FIGURES / "occupancy_gp_vs_icar.png"
        plt.savefig(fig_path, dpi=130)
        print(f"\n[OK] figure : {fig_path}")
    except ImportError:
        print("[INFO] matplotlib absent -> figure de comparaison non generee")


if __name__ == "__main__":
    main()
