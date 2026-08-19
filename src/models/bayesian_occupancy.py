"""
bayesian_occupancy.py
=====================
Modele d'occupation bayesien (occupancy model, MacKenzie 2002) avec lissage
spatial ICAR pour inferer la probabilite de presence de Phlebotomus sergenti
par province au Maroc -- Y COMPRIS pour les provinces SANS donnee (gap).

C'est la reponse a "inference bayesienne pour les donnees manquantes" :
le modele infere psi_i (proba de presence) pour chaque province via
  - un champ spatial ICAR (les provinces voisines se ressemblent)
  - des covariables : latitude + climat (temperature, precipitations, aridite)
    quand elles sont disponibles (sinon latitude seule)
  - l'evidence epidemiologique (cas LCT) et entomologique (captures).

Entrees :
  outputs/processed/province_table.csv
  outputs/processed/adjacency_edges.npy

Sorties :
  outputs/posterior/psergenti_posterior_presence.csv
  outputs/posterior/occupancy_trace.nc

Usage :
  python src/models/bayesian_occupancy.py
"""

import sys
from pathlib import Path

# Sur cette machine (Windows, pas de g++), PyTensor ne peut pas compiler en
# C -- cxx="" le desactive explicitement pour eviter qu'il essaie et echoue.
# Il utilise alors son linker numba (JIT, rapide, ne necessite pas de
# compilateur C). `pip install numba` est requis (absent -> ModuleNotFoundError,
# pas un "bug DLL" comme le suggerait un commentaire precedent ici -- verifie
# empiriquement). Doit etre defini AVANT le premier import de pytensor/pymc.
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# permettre d'importer config depuis src/data_prep
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402


def main() -> None:
    config.ensure_dirs()

    prov = pd.read_csv(config.PROVINCE_TABLE)
    edges = np.load(config.ADJ_EDGES)
    n = len(prov)

    # ---------- matrice d'adjacence pour ICAR ----------
    W = np.zeros((n, n), dtype=int)
    for a, b in edges:
        W[a, b] = 1
        W[b, a] = 1
    assert (W.sum(axis=1) > 0).all(), "noeud isole dans le graphe d'adjacence"

    # facteur d'echelle BYM2 (Riebler et al. 2016) : moyenne geometrique des
    # variances marginales du champ ICAR (via la pseudo-inverse de la matrice
    # de precision Q = D - W, jitter car Q est de rang n-1 -- un graphe
    # connexe a une seule composante). Ramene le champ ICAR brut a une echelle
    # unitaire pour que `rho` (part spatiale) soit reellement interpretable.
    D_deg = np.diag(W.sum(axis=1))
    Q = (D_deg - W).astype(float)
    jitter = max(np.diag(Q).mean() * 1e-4, 1e-6)
    Q_inv = np.linalg.inv(Q + np.eye(n) * jitter)
    scaling_factor = float(np.exp(np.mean(np.log(np.diag(Q_inv)))))
    print(f"[INFO] facteur d'echelle ICAR (BYM2) : {scaling_factor:.3f}")

    y_epi = prov["y_epi"].values.astype(int)
    hard_idx = np.where(prov["y_ento_hard"].values == 1)[0]
    soft_idx = np.where(prov["y_ento_soft"].values == 1)[0]

    # covariables disponibles (climat optionnel)
    lat_z = prov["lat_z"].values
    clim_cols = [c for c in ["temp_z", "precip_z", "arid_z"] if c in prov.columns]
    has_climate = bool(clim_cols) and prov[clim_cols].abs().sum().sum() > 0
    if has_climate:
        X_clim = prov[clim_cols].fillna(0).values
        print(f"[INFO] covariables climatiques utilisees : {clim_cols}")
    else:
        X_clim = None
        print("[INFO] pas de covariables climatiques -> latitude seule")

    with pm.Model() as occ_model:
        # ---- process : proba de presence reelle psi_i ----
        # Parametrisation BYM2 (Besag-York-Mollie, forme Riebler 2016) au lieu
        # du sigma_phi*phi_raw brut d'origine : on decompose l'effet spatial
        # en une part structuree (ICAR, `phi_icar`) et une part non-structuree
        # iid (`theta`), melangees par `rho` in [0,1], le tout mis a l'echelle
        # par `sigma_total`. C'est la reponse au probleme de geometrie en
        # entonnoir de l'ancienne version (sigma_phi ne convergeait pas,
        # r_hat 1.10-1.16 meme apres 4h de sampling) -- rho/sigma_total sont
        # beaucoup moins degeneres a explorer pour NUTS que sigma_phi seul.
        sigma_total = pm.HalfNormal("sigma_total", sigma=1.0)
        rho = pm.Beta("rho", alpha=2, beta=2)
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, shape=n)
        phi_icar = pm.ICAR("phi_icar", W=W, sigma=1.0)

        convolved = pt.sqrt(rho / scaling_factor) * phi_icar + pt.sqrt(1.0 - rho) * theta
        spatial_effect = pm.Deterministic("spatial_effect", sigma_total * convolved)

        alpha = pm.Normal("alpha", mu=-1.0, sigma=1.5)
        beta_lat = pm.Normal("beta_lat", mu=0.0, sigma=1.0)

        logit_psi = alpha + beta_lat * lat_z + spatial_effect
        if X_clim is not None:
            beta_clim = pm.Normal("beta_clim", mu=0.0, sigma=1.0, shape=X_clim.shape[1])
            logit_psi = logit_psi + pt.dot(X_clim, beta_clim)

        psi = pm.Deterministic("psi", pm.math.invlogit(logit_psi))

        # ---- detection epidemiologique (cas LCT) ----
        p_epi = pm.Beta("p_epi", alpha=2, beta=3)

        log_psi = pm.math.log(psi)
        log1m_psi = pm.math.log1p(-psi)
        log_p = pm.math.log(p_epi)
        log1m_p = pm.math.log1p(-p_epi)

        ll_detect = log_psi + log_p
        ll_nondetect = pm.math.logsumexp(
            pt.stack([log_psi + log1m_p, log1m_psi], axis=0), axis=0
        )
        loglik_epi = pt.where(pt.eq(y_epi, 1), ll_detect, ll_nondetect)
        pm.Potential("epi_evidence", loglik_epi.sum())

        if len(hard_idx) > 0:
            pm.Potential("hard_ento_evidence", log_psi[hard_idx].sum())

        if len(soft_idx) > 0:
            p_soft = pm.Beta("p_soft", alpha=1.5, beta=3)
            pm.Potential(
                "soft_ento_evidence",
                (log_psi[soft_idx] + pm.math.log(p_soft)).sum(),
            )

        # cores=4 valide empiriquement : 602s, 0 divergence, max r_hat=1.09
        # (cores=1 teste avec un budget reduit -> n'a meme pas fini de
        # compiler+echantillonner en 5 min, trop lent pour ce modele une fois
        # BYM2 ajoute ~150 parametres de plus que l'ancienne version).
        idata = pm.sample(
            draws=800, tune=800, chains=4, cores=4, target_accept=0.9,
            max_treedepth=10, random_seed=42, progressbar=True,
        )

    # ---------- diagnostics ----------
    var_names = ["alpha", "beta_lat", "sigma_total", "rho", "p_epi"]
    if X_clim is not None:
        var_names.append("beta_clim")
    if len(soft_idx):
        var_names.append("p_soft")
    summary = az.summary(idata, var_names=var_names)
    print(summary)
    max_rhat = pd.to_numeric(summary["r_hat"], errors="coerce").max()
    print(f"\nMax r_hat: {max_rhat:.4f} (viser < 1.01)")
    print(f"Divergences: {int(idata.sample_stats['diverging'].sum())}")

    # ---------- posterior par province ----------
    psi_samples = idata.posterior["psi"].values.reshape(-1, n)
    prov["psi_mean"] = psi_samples.mean(axis=0)
    prov["psi_sd"] = psi_samples.std(axis=0)
    prov["psi_q05"] = np.quantile(psi_samples, 0.05, axis=0)
    prov["psi_q50"] = np.quantile(psi_samples, 0.50, axis=0)
    prov["psi_q95"] = np.quantile(psi_samples, 0.95, axis=0)

    prov["evidence_type"] = np.select(
        [prov["y_ento_hard"] == 1, prov["y_ento_soft"] == 1, prov["y_epi"] == 1],
        ["confirmed_capture", "unverified_capture", "epi_only"],
        default="no_data_gap",
    )

    out_cols = ["province", "region", "n_communes", "lct_cases", "evidence_type",
                "psi_mean", "psi_sd", "psi_q05", "psi_q50", "psi_q95"]
    result = prov[out_cols].sort_values("psi_mean", ascending=False)
    result.to_csv(config.POSTERIOR_CSV, index=False, encoding="utf-8")
    trace_path = config.POSTERIOR / "occupancy_trace.nc"
    try:
        idata.to_netcdf(str(trace_path))
    except Exception as e:
        print("WARN: sauvegarde trace netcdf impossible:", e)

    print("\n=== Provinces 'gap' (aucune donnee), classees par proba de presence inferee ===")
    gap = result[result["evidence_type"] == "no_data_gap"]
    print(gap.to_string(index=False))
    print(f"\nEcrit : {config.POSTERIOR_CSV}")
    print(f"Ecrit : {trace_path}")


if __name__ == "__main__":
    main()
