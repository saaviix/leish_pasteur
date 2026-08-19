"""
pinn_seasonal_dynamics.py
===========================
Reponse directe a "comment les parametres de transmission evoluent au fil de
l'annee, pour chaque categorie" : le PINN SEIR-V (pinn_seirv.py) apprend TROIS
fonctions climat-dependantes (emergence du vecteur, mortalite du vecteur,
taux d'incubation extrinseque), en plus de 5 constantes biologiques
partagees. Ce script evalue ces 3 fonctions APPRISES (poids deja entraines,
pas de reentrainement) sur le climat mensuel moyen de chaque classe
d'altitude -- ca donne directement un profil saisonnier de la dynamique
vectorielle, PAR categorie, a partir du vrai modele physique, pas d'une
approximation.

Entrees :
  outputs/processed/pinn_seirv_weights.pt   (src/models/pinn_seirv.py)
  outputs/processed/commune_panel.csv
  outputs/processed/province_elevation_classes.csv

Sortie :
  outputs/processed/pinn_seasonal_by_class.csv

Usage :
  python src/analysis/pinn_seasonal_dynamics.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
sys.path.insert(0, str(ROOT / "src" / "models"))
import config  # noqa: E402
from pinn_seirv import SEIRVPINN  # noqa: E402


def load_model() -> SEIRVPINN:
    path = config.PROCESSED / "pinn_seirv_weights.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable.\nLance d'abord : python src/models/pinn_seirv.py")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SEIRVPINN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def monthly_climate_by_class(panel: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    panel = panel.merge(classes[["province", "classe_altitude", "classe_altitude_rank"]], on="province", how="left")
    panel = panel.dropna(subset=["classe_altitude"])
    monthly = (
        panel.groupby(["classe_altitude_rank", "classe_altitude", "mois"])
        .agg(temp_moy=("temp_moy", "mean"), precip_mm=("precip_mm", "mean"))
        .reset_index()
    )
    return monthly


def main() -> None:
    config.ensure_dirs()
    model = load_model()

    panel = pd.read_csv(config.PROCESSED / "commune_panel.csv", usecols=["province", "mois", "temp_moy", "precip_mm"])
    classes_path = config.PROCESSED / "province_elevation_classes.csv"
    if not classes_path.exists():
        raise FileNotFoundError(f"{classes_path} introuvable.\nLance d'abord : python src/analysis/elevation_classification.py")
    classes = pd.read_csv(classes_path)

    monthly = monthly_climate_by_class(panel, classes)

    with torch.no_grad():
        temp_t = torch.tensor(monthly["temp_moy"].values, dtype=torch.float32).unsqueeze(1)
        precip_t = torch.tensor(monthly["precip_mm"].values, dtype=torch.float32).unsqueeze(1)
        emergence = model.emergence_fn(torch.cat([temp_t, precip_t], dim=1)).numpy().flatten()
        mu_V = model.mortality_fn(temp_t).numpy().flatten()
        sigma_V = model.eip_fn(temp_t).numpy().flatten()

    monthly["emergence_vecteur"] = emergence
    monthly["mortalite_vecteur_mu_V"] = mu_V
    monthly["duree_incubation_extrinseque_jours"] = (1.0 / sigma_V) * 30.44
    # capacite vectorielle "proxy" : rapport emergence/mortalite -> taille
    # d'equilibre relative de la population de vecteurs (pas un R0 formel,
    # mais un indicateur monotone de favorabilite climatique au vecteur)
    monthly["capacite_vectorielle_proxy"] = emergence / mu_V

    monthly = monthly.sort_values(["classe_altitude_rank", "mois"])
    out_path = config.PROCESSED / "pinn_seasonal_by_class.csv"
    monthly.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n{'='*100}\nPARAMETRES BIOLOGIQUES CONSTANTS (partages, appris par le PINN)\n{'='*100}")
    print(f"  taux de piqure a          = {model.a.item():.4f} / mois")
    print(f"  b_h (vecteur -> humain)   = {model.b_h.item():.4f}")
    print(f"  c_v (humain -> vecteur)   = {model.c_v.item():.4f}")
    print(f"  1/sigma_H (incubation)    = {1.0/model.sigma_H.item():.2f} mois")
    print(f"  1/gamma_H (infectiosite)  = {1.0/model.gamma_H.item():.2f} mois")

    print(f"\n{'='*100}\nDYNAMIQUE VECTORIELLE SAISONNIERE APPRISE, PAR CLASSE D'ALTITUDE\n{'='*100}")
    for rank, g in monthly.groupby("classe_altitude_rank"):
        label = g["classe_altitude"].iloc[0]
        print(f"\n--- [{rank}] {label} ---")
        peak = g.loc[g["capacite_vectorielle_proxy"].idxmax()]
        low = g.loc[g["capacite_vectorielle_proxy"].idxmin()]
        print(g[["mois", "temp_moy", "precip_mm", "emergence_vecteur", "mortalite_vecteur_mu_V",
                 "duree_incubation_extrinseque_jours", "capacite_vectorielle_proxy"]].to_string(index=False))
        print(f"  -> pic de capacite vectorielle : mois {int(peak['mois'])} ({peak['temp_moy']:.1f}C)  |  "
              f"minimum : mois {int(low['mois'])} ({low['temp_moy']:.1f}C)")

    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
