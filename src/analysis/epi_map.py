"""
epi_map.py
==========
Carte epidemiologique/ecologique du Maroc au niveau COMMUNE -- PAS un
decoupage administratif.

  Panneau 1 : zones bioclimatiques (bioclimatic_zoning.py), niveau commune.
  Panneau 2 : meme geographie, charge de cas LCT (log) + statut vecteur
              confirme en surcouche (province).

Migre et adapte depuis l'ancien script exploratoire
`main src/carte_epidemiologique.py` (chemins ad hoc /home/claude/...
remplaces par config.py).

Entrees :
  outputs/processed/zone_bioclim_province.csv  (bioclimatic_zoning.py)
  data/raw/communes_maroc_final.csv

Sortie :
  outputs/figures/epi_map.png

Usage :
  python src/analysis/epi_map.py
"""

import sys
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "data_prep"))
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from data_prep import config  # noqa: E402


def main() -> None:
    config.ensure_dirs()

    zone_path = config.PROCESSED / "zone_bioclim_province.csv"
    if not zone_path.exists():
        raise FileNotFoundError(
            f"{zone_path} introuvable.\nLance d'abord : python src/models/bioclimatic_zoning.py"
        )
    profile = pd.read_csv(zone_path)
    communes = pd.read_csv(config.COMMUNES_CSV)

    communes_z = communes.merge(
        profile[["province", "zone_bioclim", "total_cas", "statut_vecteur"]],
        on="province", how="left",
    )

    n_zones = profile["zone_bioclim"].nunique()
    cmap = plt.get_cmap("tab10" if n_zones <= 10 else "tab20")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ---- panneau 1 : zones bioclimatiques ----
    ax = axes[0]
    for z in sorted(communes_z["zone_bioclim"].dropna().unique()):
        sub = communes_z[communes_z["zone_bioclim"] == z]
        ax.scatter(sub["longitude"], sub["latitude"], s=8, color=cmap(int(z) % 10),
                   label=f"Zone {int(z)}", alpha=0.75)
    ax.set_title("Zonage ecologique (climat reel ERA5 + elevation)\nniveau commune, PAS administratif")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=2.5, fontsize=8, loc="lower left", ncol=2)
    ax.set_aspect("equal")

    # ---- panneau 2 : charge de cas + statut vecteur ----
    ax = axes[1]
    sizes = 6 + 2 * np.log1p(communes_z["total_cas"].fillna(0))
    sc = ax.scatter(communes_z["longitude"], communes_z["latitude"], s=sizes,
                     c=np.log1p(communes_z["total_cas"].fillna(0)), cmap="YlOrRd", alpha=0.8)
    vecteur_confirme = communes_z[communes_z["statut_vecteur"] == "confirme"]
    ax.scatter(vecteur_confirme["longitude"], vecteur_confirme["latitude"], s=25,
               facecolors="none", edgecolors="blue", linewidths=0.6,
               label="P. sergenti confirme (province)")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04)
    cbar.set_label("log(1 + cas cumules, province)")
    ax.set_title("Charge de cas LCT + statut vectoriel confirme")
    ax.set_xlabel("Longitude")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path = config.FIGURES / "epi_map.png"
    plt.savefig(out_path, dpi=140)
    print(f"[OK] carte sauvegardee : {out_path}")


if __name__ == "__main__":
    main()
