"""
Carte épidémiologique/écologique du Maroc — PAS un découpage administratif.
Panneau 1 : zones bioclimatiques (clustering sur climat réel + élévation), au niveau commune.
Panneau 2 : même géographie, charge de cas (log) + statut vecteur en surcouche.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

profile = pd.read_csv("/home/claude/zone_epi_final.csv")
communes = pd.read_csv("/mnt/user-data/uploads/communes_maroc_final.csv")

communes_z = communes.merge(
    profile[["province", "zone_bioclim", "total_cas", "statut_vecteur"]],
    on="province", how="left"
)

n_zones = profile["zone_bioclim"].nunique()
cmap = plt.get_cmap("tab10" if n_zones <= 10 else "tab20")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# --- Panneau 1 : zones bioclimatiques ---
ax = axes[0]
for z in sorted(communes_z["zone_bioclim"].dropna().unique()):
    sub = communes_z[communes_z["zone_bioclim"] == z]
    ax.scatter(sub["longitude"], sub["latitude"], s=8, color=cmap(int(z) % 10),
               label=f"Zone {int(z)}", alpha=0.75)
ax.set_title("Zonage écologique (climat réel ERA5 + élévation)\nniveau commune, PAS administratif")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(markerscale=2.5, fontsize=8, loc="lower left", ncol=2)
ax.set_aspect("equal")

# --- Panneau 2 : charge de cas + statut vecteur ---
ax = axes[1]
sizes = 6 + 2 * np.log1p(communes_z["total_cas"].fillna(0))
sc = ax.scatter(communes_z["longitude"], communes_z["latitude"], s=sizes,
                 c=np.log1p(communes_z["total_cas"].fillna(0)), cmap="YlOrRd", alpha=0.8)
vecteur_confirme = communes_z[communes_z["statut_vecteur"] == "confirme"]
ax.scatter(vecteur_confirme["longitude"], vecteur_confirme["latitude"], s=25,
           facecolors="none", edgecolors="blue", linewidths=0.6, label="P. sergenti confirmé (province)")
cbar = plt.colorbar(sc, ax=ax, fraction=0.04)
cbar.set_label("log(1 + cas cumulés, province)")
ax.set_title("Charge de cas LCT + statut vectoriel confirmé")
ax.set_xlabel("Longitude")
ax.legend(fontsize=8, loc="lower left")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("/home/claude/carte_epidemiologique.png", dpi=140)
print("Carte sauvegardée : /home/claude/carte_epidemiologique.png")
