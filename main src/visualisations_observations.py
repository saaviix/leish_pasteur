import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

climate = pd.read_csv("/home/claude/climat_reel_province_mois.csv")
zone_epi = pd.read_csv("/home/claude/zone_epi_final.csv")[["province", "zone_bioclim"]]
features = pd.read_csv("/home/claude/features_finales.csv")

climate_z = climate.merge(zone_epi, on="province")

# ---------------------------------------------------------------------------
# 1. Profil climatique mensuel moyen par zone bioclimatique
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
monthly_profile = climate_z.groupby(["zone_bioclim", "mois"])[["temp_moy", "precip_mm", "humidite_pct"]].mean().reset_index()

for var, ax, title in zip(["temp_moy", "precip_mm", "humidite_pct"], axes,
                            ["Température (°C)", "Précipitations (mm/mois)", "Humidité relative (%)"]):
    for z, g in monthly_profile.groupby("zone_bioclim"):
        ax.plot(g["mois"], g[var], marker="o", markersize=3, label=f"Zone {int(z)}")
    ax.set_title(title)
    ax.set_xlabel("Mois")
    ax.set_xticks(range(1, 13))
axes[0].legend(fontsize=7, ncol=2)
plt.suptitle("Profil climatique mensuel réel par zone bioclimatique (2009-2021)")
plt.tight_layout()
plt.savefig("/home/claude/profil_climatique_par_zone.png", dpi=130)
print("Sauvegardé : profil_climatique_par_zone.png")

# ---------------------------------------------------------------------------
# 2. Relation température retardée (2 mois) <-> cas -- répond direct à Q1
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
valid = features.dropna(subset=["temp_lag2", "cas"])
ax.scatter(valid["temp_lag2"], valid["cas"], alpha=0.25, s=15)
z = np.polyfit(valid["temp_lag2"], valid["cas"], 2)
xs = np.linspace(valid["temp_lag2"].min(), valid["temp_lag2"].max(), 100)
ax.plot(xs, np.polyval(z, xs), color="red", lw=2, label="tendance (poly deg 2)")
corr = valid["temp_lag2"].corr(valid["cas"])
ax.set_title(f"Cas mensuels vs température 2 mois avant\n(corrélation = {corr:.2f})")
ax.set_xlabel("Température moyenne, retard 2 mois (°C)")
ax.set_ylabel("Cas / mois")
ax.legend()

ax = axes[1]
top_province = features.groupby("province")["cas"].sum().idxmax()
sub = features[features["province"] == top_province].sort_values(["annee", "mois"])
t_axis = sub["annee"] + (sub["mois"] - 1) / 12
ax2 = ax.twinx()
ax.plot(t_axis, sub["cas"], color="black", label="Cas observés")
ax2.plot(t_axis, sub["temp_lag2"], color="tab:red", alpha=0.7, label="Température (retard 2 mois)")
ax.set_ylabel("Cas / mois")
ax2.set_ylabel("Température retardée (°C)", color="tab:red")
ax.set_title(f"Cas vs température retardée dans le temps — {top_province}")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("/home/claude/climat_vs_cas.png", dpi=130)
print("Sauvegardé : climat_vs_cas.png")
print(f"\nCorrélation cas / temp_lag2 (toutes provinces confondues) : {corr:.3f}")
