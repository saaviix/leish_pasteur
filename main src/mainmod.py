import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Données filtrées pour XGBoost, LightGBM et Random Forest
data = {
    "modele": ["XGBoost", "LightGBM", "Random Forest"],
    "MAE": [3.866702, 3.881292, 4.141054],
    "RMSE": [8.730408, 8.795232, 8.823029],
    "MedAE": [1.590518, 1.570283, 1.938277],
    "Deviance_Poisson": [8.861116, 8.984361, 9.341700],
    "Spearman": [0.337090, 0.327617, 0.324254],
    "R2": [0.115942, 0.102764, 0.097084],
    "Erreur_pic_mois": [2.500000, 2.607143, 2.500000],
}

df = pd.DataFrame(data)
models = df["modele"]
metrics = [col for col in df.columns if col != "modele"]

# 2. Configuration de la figure (grille de 2x4 pour 7 métriques)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Palette de couleurs personnalisée
colors = ["#2b5c8f", "#2ca02c", "#d62728"]

for i, metric in enumerate(metrics):
  ax = axes[i]
  values = df[metric]

  # Création des barres
  bars = ax.bar(models, values, color=colors, width=0.55, alpha=0.85)

  # Personnalisation des axes et du titre
  ax.set_title(metric, fontsize=12, fontweight="bold", pad=10)
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  ax.set_axisbelow(True)

  # Rotation des étiquettes des modèles pour plus de lisibilité
  ax.tick_params(axis="x", rotation=15, labelsize=9)

  # Ajout des valeurs au-dessus de chaque barre
  for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 points de décalage vertical
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# Suppression du 8ème subplot vide (puisqu'il y a 7 métriques)
fig.delaxes(axes[7])

# Ajustement global de la mise en page
plt.suptitle(
    "Comparaison des Performances : XGBoost vs LightGBM vs Random Forest",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()

# Affichage du graphique
plt.show()