

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "combined_leishmaniose_complet.csv"   
FIG_DIR = BASE_DIR / "documentationarticle" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "LCM": "#2E5266",
    "LCT": "#C0392B",
    "LV": "#D4A017",
    "primary": "#2E5266",
    "secondary": "#6E9887",
    "accent": "#C0392B",
    "grid": "#DDDDDD",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def load_and_clean(path: Path) -> pd.DataFrame:
    """Charge le CSV et applique un nettoyage minimal mais tracé."""
    df = pd.read_csv(path)
    df["Sexe"] = df["Sexe"].astype("category")
    df["Type_Leish"] = df["Type_Leish"].astype("category")
    df["Classification"] = df["Classification"].astype("category")
    df["Region"] = df["Region"].astype("category")
    df["Age_ans"] = pd.to_numeric(df["Age_ans"], errors="coerce")
    bins = [0, 5, 15, 25, 45, 65, 130]
    labels = ["0-4", "5-14", "15-24", "25-44", "45-64", "65+"]
    df["Classe_age"] = pd.cut(df["Age_ans"], bins=bins, labels=labels,
                               right=False, include_lowest=True)
    return df


def fig_missingness(df: pd.DataFrame) -> Path:
    miss = (df.isna().mean() * 100).sort_values(ascending=True)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(miss.index, miss.values, color=PALETTE["accent"])
    for bar, val in zip(bars, miss.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% de valeurs manquantes")
    ax.set_title("Complétude des variables du dataset")
    ax.set_xlim(0, max(miss.values) * 1.15)
    fig.tight_layout()
    path = FIG_DIR / "A_qualite_donnees.png"
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    path = fig_missingness(df)
    print(f"Figure enregistree : {path}")
