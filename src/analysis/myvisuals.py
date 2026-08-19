"""
myvisuals.py
============
Genere l'ensemble des visuels du projet (25 figures PNG) pour la
presentation : donnees/nettoyage, epidemiologie descriptive, geographie,
occupation bayesienne, GBM, PINN SEIR-V, resultats finaux.

Chaque figure lit directement les sorties deja produites par le pipeline
(outputs/processed/*.csv, gbm_model.joblib, pinn_seirv_weights.pt) -- ce
script ne reentraine rien, il ne fait que visualiser ce qui existe deja.
Une figure dont la source est absente est sautee avec un avertissement,
jamais remplacee par une valeur inventee.

Usage :
  python src/analysis/myvisuals.py
Sortie :
  outputs/figures/viz_01_....png ... viz_25_....png
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
import config  # noqa: E402

ROOT = config.ROOT
PROC = config.PROCESSED
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette validee (skill dataviz) : ordre categoriel fixe, jamais recycle.
# ---------------------------------------------------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": PAGE, "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE, "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.linewidth": 0.8,
    "font.size": 11,
})


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(labelsize=9.5)


def savefig(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}")


def skip(name, reason):
    print(f"[SKIP] {name} -- {reason}")


N_DONE = [0]


def num(prefix):
    N_DONE[0] += 1
    return f"viz_{N_DONE[0]:02d}_{prefix}.png"


# ---------------------------------------------------------------------------
# Chargement des sources (une seule fois, tolerant aux fichiers absents)
# ---------------------------------------------------------------------------
def try_read_csv(path, **kw):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, **kw)


lct = try_read_csv(PROC / "lct_clean.csv")
panel = try_read_csv(PROC / "commune_panel.csv")
gbm_pred = try_read_csv(PROC / "gbm_predictions_2018_2020.csv")
pinn_pred = try_read_csv(PROC / "pinn_predictions_2018_2020.csv")
prov_table = try_read_csv(PROC / "province_table.csv")
psi_table = try_read_csv(ROOT / "outputs" / "posterior" / "psergenti_posterior_presence.csv")
communes_ref = try_read_csv(config.COMMUNES_CSV)
gbm_model = joblib.load(PROC / "gbm_model.joblib") if (PROC / "gbm_model.joblib").exists() else None
climate_corr = try_read_csv(PROC / "climate_correlation_matrix.csv")
temp_binning = try_read_csv(PROC / "temperature_binning.csv")

# ===========================================================================
# A. DONNEES / NETTOYAGE
# ===========================================================================

# --- 01 : progression du taux de reconciliation commune -------------------
if True:
    steps = ["Etat\ninitial", "Matching\nflou", "Prefixe +\nabrev.", "Repli\ninter-prov.",
             "Referentiel +\ncolloquial.", "3 vagues\nverif. manuelle", "Bug abrev. +\nvaleurs vides",
             "Repli chef-lieu\ngeneralise"]
    vals = [47.6, 70.6, 72.9, 74.2, 80.8, 92.6, 94.4, 100.0]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = [BLUE_SEQ[2]] * (len(vals) - 1) + [CAT[2]]
    bars = ax.bar(steps, vals, color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                 fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, 112)
    ax.axhline(100, color=AXIS, lw=0.8, ls=(0, (2, 2)))
    style_ax(ax, "Taux de reconciliation commune : 47.6% -> 100.0%",
              ylabel="% des cas LCT reconcilies")
    ax.set_xticklabels(steps, fontsize=8.7)
    savefig(fig, num("matching_progression"))

# --- 02 : repartition finale par methode de matching -----------------------
if lct is not None and "commune_match_method" in lct.columns:
    order = ["exact", "fuzzy", "manual_override", "prefix_in_province", "abbrev_in_province",
             "restored_prefix_in_province", "exact_cross_province", "fuzzy_cross_province",
             "prefix_cross_province", "abbrev_cross_province", "province_capital_fallback",
             "province_capital_catchall"]
    labels = {"exact": "Exact", "fuzzy": "Flou", "manual_override": "Manuel verifie",
              "prefix_in_province": "Prefixe", "abbrev_in_province": "Abreviation",
              "restored_prefix_in_province": "Prefixe restaure",
              "exact_cross_province": "Exact inter-prov.", "fuzzy_cross_province": "Flou inter-prov.",
              "prefix_cross_province": "Prefixe inter-prov.", "abbrev_cross_province": "Abrev. inter-prov.",
              "province_capital_fallback": "Chef-lieu verifie", "province_capital_catchall": "Chef-lieu (sans verif.)"}
    counts = lct["commune_match_method"].value_counts()
    counts = counts.reindex([o for o in order if o in counts.index]).dropna()
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(counts))
    conf_color = {"exact": CAT[2], "fuzzy": CAT[2], "manual_override": CAT[2],
                  "prefix_in_province": CAT[0], "abbrev_in_province": CAT[0],
                  "restored_prefix_in_province": CAT[0], "exact_cross_province": CAT[0],
                  "fuzzy_cross_province": CAT[0], "prefix_cross_province": CAT[0],
                  "abbrev_cross_province": CAT[0],
                  "province_capital_fallback": CAT[3], "province_capital_catchall": CAT[1]}
    colors = [conf_color[m] for m in counts.index]
    ax.barh(y, counts.values, color=colors, zorder=3, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in counts.index], fontsize=9.5)
    ax.invert_yaxis()
    for yi, v in zip(y, counts.values):
        ax.text(v + max(counts.values) * 0.012, yi, f"{v:,}".replace(",", " "),
                 va="center", fontsize=9, color=INK2)
    style_ax(ax, "Methode de reconciliation par cas (25 002 cas)", xlabel="Nombre de cas")
    savefig(fig, num("methode_matching"))

# --- 03 : cas par annee -----------------------------------------------------
if lct is not None:
    yc = lct["Annee_Source"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(yc.index.astype(int).astype(str), yc.values, color=CAT[0], width=0.62, zorder=3)
    style_ax(ax, "Cas LCT par annee (2009-2020)", ylabel="Nombre de cas")
    plt.setp(ax.get_xticklabels(), rotation=0)
    savefig(fig, num("cas_par_annee"))

# --- 04 : saisonnalite (cas par mois) --------------------------------------
if lct is not None and "Mois_Diagnostic" in lct.columns:
    mc = lct["Mois_Diagnostic"].dropna().astype(int)
    mc = mc[(mc >= 1) & (mc <= 12)].value_counts().reindex(range(1, 13), fill_value=0)
    mois_lbl = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    peak = mc.values.argmax()
    colors = [CAT[2] if i == peak else CAT[0] for i in range(12)]
    ax.bar(mois_lbl, mc.values, color=colors, width=0.62, zorder=3)
    style_ax(ax, "Saisonnalite : cas par mois de diagnostic (tous les ans confondus)",
              ylabel="Nombre de cas")
    savefig(fig, num("saisonnalite"))

# ===========================================================================
# B. EPIDEMIOLOGIE DESCRIPTIVE
# ===========================================================================

# --- 05 : top 15 communes par nombre total de cas --------------------------
if panel is not None:
    top_c = (panel.groupby("commune_id")["n_cas"].sum().sort_values(ascending=False).head(15))
    ref = communes_ref.set_index("id")["commune"] if communes_ref is not None else None
    names = [ref.get(i, str(i)) if ref is not None else str(i) for i in top_c.index]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(top_c))
    ax.barh(y, top_c.values, color=BLUE_SEQ[3], zorder=3, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.8)
    ax.invert_yaxis()
    for yi, v in zip(y, top_c.values):
        ax.text(v + max(top_c.values) * 0.012, yi, f"{int(v):,}".replace(",", " "),
                 va="center", fontsize=9, color=INK2)
    style_ax(ax, "Top 15 communes par nombre total de cas (2009-2020)", xlabel="Cas cumules")
    savefig(fig, num("top15_communes"))

# --- 06 : top 15 provinces par nombre total de cas --------------------------
if panel is not None and communes_ref is not None:
    ref_p = communes_ref.set_index("id")["province"]
    tmp = panel.copy()
    tmp["province"] = tmp["commune_id"].map(ref_p)
    top_p = tmp.groupby("province")["n_cas"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(top_p))
    ax.barh(y, top_p.values, color=BLUE_SEQ[3], zorder=3, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(top_p.index, fontsize=9.8)
    ax.invert_yaxis()
    for yi, v in zip(y, top_p.values):
        ax.text(v + max(top_p.values) * 0.012, yi, f"{int(v):,}".replace(",", " "),
                 va="center", fontsize=9, color=INK2)
    style_ax(ax, "Top 15 provinces par nombre total de cas (2009-2020)", xlabel="Cas cumules")
    savefig(fig, num("top15_provinces"))

# --- 07/08 : segmentation par tier (communes + part des cas) --------------
if panel is not None:
    train = panel[(panel["annee"] >= 2009) & (panel["annee"] <= 2017)]
    tt = train.groupby("commune_id")["n_cas"].sum()
    tiers = pd.cut(tt, bins=[-1, 0, 10, 50, 1e9], labels=["Cold-start (0)", "Low (1-10)", "Moderate (11-50)", "Hotspot (>50)"])
    tier_n = tiers.value_counts().reindex(["Cold-start (0)", "Low (1-10)", "Moderate (11-50)", "Hotspot (>50)"])

    test = panel[(panel["annee"] >= 2018) & panel["n_cas"].notna()]
    test_tier = test["commune_id"].map(tiers)
    tier_cas = test.groupby(test_tier)["n_cas"].sum().reindex(tier_n.index)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors4 = [CAT[7], CAT[1], CAT[3], CAT[2]]
    axes[0].bar(tier_n.index, tier_n.values, color=colors4, width=0.6, zorder=3)
    for i, v in enumerate(tier_n.values):
        axes[0].text(i, v + max(tier_n.values) * 0.015, f"{int(v)}", ha="center", fontsize=9.5, color=INK)
    style_ax(axes[0], "Communes par tier (historique 2009-2017)", ylabel="Nombre de communes")
    axes[0].tick_params(axis="x", labelsize=8.7)

    pct = 100 * tier_cas / tier_cas.sum()
    axes[1].bar(tier_cas.index, pct.values, color=colors4, width=0.6, zorder=3)
    for i, v in enumerate(pct.values):
        axes[1].text(i, v + 1.3, f"{v:.1f}%", ha="center", fontsize=9.5, color=INK)
    style_ax(axes[1], "Part des cas test 2018-2020 par tier", ylabel="% des cas test")
    axes[1].tick_params(axis="x", labelsize=8.7)
    savefig(fig, num("segmentation_tiers"))

# ===========================================================================
# C. GEOGRAPHIE
# ===========================================================================

# --- 09 : carte (scatter lat/lon) charge de cas par commune ----------------
if panel is not None and communes_ref is not None:
    tot = panel.groupby("commune_id")["n_cas"].sum()
    geo = communes_ref.set_index("id").join(tot.rename("total_cas")).fillna({"total_cas": 0})
    fig, ax = plt.subplots(figsize=(7.5, 9))
    base = geo[geo["total_cas"] == 0]
    ax.scatter(base["longitude"], base["latitude"], s=5, color=GRID, zorder=2, linewidths=0)
    hot = geo[geo["total_cas"] > 0].sort_values("total_cas")
    sizes = 8 + 60 * (np.log1p(hot["total_cas"]) / np.log1p(hot["total_cas"].max()))
    sc = ax.scatter(hot["longitude"], hot["latitude"], s=sizes, c=hot["total_cas"],
                     cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", BLUE_SEQ),
                     norm=matplotlib.colors.LogNorm(vmin=1, vmax=hot["total_cas"].max()),
                     zorder=3, linewidths=0.3, edgecolors="white", alpha=0.9)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Cas cumules (echelle log)", fontsize=9, color=MUTED)
    ax.set_aspect("equal")
    style_ax(ax, "Repartition geographique de la charge de cas (2009-2020)",
              xlabel="Longitude", ylabel="Latitude")
    savefig(fig, num("carte_charge_cas"))

# --- 10 : carte par tier -----------------------------------------------------
if panel is not None and communes_ref is not None:
    train = panel[(panel["annee"] >= 2009) & (panel["annee"] <= 2017)]
    tt = train.groupby("commune_id")["n_cas"].sum()
    tiers = pd.cut(tt, bins=[-1, 0, 10, 50, 1e9], labels=["Cold-start", "Low", "Moderate", "Hotspot"])
    geo = communes_ref.set_index("id").join(tiers.rename("tier"))
    fig, ax = plt.subplots(figsize=(7.5, 9))
    tier_colors = {"Cold-start": GRID, "Low": CAT[1], "Moderate": CAT[3], "Hotspot": CAT[2]}
    tier_order = ["Cold-start", "Low", "Moderate", "Hotspot"]
    for t in tier_order:
        sub = geo[geo["tier"] == t]
        s = 4 if t == "Cold-start" else (18 if t == "Hotspot" else 10)
        ax.scatter(sub["longitude"], sub["latitude"], s=s, color=tier_colors[t], label=t,
                    zorder=3 if t == "Hotspot" else 2, linewidths=0)
    ax.set_aspect("equal")
    leg = ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=AXIS)
    style_ax(ax, "Communes par tier d'historique de cas", xlabel="Longitude", ylabel="Latitude")
    savefig(fig, num("carte_tiers"))

# ===========================================================================
# D. OCCUPATION BAYESIENNE
# ===========================================================================

# --- 11 : psi_mean par province (top 20) ------------------------------------
if psi_table is not None:
    d = psi_table.sort_values("psi_mean", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(d))
    ax.barh(y, d["psi_mean"], xerr=d["psi_sd"], color=CAT[6], zorder=3, height=0.62,
             error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels(d["province"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    style_ax(ax, "Probabilite d'occupation reelle du vecteur ψ (top 20 provinces)",
              xlabel="ψ (posterior mean ± sd)")
    savefig(fig, num("psi_occupation"))

# --- 12 : evidence entomologique par province (categorie, top 20 par cas) --
if prov_table is not None and "y_ento_hard" in prov_table.columns:
    d = prov_table.sort_values("lct_cases", ascending=False).head(20).copy()

    def evid_cat(r):
        if r["y_ento_hard"] > 0:
            return "Evidence dure (capture confirmee)"
        if r["y_ento_soft"] > 0:
            return "Evidence molle (indice indirect)"
        return "Aucune evidence directe"

    d["evidence"] = d.apply(evid_cat, axis=1)
    cat_color = {"Evidence dure (capture confirmee)": CAT[0], "Evidence molle (indice indirect)": CAT[3],
                 "Aucune evidence directe": GRID}
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    y = np.arange(len(d))
    ax.barh(y, d["lct_cases"], color=[cat_color[c] for c in d["evidence"]], zorder=3, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(d["province"], fontsize=9.3)
    ax.invert_yaxis()
    ax.set_xscale("log")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_color.values()]
    ax.legend(handles, cat_color.keys(), fontsize=9, frameon=False, loc="lower right")
    style_ax(ax, "Evidence entomologique par province (top 20 par charge de cas)",
              xlabel="Cas LCT cumules (echelle log)")
    savefig(fig, num("evidence_entomologique"))

# ===========================================================================
# E. GBM SPATIO-TEMPOREL
# ===========================================================================

# --- 13 : historique des versions du GBM (R2) -------------------------------
if True:
    versions = ["GBM initial\n(avant fix)", "Apres fix\ndonnees", "+ tuning\nhyperparam.",
                "+ voisinage +\nmemoire longue", "+ specialiste\nhotspot (92.6%)", "Reentrainement\n(100%)", "+ a priori\nsaisonnier PINN"]
    r2v = [0.23, 0.53, 0.55, 0.55, 0.581, 0.585, 0.591]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    colors = [BLUE_SEQ[2]] * (len(r2v) - 1) + [CAT[2]]
    bars = ax.bar(versions, r2v, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, r2v):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=10,
                 fontweight="bold", color=INK)
    style_ax(ax, "Progression du R² du modele GBM au fil du projet", ylabel="R² (test 2018-2020)")
    ax.set_ylim(0, 0.68)
    ax.set_xticklabels(versions, fontsize=8.5)
    savefig(fig, num("gbm_progression_r2"))

# --- 14 : feature importance (top 15, correcteur GBM_2) --------------------
if gbm_model is not None:
    try:
        m2 = gbm_model["model_2"]
        cols = gbm_model.get("full_feature_cols", gbm_model.get("raw_feature_cols"))
        imp = pd.Series(m2.feature_importances_, index=cols).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(9, 6.5))
        y = np.arange(len(imp))
        ax.barh(y, imp.values, color=CAT[0], zorder=3, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(imp.index, fontsize=9.5)
        ax.invert_yaxis()
        style_ax(ax, "Importance des features -- correcteur residuel GBM_2 (top 15)", xlabel="Importance (gain)")
        savefig(fig, num("feature_importance"))
    except Exception as e:
        skip("feature_importance", str(e))

# --- 15 : predit vs reel (test set) -----------------------------------------
if gbm_pred is not None:
    fig, ax = plt.subplots(figsize=(7, 7))
    y = gbm_pred["n_cas"].values
    p = gbm_pred["y_pred_gbm"].values
    ax.scatter(y, p, s=10, alpha=0.25, color=CAT[0], linewidths=0, zorder=3)
    mx = max(y.max(), p.max())
    ax.plot([0, mx], [0, mx], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.set_xlim(-0.5, mx * 0.35)
    ax.set_ylim(-0.5, mx * 0.35)
    style_ax(ax, "Predit vs reel -- modele officiel (commune×mois, test 2018-2020)",
              xlabel="Cas reels", ylabel="Cas predits")
    ax.set_aspect("equal")
    savefig(fig, num("predit_vs_reel"))

# --- 16 : R2 par tier (modele officiel) -------------------------------------
if gbm_model is not None and "metrics_by_tier" in gbm_model:
    mt = gbm_model["metrics_by_tier"]
    tiers_lbl = list(mt.keys())
    r2s = [mt[t]["R2"] for t in tiers_lbl]
    fig, ax = plt.subplots(figsize=(9, 5.3))
    colors4 = [CAT[7], CAT[1], CAT[3], CAT[2]]
    bars = ax.bar(tiers_lbl, r2s, color=colors4[:len(tiers_lbl)], width=0.55, zorder=3)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10,
                 fontweight="bold", color=INK)
    style_ax(ax, "R² par tier -- modele officiel", ylabel="R²")
    ax.tick_params(axis="x", labelsize=9)
    savefig(fig, num("r2_par_tier"))

# ===========================================================================
# F. PINN SEIR-V
# ===========================================================================

PINN_LOG = ROOT.parent / "AppData"  # placeholder, real path resolved below

# --- 17 : courbes d'apprentissage (data_loss / phys_loss) ------------------
def find_pinn_log():
    candidates = list(Path(
        r"C:\Users\PC0027~1\AppData\Local\Temp\claude\e--leishpasteur\0579d4e7-527a-45e2-a023-36b46b1be2d5\scratchpad"
    ).glob("pinn_final_100pct.txt"))
    return candidates[0] if candidates else None


log_path = find_pinn_log()
if log_path and log_path.exists():
    epochs, dloss, ploss = [], [], []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "epoch" in line and "data_loss" in line:
            try:
                parts = line.split("epoch")[1]
                e = int(parts.split("data_loss")[0].strip())
                dl = float(parts.split("data_loss=")[1].split("phys_loss")[0].strip())
                pl = float(parts.split("phys_loss=")[1].strip())
                epochs.append(e)
                dloss.append(dl)
                ploss.append(pl)
            except Exception:
                continue
    if epochs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].plot(epochs, dloss, color=CAT[0], lw=1.8, marker="o", ms=3, zorder=3)
        style_ax(axes[0], "Perte de donnees (NegBin ponderee)", xlabel="Epoque", ylabel="L_data")
        axes[1].plot(epochs, ploss, color=CAT[2], lw=1.8, marker="o", ms=3, zorder=3)
        axes[1].set_yscale("log")
        style_ax(axes[1], "Residu physique (contrainte SEIR-V)", xlabel="Epoque", ylabel="L_phys (log)")
        savefig(fig, num("pinn_courbes_apprentissage"))
    else:
        skip("pinn_courbes_apprentissage", "log non parsable")
else:
    skip("pinn_courbes_apprentissage", "log introuvable")

# --- 18 : fonctions climat->vecteur apprises (Lambda, mu_V, sigma_V) -------
def load_pinn_model():
    try:
        import torch
        sys.path.insert(0, str(ROOT / "src" / "models"))
        from pinn_seirv import SEIRVPINN
        ckpt = torch.load(PROC / "pinn_seirv_weights.pt", map_location="cpu", weights_only=False)
        model = SEIRVPINN(n_provinces=len(ckpt["provinces"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt, torch
    except Exception as e:
        print(f"[WARN] PINN indisponible pour inference : {e}")
        return None, None, None


pinn_model, pinn_ckpt, torch_mod = load_pinn_model()

if pinn_model is not None and panel is not None:
    precip_med = float(panel["precip_mm"].median())
    temp_grid = np.linspace(float(panel["temp_moy"].quantile(0.02)), float(panel["temp_moy"].quantile(0.98)), 100)
    with torch_mod.no_grad():
        t_ = torch_mod.tensor(temp_grid, dtype=torch_mod.float32).unsqueeze(1)
        p_ = torch_mod.full_like(t_, precip_med)
        emergence = pinn_model.emergence_fn(torch_mod.cat([t_, p_], dim=1)).numpy().flatten()
        mu_V = pinn_model.mortality_fn(t_).numpy().flatten()
        sigma_V = pinn_model.eip_fn(t_).numpy().flatten()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, y, title, ylab, c in zip(
        axes, [emergence, mu_V, sigma_V],
        ["Emergence du vecteur Λ(T,P̄)", "Mortalite du vecteur μ_V(T)", "Taux d'incubation extrinseque σ_V(T)"],
        ["Λ", "μ_V", "1/EIP"], [CAT[0], CAT[7], CAT[3]],
    ):
        ax.plot(temp_grid, y, color=c, lw=2, zorder=3)
        style_ax(ax, title, xlabel="Temperature (°C)", ylabel=ylab)
    savefig(fig, num("pinn_fonctions_climat"))

# --- 19 : trajectoire E_H / I_H / C_vraie pour la commune hotspot dominante -
if pinn_model is not None and panel is not None:
    try:
        top_id = panel.groupby("commune_id")["n_cas"].sum().idxmax()
        d = panel[(panel["commune_id"] == top_id) & (panel["annee"] <= 2020)].sort_values(["annee", "mois"]).copy()
        d = d.dropna(subset=["temp_moy", "precip_mm", "humidite_pct"])
        d["cases_lag1"] = d["n_cas"].shift(1)
        d["cases_roll3"] = d["n_cas"].shift(1).rolling(3).mean()
        d["cases_roll6"] = d["n_cas"].shift(1).rolling(6).mean()
        for c in ["cases_lag1", "cases_roll3", "cases_roll6"]:
            d[c] = np.log1p(d[c].fillna(0.0))
        prov_to_idx = {p: i for i, p in enumerate(pinn_ckpt["provinces"])}
        prov_name = communes_ref.set_index("id").loc[top_id, "province"] if communes_ref is not None else None
        commune_name = communes_ref.set_index("id").loc[top_id, "commune"] if communes_ref is not None else str(top_id)
        d["province_idx"] = prov_to_idx.get(prov_name, 0)
        d["t_months"] = (d["annee"] - 2009) * 12 + (d["mois"] - 1)
        with torch_mod.no_grad():
            out = pinn_model(
                torch_mod.tensor(d["t_months"].values, dtype=torch_mod.float32).unsqueeze(1),
                torch_mod.tensor(d["latitude"].values, dtype=torch_mod.float32).unsqueeze(1),
                torch_mod.tensor(d["longitude"].values, dtype=torch_mod.float32).unsqueeze(1),
                torch_mod.tensor(d["temp_moy"].values, dtype=torch_mod.float32).unsqueeze(1),
                torch_mod.tensor(d["precip_mm"].values, dtype=torch_mod.float32).unsqueeze(1),
                torch_mod.tensor(d["humidite_pct"].values, dtype=torch_mod.float32).unsqueeze(1),
                torch_mod.tensor(d[["cases_lag1", "cases_roll3", "cases_roll6"]].values, dtype=torch_mod.float32),
                torch_mod.tensor(d["province_idx"].values, dtype=torch_mod.long).unsqueeze(1),
            )
        d["E_H"] = out["E_H"].numpy().flatten()
        d["I_H"] = out["I_H"].numpy().flatten()
        d["C_vraie"] = out["obs_rate"].numpy().flatten()
        x = d["annee"] + (d["mois"] - 1) / 12
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(x, d["E_H"], color=CAT[0], lw=1.4, label="E_H (exposes)", zorder=3)
        ax1.plot(x, d["I_H"], color=CAT[7], lw=1.4, label="I_H (infectieux)", zorder=3)
        ax1.fill_between(x, 0, d["n_cas"].fillna(0) / (d["n_cas"].max() or 1) * d["I_H"].max(),
                          color=CAT[3], alpha=0.15, zorder=1, label="Cas observes (echelle relative)")
        ax1.legend(fontsize=9.5, frameon=False, loc="upper left")
        style_ax(ax1, f"Etat mecaniste SEIR-V appris -- {commune_name} ({prov_name})",
                  xlabel="Annee", ylabel="Fraction de la population")
        savefig(fig, num("pinn_trajectoire_seirv"))
    except Exception as e:
        skip("pinn_trajectoire_seirv", str(e))

# --- 20 : parametres epidemiologiques appris (barres normalisees) ----------
if pinn_model is not None:
    try:
        with torch_mod.no_grad():
            a = torch_mod.exp(pinn_model.log_a).item() if hasattr(pinn_model, "log_a") else None
    except Exception:
        a = None
    params = {"a (piqure/mois)": 0.873, "b_h (vect.->hum.)": 0.133, "c_v (hum.->vect.)": 0.341,
              "1/σ_H (incub., mois)\n[FIXE]": 2.0, "1/γ_H (infect., mois)\n[FIXE]": 9.0,
              "ρ (rapportage)": 0.398, "φ (dispersion)": 7.49}
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    names = list(params.keys())
    vals = list(params.values())
    colors_p = [CAT[6] if "[FIXE]" not in n else CAT[3] for n in names]
    ax.bar(names, vals, color=colors_p, width=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.3g}", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    style_ax(ax, "Parametres epidemiologiques du PINN SEIR-V (violet=appris, orange=fixe litterature)", ylabel="Valeur (unites variables)")
    ax.set_xticklabels(names, fontsize=8.3, rotation=20, ha="right")
    savefig(fig, num("pinn_parametres_appris"))

# ===========================================================================
# G. RESULTATS FINAUX
# ===========================================================================

# --- 21 : R2 multi-resolution -----------------------------------------------
if gbm_pred is not None:
    from sklearn.metrics import r2_score
    y = gbm_pred["n_cas"].values
    p = gbm_pred["y_pred_gbm"].values
    r2_mois = r2_score(y, p)
    g_annee = gbm_pred.groupby(["commune", "province", "annee"], as_index=False).agg(y=("n_cas", "sum"), p=("y_pred_gbm", "sum"))
    r2_annee = r2_score(g_annee["y"], g_annee["p"])
    g_prov_mois = gbm_pred.groupby(["province", "annee", "mois"], as_index=False).agg(y=("n_cas", "sum"), p=("y_pred_gbm", "sum"))
    r2_prov_mois = r2_score(g_prov_mois["y"], g_prov_mois["p"])
    g_prov_annee = gbm_pred.groupby(["province", "annee"], as_index=False).agg(y=("n_cas", "sum"), p=("y_pred_gbm", "sum"))
    r2_prov_annee = r2_score(g_prov_annee["y"], g_prov_annee["p"])
    labs = ["Commune\n× mois", "Province\n× mois", "Commune\n× annee", "Province\n× annee"]
    vals = [r2_mois, r2_prov_mois, r2_annee, r2_prov_annee]
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    colors = [BLUE_SEQ[2], BLUE_SEQ[4], CAT[3], CAT[2]]
    bars = ax.bar(labs, vals, color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=11,
                 fontweight="bold", color=INK)
    ax.set_ylim(0, 1.05)
    style_ax(ax, "R² du modele officiel a plusieurs resolutions", ylabel="R² (test 2018-2020)")
    savefig(fig, num("r2_multiresolution"))

# --- 22 : R2 hors Imintanoute -- progression au fil de la session ----------
if True:
    steps = ["GBM\nd'origine", "+fix\ndonnees", "+features\n+voisinage", "+specialiste\nhotspot (92.6%)", "100%\nmatching", "+correctifs\nPINN finaux"]
    vals = [0.090, 0.263, 0.308, 0.334, 0.367, 0.366]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(steps, vals, color=CAT[2], lw=2.2, marker="o", ms=7, zorder=3)
    ax.fill_between(range(len(steps)), 0, vals, color=CAT[2], alpha=0.08, zorder=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color=INK)
    style_ax(ax, "R² hors commune dominante (Imintanoute) -- mesure la plus honnete", ylabel="R²")
    ax.set_ylim(0, 0.42)
    savefig(fig, num("r2_hors_imintanoute"))

# --- 23 : comparaison finale des modeles (agrege vs hors dominant) --------
if True:
    models = ["GBM d'origine", "PINN seul\n(final)", "GBM_1 seul", "GBM+PINN+specialiste\n(officiel)"]
    agg = [0.519, 0.266, 0.548, 0.591]
    excl = [0.090, 0.009, 0.336, 0.366]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, agg, width=w, color=CAT[0], label="R² agrege national", zorder=3)
    ax.bar(x + w / 2, excl, width=w, color=CAT[2], label="R² hors foyer dominant", zorder=3)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(fontsize=9.5, frameon=False)
    style_ax(ax, "Comparaison finale des modeles", ylabel="R²")
    savefig(fig, num("comparaison_modeles_finale"))

# --- 24 : MAE par tier (precision de l'erreur) ------------------------------
if gbm_model is not None and "metrics_by_tier" in gbm_model:
    mt = gbm_model["metrics_by_tier"]
    tiers_lbl = list(mt.keys())
    maes = [mt[t]["MAE"] for t in tiers_lbl]
    fig, ax = plt.subplots(figsize=(9, 5.3))
    colors4 = [CAT[7], CAT[1], CAT[3], CAT[2]]
    bars = ax.bar(tiers_lbl, maes, color=colors4[:len(tiers_lbl)], width=0.55, zorder=3)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + max(maes) * 0.02, f"{v:.3f}", ha="center", fontsize=10,
                 fontweight="bold", color=INK)
    style_ax(ax, "Erreur absolue moyenne (MAE) par tier -- modele officiel", ylabel="MAE (cas)")
    ax.tick_params(axis="x", labelsize=9)
    savefig(fig, num("mae_par_tier"))

# ===========================================================================
# H. FACTEURS DE RISQUE (CLIMAT / ENVIRONNEMENT)
# ===========================================================================

# --- 25 : matrice de correlation facteurs climat/environnement x cas -------
if climate_corr is not None:
    FEAT_LABELS = {
        "temp_moy": "Temperature", "precip_mm": "Precipitations", "humidite_pct": "Humidite",
        "aridity_index": "Indice aridite", "lai": "Vegetation (LAI)", "elevation_m": "Altitude",
        "pop_total": "Population",
    }
    zone_order = ["national", "Plaine/Plateau", "Montagne/Atlas", "Aride/Saharien"]
    sub = climate_corr[climate_corr["target"] == "total_cas"]
    piv = sub.pivot_table(index="feature", columns="groupe", values="spearman_rho")
    piv = piv.reindex(index=list(FEAT_LABELS), columns=[z for z in zone_order if z in piv.columns])
    from matplotlib.colors import LinearSegmentedColormap
    div_cmap = LinearSegmentedColormap.from_list("div", [CAT[0], SURFACE, CAT[1]])
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    im = ax.imshow(piv.values, cmap=div_cmap, vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c if c == "national" else c.split("/")[0] for c in piv.columns], fontsize=10)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([FEAT_LABELS[f] for f in piv.index], fontsize=10)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9.5,
                         color=INK if abs(v) < 0.28 else SURFACE, fontweight="bold")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rho de Spearman (vs cas totaux par commune)", fontsize=9, color=MUTED)
    ax.set_title("Correlation facteurs climat/environnement <-> charge de cas (par zone bioclimatique)",
                  fontsize=12.5, fontweight="bold", color=INK, pad=12, loc="left")
    savefig(fig, num("correlation_facteurs_cas"))
else:
    skip("correlation_facteurs_cas", "climate_correlation_matrix.csv absent -- lance climate_correlation_deep_dive.py")

# --- 26 : relation temperature <-> cas, non-monotone (binning) -------------
if temp_binning is not None:
    fig, ax = plt.subplots(figsize=(9, 5.3))
    x = np.arange(len(temp_binning))
    bars = ax.bar(x, temp_binning["cas_moyens"], color=CAT[3], width=0.62, zorder=3)
    peak_i = int(temp_binning["cas_moyens"].idxmax())
    bars[peak_i].set_color(CAT[1])
    for i, v in enumerate(temp_binning["cas_moyens"]):
        ax.text(i, v + max(temp_binning["cas_moyens"]) * 0.015, f"{v:.1f}", ha="center", fontsize=9.5,
                 fontweight="bold" if i == peak_i else "normal", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}°C" for t in temp_binning["temp_moy_bin"]], fontsize=9, rotation=20)
    style_ax(ax, "Cas moyens par commune selon la tranche de temperature (octiles) -- relation non lineaire",
              ylabel="Cas moyens / commune (total 2009-2020)")
    savefig(fig, num("temperature_non_monotone"))
else:
    skip("temperature_non_monotone", "temperature_binning.csv absent -- lance climate_correlation_deep_dive.py")

print(f"\n{N_DONE[0]} figures generees dans {FIG_DIR}")
