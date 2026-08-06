"""
summarize_results.py
====================
Lit les outputs du modele bayesien et produit un resume lisible :
  - tableau des provinces classees par risque
  - liste des provinces "gap" (inference)
  - statistiques du modele (r_hat, divergences sauvegardees)

Entrees :
  outputs/posterior/psergenti_posterior_presence.csv

Sorties :
  outputs/figures/results_summary.txt
  stdout

Usage :
  python src/models/summarize_results.py
"""

import sys
from pathlib import Path

import pandas as pd

# rendre l'import de config robuste quel que soit le cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main() -> None:
    csv_path = config.POSTERIOR_CSV
    if not csv_path.exists():
        print(f"ERREUR : {csv_path} introuvable.")
        print("Lance d'abord : python src/models/bayesian_occupancy.py")
        sys.exit(1)

    prov = pd.read_csv(csv_path)

    # renommer pour l'affichage
    display = prov.copy()
    display["psi_mean"] = display["psi_mean"].apply(lambda v: f"{v:.3f}")
    display["psi_q05"] = display["psi_q05"].apply(lambda v: f"{v:.3f}")
    display["psi_q95"] = display["psi_q95"].apply(lambda v: f"{v:.3f}")

    lines = []
    lines.append("=" * 72)
    lines.append("RESUME DES RESULTATS - Modele bayesien P. sergenti (Maroc)")
    lines.append("=" * 72)
    lines.append("")

    # top 10 risque
    lines.append("--- TOP 10 provinces (proba de presence la plus elevee) ---")
    top10 = display.head(10)
    for _, row in top10.iterrows():
        lines.append(f"  {row['province']:25s} | {row['region']:20s} | "
                     f"psi={row['psi_mean']} [{row['psi_q05']} - {row['psi_q95']}] | {row['evidence_type']}")
    lines.append("")

    # provinces gap
    gap = display[display["evidence_type"] == "no_data_gap"]
    lines.append(f"--- Provinces SANS donnee (inference bayesienne) : {len(gap)} ---")
    for _, row in gap.iterrows():
        lines.append(f"  {row['province']:25s} | {row['region']:20s} | "
                     f"psi={row['psi_mean']} [{row['psi_q05']} - {row['psi_q95']}]")
    lines.append("")

    # stats globales
    lines.append("--- Statistiques globales ---")
    lines.append(f"  Provinces totales              : {len(prov)}")
    lines.append(f"  Avec donnee entomo hard        : {(prov['evidence_type'] == 'confirmed_capture').sum()}")
    lines.append(f"  Avec donnee entomo soft        : {(prov['evidence_type'] == 'unverified_capture').sum()}")
    lines.append(f"  Avec cas LCT seulement (epi)   : {(prov['evidence_type'] == 'epi_only').sum()}")
    lines.append(f"  SANS donnee (inference)        : {len(gap)}")
    lines.append(f"  psi_mean min                   : {prov['psi_mean'].min():.3f}")
    lines.append(f"  psi_mean max                   : {prov['psi_mean'].max():.3f}")
    lines.append(f"  psi_mean median                : {prov['psi_mean'].median():.3f}")
    lines.append("")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print(report)

    out_path = config.FIGURES / "results_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nEcrit : {out_path}")


if __name__ == "__main__":
    main()
