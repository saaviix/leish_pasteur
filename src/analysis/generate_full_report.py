"""
generate_full_report.py
========================
Rapport de synthèse final : vraies métriques de vérification 2021/2023/2024
(robust_ensemble_recalibrated.py) + comparatif de modèles (model_benchmark.py)
+ projections 2025 (forecast_future.py), quand ces fichiers existent.

Réécrit en Phase 3 de la refonte -- la version précédente se contentait de
fusionner les vérités terrain 2021/2023/2024 avec une projection 2025 (deux
années différentes, pas une comparaison), sans jamais charger ni calculer de
métrique d'erreur. Le vrai calcul de métriques vit désormais dans
robust_ensemble_recalibrated.py (verification_metrics_2021_2024.csv) ; ce
script se contente de les assembler dans un rapport lisible, sans rien
recalculer ni fabriquer.

Sorties :
  - outputs/processed/final_verification_report_2021_2024.csv
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_prep"))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_report():
    metrics_path = config.PROCESSED / "verification_metrics_2021_2024.csv"
    if not metrics_path.exists():
        logger.error(f"{metrics_path} introuvable. Lance d'abord : "
                      f"python src/models/robust_ensemble_recalibrated.py")
        return None
    metrics = pd.read_csv(metrics_path)

    print("\n" + "=" * 90)
    print("      RAPPORT DE VÉRIFICATION RÉELLE 2021 / 2023 / 2024 (niveau régional)")
    print("=" * 90)
    print(metrics.round(3).to_string(index=False))
    print("=" * 90)

    n_real = int((metrics["source_climat"] == "mesure_era5").sum())
    n_extra = int((metrics["source_climat"] == "climatologie_extrapolee").sum())
    print(f"\n{n_real} année(s) sur climat ERA5 mesuré, {n_extra} sur climatologie extrapolée "
          f"(pas de mesure ERA5 au-delà de 2021 sur cette machine).")

    benchmark_path = config.PROCESSED / "metrics_comparatif_etendu.csv"
    if benchmark_path.exists():
        bench = pd.read_csv(benchmark_path)
        print("\n" + "-" * 90)
        print("Comparatif de modèles (test 2018-2020, commune x mois) -- model_benchmark.py")
        print("-" * 90)
        print(bench.round(3).to_string(index=False))

    out_path = config.PROCESSED / "final_verification_report_2021_2024.csv"
    metrics.to_csv(out_path, index=False)
    logger.info(f"\nRapport écrit : {out_path}")
    return metrics


if __name__ == "__main__":
    generate_report()
