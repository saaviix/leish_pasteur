"""
run_pipeline.py
===============
Lance tout le pipeline de bout en bout, dans l'ordre. Chaque etape est
optionnelle et sautee proprement si ses dependances/donnees manquent.

Etapes :
  1. validate_inputs.py           (verification coherence des donnees brutes)
  2. communes_by_region.py        (requete communes par region/province)
  3. clean_lct.py                 (nettoyage cas + rapport donnees manquantes)
  4. download_era5.py             (telechargement ERA5 depuis le CDS) [reseau, cle API]
  5. extract_climate.py           (ERA5 -> climate_morocco.db + covariables) [lourd]
  6. build_environment.py         (altitude/sol/vegetation/aridite -> environment_morocco.db)
  7. fetch_geojson.py             (polygones communes -> geojson)
  8. build_province_table.py      (table province + climat + voisinage ICAR)
  9. bayesian_occupancy.py        (inference bayesienne de la presence)
  10. run_analysis.py             (graphes, correlations, projections, patterns)
  11. bioclimatic_zoning.py + epi_map.py  (zonage bioclimatique/epi + carte)
  11b. elevation_classification.py + elevation_case_drivers.py +
       elevation_stratified_models.py (classes d'altitude, facteurs
       climat/env par classe, modeles GBM separes par classe)
  12. summarize_results.py        (resume lisible des resultats)
  13. dashboard : a lancer separement

Etapes optionnelles hors orchestrateur (exploratoires/lourdes, a lancer a la
main quand besoin) :
  - src/models/occupancy_gp.py     (variante GP spatial de l'occupancy model)
  - src/models/model_benchmark.py  (comparatif etendu de modeles de prediction)

Usage :
  python run_pipeline.py                    # base (sans download/climat/env/analysis)
  python run_pipeline.py --download         # + telechargement ERA5
  python run_pipeline.py --with-climate     # + extraction climat
  python run_pipeline.py --with-env         # + environnement
  python run_pipeline.py --with-analysis    # + graphes/correlations/projections
  python run_pipeline.py --with-zoning      # + zonage bioclimatique + carte epi
  python run_pipeline.py --all              # tout
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, label: str) -> bool:
    path = ROOT / script
    print("\n" + "=" * 70)
    print(f">>> {label}")
    print(f"    {script}")
    print("=" * 70)
    if not path.exists():
        print(f"[SKIP] script introuvable : {path}")
        return False
    try:
        subprocess.run([PY, str(path)], check=True, cwd=ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] {script} a echoue (code {e.returncode}). On continue.")
        return False


def main() -> None:
    args = set(sys.argv[1:])
    do_all = "--all" in args
    do_download = do_all or "--download" in args
    with_climate = do_all or "--with-climate" in args
    with_env = do_all or "--with-env" in args
    with_scraping = do_all or "--with-scraping" in args
    with_analysis = do_all or "--with-analysis" in args
    with_zoning = do_all or "--with-zoning" in args

    # etapes de base
    run("src/data_prep/validate_inputs.py", "Validation des entrees")
    run("src/data_prep/communes_by_region.py", "Requete communes par region/province")
    run("src/data_prep/clean_lct.py", "Nettoyage LCT + rapport donnees manquantes")
    run("src/data_prep/fix_mun_pop.py", "Conversion/reconciliation population municipale (xlsx -> csv)")

    if do_download:
        run("src/data_prep/download_era5.py", "Telechargement ERA5 depuis le CDS")
    else:
        print("\n[INFO] telechargement ERA5 saute (--download)")

    if with_climate:
        run("src/data_prep/extract_climate.py", "Extraction climat ERA5")
    else:
        print("[INFO] extraction climat sautee (--with-climate)")

    if with_env:
        run("src/data_prep/build_environment.py", "Attributs environnementaux")
    else:
        print("[INFO] environnement saute (--with-env)")

    run("src/data_prep/fetch_geojson.py", "GeoJSON communes")

    if with_scraping:
        run("src/scraping/main.py", "Scraping articles P. sergenti")
    else:
        print("[INFO] scraping saute (--with-scraping)")

    # coeur : table province + modele bayesien
    run("src/data_prep/build_province_table.py", "Table province + voisinage ICAR + climat")
    run("src/models/bayesian_occupancy.py", "Inference bayesienne de la presence")

    with_predict = do_all or "--predict" in args

    if with_analysis:
        run("src/analysis/run_analysis.py", "Graphes, correlations, projections, patterns")
    else:
        print("[INFO] analyse sautee (--with-analysis)")

    if with_zoning:
        run("src/models/bioclimatic_zoning.py", "Zonage bioclimatique/epidemiologique (K-means)")
        run("src/analysis/epi_map.py", "Carte epidemiologique (zones + charge de cas)")
        run("src/analysis/elevation_classification.py", "Classification par altitude (K-means 1D, 4 classes)")
    else:
        print("[INFO] zonage bioclimatique saute (--with-zoning)")

    if with_predict:
        run("src/data_prep/build_commune_panel.py", "Construction du panel Commune x Mois")
        # PINN AVANT le GBM : gbm_pinn_stacked.py depend de
        # pinn_seirv_weights.pt (features mecanistes E_H/I_H/C_vraie pour le
        # correcteur de residu GBM_2).
        run("src/models/pinn_seirv.py", "Modèle PINN SEIR-V (Vector-Host Physics)")
        # Modele officiel : stacking residuel GBM_1 + GBM_2(PINN), R2=0.5448
        # (vs 0.5312 pour un GBM seul). gbm_spatial_temporal.py reste
        # disponible comme outil de comparaison/diagnostic (candidats
        # LightGBM/ExtraTrees/RandomForest/HistGB) mais n'est plus l'etape
        # officielle du pipeline.
        run("src/models/gbm_pinn_stacked.py", "Modèle officiel : GBM + correcteur de résidu PINN (stacking)")
        run("src/models/robust_ensemble_recalibrated.py", "Recalibration & Stacking sur Données Réelles 2021-2024")
        run("src/analysis/forecast_future.py", "Projections 20 ans (2025-2045) par Commune, Province, Région")
        run("src/analysis/generate_full_report.py", "Rapport final d'étalonnage et de vérification")

        if with_zoning:
            # a besoin du panel + de la classification altitude, donc placee ici
            run("src/analysis/elevation_case_drivers.py", "Facteurs climat/env par classe d'altitude (correlations)")
            run("src/models/elevation_stratified_models.py", "Modeles GBM separes par classe d'altitude")

    # depend du modele bayesien (occupancy_trace.nc) et, si dispo, du GBM
    run("src/analysis/underreporting.py", "Estimation de la sous-declaration (capture-recapture + occupancy)")

    run("src/models/summarize_results.py", "Resume des resultats")

    print("\n" + "=" * 70)
    print("PIPELINE TERMINE.")
    print("Pour lancer le tableau de bord :  python src/interface/dashboard.py")
    print("Pour lancer la prédiction seule :  python run_pipeline.py --predict")
    print("Pour voir le resume :              cat outputs/figures/results_summary.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
