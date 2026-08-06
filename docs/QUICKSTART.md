# GUIDE RAPIDE - PROJET LEISHMANIOSE MAROC

## Ce que tu as maintenant

Projet end-to-end complet et robuste pour :
- Cartographier la presence de P. sergenti au Maroc
- Inferer les provinces sans donnee (modele bayesien)
- Analyser les correlations climatiques
- Projeter l'evolution future (2030-2100, scenarios SSP)
- Visualiser sur une carte interactive

## Lancer le projet

### 1. Nettoyage (une fois)
Double-clic sur `CLEANUP.bat` a la racine.

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Verification des donnees
```bash
python src/data_prep/validate_inputs.py
```

### 4. Pipeline complet
```bash
# Tout (telechargement ERA5 + analyse + dashboard)
python run_pipeline.py --all

# Sans telechargement (si .nc deja present)
python run_pipeline.py --with-climate --with-env --with-analysis

# Juste le coeur (donnees deja pretes)
python src/models/bayesian_occupancy.py
python src/analysis/run_analysis.py
python src/interface/dashboard.py
```

## Structure du projet

```
E:\leishpasteur
|-- README.md                    <- documentation complete
|-- requirements.txt             <- pandas, pymc, xarray, flask, cdsapi, etc.
|-- run_pipeline.py              <- orchestrateur (lance tout dans l'ordre)
|-- CLEANUP.bat                  <- nettoie les anciens dossiers
|
|-- data/
|   |-- raw/                     <- tes donnees brutes
|   |   |-- communes_maroc_final.csv          (1503 communes)
|   |   |-- leish_LCT.csv                      (cas LCT 2009-2021)
|   |   |-- phlebotomus_sergenti_par_province.csv  (statuts entomo)
|   |   |-- era5_morocco_2009_2021_hourly.nc   (ERA5 horaire, ignore par git)
|   |   |-- era5_env_static.nc                 (sol/vegetation/altitude)
|   |   \-- *.nc                                (autres fichiers climat)
|   |-- processed/               <- outputs intermediaires
|   \-- external/                <- PDFs articles (120+ articles)
|
|-- src/
|   |-- data_prep/               <- 9 scripts de preparation
|   |   |-- config.py            <- chemins centralises
|   |   |-- validate_inputs.py   <- verifie coherence des donnees
|   |   |-- communes_by_region.py <- REQUETE : toutes les communes par region
|   |   |-- clean_lct.py         <- nettoyage + rapport donnees manquantes
|   |   |-- download_era5.py     <- telechargement ERA5 depuis CDS
|   |   |-- extract_climate.py   <- NetCDF -> SQLite daily + covariables
|   |   |-- build_environment.py <- altitude/sol/vegetation/aridite
|   |   |-- fetch_geojson.py     <- polygones pour la carte
|   |   \-- build_province_table.py <- table province + voisinage ICAR
|   |
|   |-- scraping/                <- scraper P. sergenti
|   |   |-- main.py              <- orchestration du scraping
|   |   |-- matching.py          <- reperage mentions + communes
|   |   |-- text_extraction.py   <- extraction PDF/URL
|   |   \-- urls.txt             <- URLs a scraper
|   |
|   |-- models/                  <- modele bayesien
|   |   |-- bayesian_occupancy.py <- inference presence (y compris gap)
|   |   \-- summarize_results.py  <- resume lisible des resultats
|   |
|   |-- analysis/                <- couche d'analyse complete
|   |   |-- seasonality.py       <- saisonnalite LCT + climat
|   |   |-- climate_response.py  <- correlations climat/presence
|   |   |-- projections.py       <- projections 2030-2100 (SSP126/SSP585)
|   |   |-- spatial.py           <- clusters spatiaux, Moran's I
|   |   |-- figures.py           <- generateur de figures publication
|   |   \-- run_analysis.py      <- lance toute l'analyse
|   |
|   \-- interface/               <- dashboard interactif
|       \-- dashboard.py         <- Flask + Leaflet + Chart.js
|
|-- outputs/
|   |-- processed/               <- outputs intermediaires
|   |   |-- communes_by_region.csv          <- ta requete (R>P>C)
|   |   |-- communes_by_region.json         <- hierarchie JSON
|   |   |-- lct_clean.csv                   <- cas LCT nettoyes
|   |   |-- lct_missing_report.csv          <- provinces sans donnee
|   |   |-- validation_report.txt           <- check coherence
|   |   |-- province_table.csv              <- table par province + climat
|   |   |-- adjacency_edges.npy             <- graphe voisinage ICAR
|   |   |-- communes_climate.csv            <- covariables par commune
|   |   |-- climate_morocco.db              <- SQLite daily (dashboard)
|   |   |-- environment_morocco.db          <- altitude/sol/vegetation
|   |   |-- communes_morocco.geojson        <- polygones pour carte
|   |   |-- sergenti_par_commune_scraped.csv <- scraping
|   |   |-- spatial_clusters.csv            <- clusters spatiaux
|   |   \-- future_projections.csv          <- projections 2030-2100
|   |
|   |-- posterior/               <- resultats du modele
|   |   |-- psergenti_posterior_presence.csv <- psi par province
|   |   \-- occupancy_trace.nc               <- trace MCMC complete
|   |
|   \-- figures/                 <- graphes publication-ready
|       |-- results_summary.txt   <- resume textuel
|       |-- seasonality_*.png     <- saisonnalite
|       |-- climate_response_*.png <- correlations
|       |-- projections_*.png     <- projections futures
|       |-- spatial_*.png         <- clusters spatiaux
|       \-- model_diagnostics_*.png <- diagnostics bayesiens
|
\-- docs/                        <- documentation technique
    |-- schema.dot               <- schema pipeline
    \-- modeles_notes.md         <- notes methodologiques
```

## Les 12 etapes du pipeline

| # | Script | Ce qu'il fait | Duree |
|---|--------|---------------|-------|
| 1 | validate_inputs.py | Verifie fichiers, colonnes, doublons | 5s |
| 2 | communes_by_region.py | Requete R>P>C (CSV + JSON) | 10s |
| 3 | clean_lct.py | Nettoie cas + rapport manquants | 30s |
| 4 | download_era5.py | Telecharge ERA5 depuis CDS (horaire 2009-2021) | 10-60min |
| 5 | extract_climate.py | NetCDF -> SQLite daily + covariables mensuelles | 5-20min |
| 6 | build_environment.py | Altitude/sol/vegetation + aridite | 2-10min |
| 7 | fetch_geojson.py | Polygones pour carte | 30s |
| 8 | build_province_table.py | Table province + climat + voisinage ICAR | 30s |
| 9 | bayesian_occupancy.py | Inference bayesienne (provinces gap incluses) | 5-30min |
| 10 | run_analysis.py | Graphes, correlations, projections, patterns | 2-5min |
| 11 | summarize_results.py | Resume lisible des resultats | 5s |
| 12 | dashboard.py | Carte interactive + couche risque | continu |

## Commandes rapides

```bash
# Setup initial
pip install -r requirements.txt

# Verification
python src/data_prep/validate_inputs.py

# Pipeline complet
python run_pipeline.py --all

# Sans telechargement ERA5
python run_pipeline.py --with-climate --with-env --with-analysis

# Juste le modele
python src/models/bayesian_occupancy.py

# Analyse complete
python src/analysis/run_analysis.py

# Dashboard
python src/interface/dashboard.py
# -> http://localhost:5050
```

## Ce qui est inclus

### 1. Collecte et preprocessing
- Download ERA5 (horaire 2009-2021) depuis CDS Copernicus
- Extraction quotidienne par commune (moyenne 24h)
- Covariables mensuelles long-terme pour le modele
- Altitude/sol/vegetation depuis ERA5-Land statique
- Nettoyage et standardisation des cas LCT
- Rapport automatique des donnees manquantes

### 2. Scraping P. sergenti
- Extraction de texte depuis PDFs et URLs
- Recherche de mentions de P. sergenti
- Matching avec les 1503 communes du referentiel
- Une ligne par commune (mention trouvee ou non)
- Anti-hallucination : extraits litteraux, pas d'invention

### 3. Modele mathematique
- Modele d'occupation bayesien (MacKenzie 2002)
- Lissage spatial ICAR (graphe de Delaunay, pruning >300km)
- Covariables : latitude + temperature + precipitations + aridite
- Evidence epidemiologique (cas LCT) + entomologique (captures)
- Inference des provinces sans donnee (gap)
- Diagnostics : r_hat, divergences, trace MCMC

### 4. Analyse et figures
- Saisonnalite : incidence LCT par mois, patterns climatiques
- Climate response : correlations psi ~ temp/precip/aridity
- Projections futures : SSP126 (+1.5C) vs SSP585 (+4C) pour 2030-2050-2070-2100
- Spatial : clusters chauds/froids, Moran's I local
- Model diagnostics : trace plots, posterior distributions
- Province ranking : top 10 risque, gap provinces

### 5. Cartographie
- Dashboard interactif Flask + Leaflet
- Carte choroplethe par commune
- Variables : climat (temp, precip, humidite, vent, rayonnement)
- Couche Risque : probabilite P. sergenti issue du modele bayesien
- Filtres : region / province / commune / annee / mois
- Series temporelles : temperature et precipitations annuelles
- Profil mensuel : temperature + humidite
- KPI : temp moy, humidite, precip, vent pour la commune selectionnee
- Environnement : altitude, sol, vegetation, aridite

## Notes importantes

- Les fichiers volumineux (.nc, .db, .pdf) sont ignores par git
- Les anciens dossiers sont dans _archive/ (lancer CLEANUP.bat pour les deplacer)
- Le scraping peut prendre du temps selon le nombre de PDFs et URLs
- Le modele bayesien prend 5-30min selon la taille du graphe
- Les projections sont approximatives (logit-link, pas re-execution complete du modele)

## References methodologiques

- MacKenzie et al. (2002) - Occupancy models
- Rue & Martino (2009) - ICAR models
- Royle & Dorazio (2008) - Hierarchical occupancy models
- IPCC SSP scenarios for climate projections
- ERA5 reanalysis (Copernicus CDS)
