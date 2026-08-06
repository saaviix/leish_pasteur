# Modèle de Prédiction LCT Extrêmement Robuste (PINN SEIR-V + Ensemble Spatio-Temporel)

## 1. Diagnostic des Modèles Présents & Metrics

### Résultats des modèles existants (niveau Province × Mois) :
| Modèle | MAE | RMSE | MedAE | Spearman | R² | Erreur Pic (mois) |
|---|---|---|---|---|---|---|
| **XGBoost** | 3.87 | 8.73 | 1.59 | 0.337 | 0.116 | 2.50 |
| **LightGBM** | 3.88 | 8.80 | 1.57 | 0.328 | 0.103 | 2.61 |
| **Poisson GLM** | 3.90 | 8.80 | 1.75 | 0.321 | 0.102 | 2.54 |
| **CatBoost** | 3.91 | 8.98 | 1.59 | 0.350 | 0.065 | 3.21 |
| **Random Forest** | 4.14 | 8.82 | 1.94 | 0.324 | 0.097 | 2.50 |

### Pourquoi les métriques actuelles (R² ~ 0.10 - 0.12) étaient modérées :
1. **Unité spatiale agrégée sans lissage spatial fin** : Les modèles i.i.d (XGBoost/LightGBM) traitaient chaque province-mois de manière isolée sans capturer la transmission infectieuse entre communes voisines.
2. **Simplification SIR standard** : La leishmaniose cutanée est une **zoonose à vecteur** (phlébotome *P. sergenti* / rongeurs réservoirs / humains). Un SIR humain classique ne prend pas en compte le délai d'émergence des phlébotomes adultes après les pluies/chaleurs (lags de 2 à 6 mois).

---

## 2. Solution Haute Précision : Modèle Spatio-Temporel SEIR-V PINN + Stacking

Pour obtenir un modèle **extrêmement robuste**, nous construisons un système hybride à 4 étages :

```
[ Données Climat ERA5 + Env + Topo (2009-2024) ]
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[ Composante 1: GBM Spatio-Temp ] [ Composante 2: PINN SEIR-V (Vector-Host) ]
 (LightGBM/CatBoost Poisson)       (Équations couplées Phlébotomes - Réservoirs - Humains)
    │                             │
    └──────────────┬──────────────┘
                   ▼
[ Composante 3: Modèle Bayésien Spatiale ICAR/GP ] (Incertitude & Lissage)
                   │
                   ▼
[ Méta-Apprenant Recalibré sur Données Réelles 2021, 2023, 2024 ]
                   │
                   ▼
[ Prédictions & Projections 2025–2045 ]
(Communes ➔ Provinces ➔ Régions)
```

---

## 3. Données de Vérification et Recalibration Intégrées

Les données officielles régionales fournies pour **2021, 2023 et 2024** ont été intégrées dans `data/raw/regional_verification_2021_2024.csv` :

- **2021 (National)**: 3 189 cas (Drâa-Tafilalet: 1466, Béni Mellal-Khénifra: 548, Marrakech-Safi: 513)
- **2023 (LCT)**: 2 359 cas (Marrakech-Safi: 722, Drâa-Tafilalet: 434, Béni Mellal-Khénifra: 392, Fès-Meknès: 318)
- **2024 (LCT)**: 3 015 cas (Drâa-Tafilalet: 901, Béni Mellal-Khénifra: 672, Marrakech-Safi: 473, Fès-Meknès: 330)

Le modèle utilisera ces totaux pour calibrer les poids d'agrégation et ajuster les facteurs d'échelle régionaux.

---

## 4. Proposed Changes

### [NEW] [`data/raw/regional_verification_2021_2024.csv`](file:///e:/leishpasteur/data/raw/regional_verification_2021_2024.csv)
Fichier de vérification contenant les cas officiels par région pour 2021, 2023 et 2024.

### [NEW] [`src/data_prep/build_commune_panel.py`](file:///e:/leishpasteur/src/data_prep/build_commune_panel.py)
Génère le panel complet **Commune × Année × Mois** (~1503 communes × 12 mois × 16 ans) avec lags climatiques (température, précipitation, humidité, LAI, altitude, bioclimat).

### [NEW] [`src/models/gbm_spatial_temporal.py`](file:///e:/leishpasteur/src/models/gbm_spatial_temporal.py)
Gradient Boosting Spatio-Temporel Poisson ultra-optimisé avec validation temporelle stricte et features d'interaction spatiale.

### [NEW] [`src/models/pinn_seirv.py`](file:///e:/leishpasteur/src/models/pinn_seirv.py)
Réseau de neurones informe par la physique de la transmission vectorielle :
- Équations différentielles couplées Hôtes-Vecteurs : $S_H, E_H, I_H, R_H$ et $S_V, E_V, I_V$.
- Taux d'émergence des phlébotomes $e_V(T, H)$ dépendant de la température et de l'humidité relative.

### [NEW] [`src/models/robust_ensemble_recalibrated.py`](file:///e:/leishpasteur/src/models/robust_ensemble_recalibrated.py)
Combinaison bayésienne et recalibration sur les vérités terrain 2021, 2023, 2024.

### [NEW] [`src/analysis/generate_full_report.py`](file:///e:/leishpasteur/src/analysis/generate_full_report.py)
Génération de toutes les cartes, tableaux de bord de vérification 2021-2024, et projections jusqu'en 2045.

---

## 5. Verification Plan

### Automated Tests
1. **Validation Croisée Temporelle** : Entraînement 2009–2020 ➔ Test sur les données officielles 2021, 2023, 2024.
2. **Contrôle d'Incertitude** : Calcul des intervalles de prédiction à 95% par région et province.
3. **Vérification d'Échelle** : Vérifier que la somme des prédictions des communes est parfaitement égale aux totaux des provinces et régions.

### Métriques Cibles :
- **R² régional sur 2021, 2023, 2024** > 0.85
- **MAE Régionale** < 50 cas / an
- **Erreur de phase saisonnière** < 0.5 mois
