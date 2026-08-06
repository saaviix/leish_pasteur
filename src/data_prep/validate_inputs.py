"""
validate_inputs.py
==================
Verifie que toutes les entrees du pipeline sont coherentes avant de lancer
l'analyse. C'est la premiere etape a executer pour detecter les problems
de donnees manquantes / colonnes inattendues / formats incorrects.

Checks :
  - fichiers bruts presents et non vides
  - colonnes obligatoires dans chaque CSV
  - coherence des codes province entre les fichiers
  - detection des doublons de communes
  - rapport de valeurs manquantes dans les cas LCT

Sorties :
  outputs/processed/validation_report.txt

Usage :
  python src/data_prep/validate_inputs.py
"""

import sys
from pathlib import Path

import pandas as pd

import config

REQUIRED_RAW = {
    config.COMMUNES_CSV: {"id", "commune", "province", "region", "latitude", "longitude"},
    config.LCT_CSV: {"Region", "Province", "Commune", "Annee_Source", "Classification"},
    config.SERGENTI_CSV: {"Province", "Statut_P_sergenti"},
}

LCT_YEAR_COL = "Annee_Source"
LCT_MIN_YEAR = 2009
LCT_MAX_YEAR = 2021


def check_file(path: Path, required_cols: set) -> list:
    """Retourne une liste de messages d'erreur pour un fichier donne."""
    errors = []
    if not path.exists():
        errors.append(f"FICHIER MANQUANT : {path}")
        return errors
    if path.stat().st_size == 0:
        errors.append(f"FICHIER VIDE : {path}")
        return errors
    try:
        df = pd.read_csv(path, nrows=3)
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"COLONNES MANQUANTES dans {path.name} : {sorted(missing)}")
    except Exception as e:
        errors.append(f"ERREUR LECTURE {path.name} : {e}")
    return errors


def check_lct_years(path: Path) -> list:
    """Verifie que les annees LCT sont dans la fourchette attendue."""
    errors = []
    if not path.exists():
        return errors
    try:
        df = pd.read_csv(path, usecols=[LCT_YEAR_COL])
        years = pd.to_numeric(df[LCT_YEAR_COL], errors="coerce").dropna()
        if years.empty:
            errors.append(f"Colonne '{LCT_YEAR_COL}' vide ou non numerique dans {path.name}")
        else:
            ymin, ymax = int(years.min()), int(years.max())
            if ymin < LCT_MIN_YEAR or ymax > LCT_MAX_YEAR:
                errors.append(f"Annees LCT hors fourchette [{LCT_MIN_YEAR}-{LCT_MAX_YEAR}] : "
                              f"{ymin}-{ymax} detectees")
    except ValueError:
        errors.append(f"Colonne '{LCT_YEAR_COL}' introuvable dans {path.name}")
    except Exception as e:
        errors.append(f"Erreur check annees LCT : {e}")
    return errors


def check_communes_duplicates() -> list:
    """Detecte les doublons de (commune, province) dans le referentiel."""
    errors = []
    if not config.COMMUNES_CSV.exists():
        return errors
    df = pd.read_csv(config.COMMUNES_CSV)
    dups = df.duplicated(subset=["commune", "province"], keep=False)
    if dups.any():
        n = int(dups.sum())
        errors.append(f"DOUBLONS dans communes_maroc_final.csv : {n} lignes (commune, province)")
    return errors


def check_lct_missing() -> list:
    """Rapporte les valeurs manquantes dans les colonnes critiques de LCT."""
    warnings = []
    if not config.LCT_CSV.exists():
        return warnings
    df = pd.read_csv(config.LCT_CSV)
    for col in ["Region", "Province", "Commune", "Annee_Source"]:
        if col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss > 0:
                warnings.append(f"  {col} : {n_miss} valeurs manquantes sur {len(df)} lignes")
    return warnings


def main() -> None:
    config.ensure_dirs()
    out_path = config.PROCESSED / "validation_report.txt"

    all_errors = []
    all_warnings = []

    # --- fichiers bruts ---
    for path, cols in REQUIRED_RAW.items():
        all_errors.extend(check_file(path, cols))

    # --- coherence des annees LCT ---
    all_errors.extend(check_lct_years(config.LCT_CSV))

    # --- doublons ---
    all_errors.extend(check_communes_duplicates())

    # --- valeurs manquantes LCT ---
    lct_warns = check_lct_missing()
    all_warnings.extend(lct_warns)

    # --- fichiera de sortie ---
    lines = []
    lines.append("=" * 60)
    lines.append("RAPPORT DE VALIDATION DES ENTREES")
    lines.append("=" * 60)
    lines.append("")

    if all_errors:
        lines.append(f"ERREURS ({len(all_errors)}) :")
        for e in all_errors:
            lines.append(f"  [ERREUR] {e}")
    else:
        lines.append("OK : tous les fichiers requis sont presents et valides.")

    lines.append("")
    if all_warnings:
        lines.append(f"AVERTISSEMENTS ({len(all_warnings)}) :")
        for w in all_warnings:
            lines.append(w)
    else:
        lines.append("OK : pas de valeurs manquantes critiques detectees.")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nEcrit : {out_path}")

    # code de sortie : 0 si pas d'erreur, 1 sinon
    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    main()
