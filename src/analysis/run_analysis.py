"""
run_analysis.py
===============
Script maitre qui execute toutes les etapes de la couche d'analyse.
Usage:
    python src/analysis/run_analysis.py
"""

import logging
import sys
from pathlib import Path

# Support both direct execution and package imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_prep import config

from analysis import seasonality, climate_response, projections, spatial, figures

logger = logging.getLogger(__name__)


def _validate_inputs() -> bool:
    required = [
        config.CLIMATE_DB,
        config.LCT_CSV,
        config.COMMUNES_CLIMATE,
        config.POSTERIOR_CSV,
        config.PROVINCE_TABLE,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.error("Missing required input files: %s", missing)
        return False
    logger.info("All required input files are present.")
    return True


def main() -> int:
    config.ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Starting full analysis pipeline...")

    if not _validate_inputs():
        logger.warning("Some inputs are missing; proceeding with graceful degradation.")

    try:
        seasonality.main()
    except Exception as exc:
        logger.error("Seasonality analysis failed: %s", exc)

    try:
        climate_response.main()
    except Exception as exc:
        logger.error("Climate response analysis failed: %s", exc)

    try:
        projections.main()
    except Exception as exc:
        logger.error("Projections failed: %s", exc)

    try:
        spatial.main()
    except Exception as exc:
        logger.error("Spatial analysis failed: %s", exc)

    try:
        figures.generate_all_figures()
    except Exception as exc:
        logger.error("Figure generation failed: %s", exc)

    logger.info("Analysis pipeline finished. Outputs located in %s and %s.",
                config.PROCESSED, config.FIGURES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
