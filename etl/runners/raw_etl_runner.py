"""
Orchestrates the raw ETL pipeline explicitly:
1. Extracts raw data (guidelines and news).
2. Transforms raw data to processed structured  data.
3. Loads processed data into JSON files .
"""

import logging
import subprocess
from pathlib import Path

# Explicit Logging setup
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "raw_etl_runner.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def run_script(script_path):
    full_script_path = project_root / script_path
    script_dir = full_script_path.parent
    try:
        logger.info(f"Running: {script_path}")
        subprocess.run(["python", full_script_path.name], cwd=script_dir, check=True)
        logger.info(f"Completed successfully: {script_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Explicit failure in {script_path}: {e}")
        raise


def main():
    scripts = [
        # etl news
        "etl/raw_etl/extract/extract_raw_news.py",
        "etl/raw_etl/transform/transform_news.py",
        "etl/raw_etl/load/load_news_json.py",
        # etl  guidelines
        "etl/raw_etl/extract/extract_raw_guidelines.py",
        "etl/raw_etl/transform/transform_guidelines.py",
        "etl/raw_etl/load/load_guidelines_json.py",
    ]

    for script in scripts:
        run_script(script)


if __name__ == "__main__":
    main()
