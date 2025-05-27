# load_news_json.py

import json
import logging
from pathlib import Path

from config.config import Config

# Explicit Logging setup
project_root = Path(__file__).parent.parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "load_news.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class LoadNewsJson:
    """Explicitly loads structured news articles data into a JSON file."""

    def __init__(self, config: Config):
        self.processed_news_path = config.processed_news_path

    def load(self, structured_data):
        try:
            # Ensure directory exists
            self.processed_news_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.processed_news_path, "w", encoding="utf-8") as json_file:
                json.dump(structured_data, json_file, ensure_ascii=False, indent=4)
            logger.info(
                f"Successfully loaded news data into {self.processed_news_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load news data into JSON: {e}")
            raise


if __name__ == "__main__":
    from etl.raw_etl.transform.transform_news import TransformNews

    project_root = Path(__file__).parent.parent.parent.parent.resolve()

    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    # run
    transformer = TransformNews(config)
    transformer.transform()
    transformed_data = transformer.transform()

    # save to json
    loader = LoadNewsJson(config)
    loader.load(transformed_data)
