import json
import logging
from pathlib import Path

from config.config import Config

# logging
project_root = Path(__file__).parent.parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "load_guidelines_json.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class LoadGuidelinesJson:
    def __init__(self, config: Config):
        self.processed_guidelines_path = config.processed_guidelines_path

    def load(self, structured_data):
        try:
            # Ensure directory exists
            self.processed_guidelines_path.parent.mkdir(parents=True, exist_ok=True)

            with open(
                self.processed_guidelines_path, "w", encoding="utf-8"
            ) as json_file:
                json.dump(structured_data, json_file, ensure_ascii=False, indent=4)
            logger.info(
                f"Guidelines explicitly saved to {self.processed_guidelines_path}"
            )
        except Exception as e:
            logger.error(f"Failed explicitly to save guidelines JSON: {e}")
            raise


if __name__ == "__main__":
    from etl.raw_etl.transform.transform_guidelines import TransformGuidelines

    project_root = Path(__file__).parent.parent.parent.parent.resolve()

    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    # run
    transformer = TransformGuidelines(config)
    transformer.transform()
    transformed_data = transformer.transform()

    # save to json
    loader = LoadGuidelinesJson(config)
    loader.load(transformed_data)
