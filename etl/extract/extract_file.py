import json
import logging
from pathlib import Path

from config.config import Config
from etl.extract.extract_base import ExtractBase

# Logging Setup
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "extract_file.log"),
        logging.StreamHandler(),
    ],
)


class ExtractFile(ExtractBase):
    """
    Class designed for JSON file extraction.
    """

    def __init__(self, config: Config):
        self.file_path = config.raw_news_path
        self.extracted_news_path = config.extracted_news_path

        # Ensure the extracted data directory exists
        Path(self.extracted_news_path).parent.mkdir(parents=True, exist_ok=True)

    def extract(self):
        """
        Extract data from JSON file .
        Returns loaded data or None if errors occur.
        """
        logging.info(f"Starting extraction from '{self.file_path}'.")

        try:
            # Load raw data
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logging.info(f"Successfully loaded raw data with {len(data)} items.")

            # Save  extracted data
            with open(self.extracted_news_path, "w", encoding="utf-8") as out_file:
                json.dump(data, out_file, ensure_ascii=False, indent=4)
            logging.info(f"Extracted data  saved to '{self.extracted_news_path}'.")

            return data

        except FileNotFoundError as e:
            logging.error(f"File '{self.file_path}' not found: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"Corrupted or invalid JSON in '{self.file_path}': {e}")
        except Exception as e:
            logging.error(f"Unexpected error during extraction: {e}")

        return None


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    file_extractor = ExtractFile(config)
    extracted_data = file_extractor.extract()
    print(extracted_data)
