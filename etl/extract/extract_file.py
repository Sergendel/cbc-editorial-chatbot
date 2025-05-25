import json
from pathlib import Path

from config.config import Config
from etl.extract.extract_base import ExtractBase


class ExtractFile(ExtractBase):
    """
    Explicit class for JSON file data extraction.
    """

    def __init__(self, config: Config):
        self.file_path = config.raw_news_path

    def extract(self):
        """
        Extract data from JSON file.
        Returns loaded data or None if errors occur.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data

        except FileNotFoundError as e:
            print(f"Error: File '{self.file_path}' not found. Details: {e}")

        except json.JSONDecodeError as e:
            print(
                f"Error: File '{self.file_path}' is"
                f" corrupted or invalid JSON. Details: {e}"
            )

        except Exception as e:
            print(f"Unexpected error: {e}")

        return None


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    file_extractor = ExtractFile(config)
    extracted_data = file_extractor.extract()
    print(extracted_data)
