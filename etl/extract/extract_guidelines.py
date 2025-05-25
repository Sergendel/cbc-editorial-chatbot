import logging
from pathlib import Path

from config.config import Config
from etl.extract.extract_base import ExtractBase
from etl.extract.scraper import SectionScraper

# Setup Logging
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "extract_guidelines.log"),
        logging.StreamHandler(),
    ],
)


class ExtractGuidelines(ExtractBase):
    """Extracts CBC guidelines HTML content ."""

    def __init__(self, config):
        self.config = config
        self.scraper = SectionScraper(config)

    def extract(self):
        """performs scraping and returns metadata."""
        logging.info("Starting CBC guidelines extraction .")
        try:
            guidelines_dict = self.scraper.scrape_all_sections()

            # Log  how many sections were scraped successfully
            logging.info(
                f"Guidelines extraction completed successfully."
                f" {len(guidelines_dict)} sections extracted."
            )

            # Return structured metadata  (titles and file paths)
            summary = {
                title: str(
                    Path(self.config.extracted_guidelines_folder)
                    / f"{self.format_title_for_url(title)}.html"
                )
                for title in guidelines_dict.keys()
            }

            return summary

        except Exception as e:
            logging.error(f"Extraction failed : {e}")
            return None

    @staticmethod
    def format_title_for_url(title: str) -> str:
        """Formats the title  into a consistent URL-friendly string."""
        return (
            title.lower()
            .replace(" ", "-")
            .replace(",", "")
            .replace(":", "")
            .replace("(", "")
            .replace(")", "")
            .replace("’", "")
        )


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    extractor = ExtractGuidelines(config)
    extraction_summary = extractor.extract()
    print(extraction_summary)
