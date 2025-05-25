from pathlib import Path

from config.config import Config
from etl.extract.extract_base import ExtractBase
from etl.extract.scraper import SectionScraper


class ExtractGuidelines(ExtractBase):
    def __init__(self, config):
        self.guidelines_url = config.guidelines_url
        self.section_scrapper = SectionScraper(config)

    def extract(self):
        return self.section_scrapper.scrape_all_sections()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    guidelines_extractor = ExtractGuidelines(config)
    extracted_data = guidelines_extractor.extract()
    print(extracted_data)
