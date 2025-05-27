import logging
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError, sync_playwright

from config.config import Config

logger = logging.getLogger(__name__)

# Explicit special-case mapping for problematic URLs
SPECIAL_URL_MAPPING = {
    "war-terror-and-natural-disasters": "war-terror-natural-disasters",
    "user-generated-content-ugc": "user-generated-content",
}

OPINION_URL = (
    "https://cbc.radio-canada.ca/en/vision/governance/"
    "journalistic-standards-and-practices/opinion"
)
LANGUAGE_URL = (
    "https://cbc.radio-canada.ca/en/vision/governance/"
    "journalistic-standards-and-practices/language"
)


def format_title_for_url(title: str) -> str:
    """Formats a section title into a URL-friendly string, handling special cases."""
    formatted = (
        title.lower()
        .replace(" ", "-")
        .replace(",", "")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .replace("’", "")
    )
    return SPECIAL_URL_MAPPING.get(formatted, formatted)


class GetSectionUrls:
    """Extracts URLs and titles for each guideline
    section from the main guidelines page."""

    def __init__(self, config: Config, browser):
        """Initialize with configuration and browser instance."""
        self.main_url = config.guidelines_url
        self.browser = browser

    def get_section_title(self):
        """Extracts and returns a list of URLs and corresponding section titles."""
        try:
            page = self.browser.new_page()
            page.goto(self.main_url, timeout=60000)
            page.wait_for_selector(
                "td.policy-title-container h3.policy-title", timeout=30000
            )
            soup = BeautifulSoup(page.content(), "html.parser")
            page.close()

            titles = [
                el.get_text(strip=True)
                for el in soup.select("td.policy-title-container h3.policy-title")
            ]

            logger.info(f"Extracted titles: {titles}")
            return titles
        except TimeoutError as e:
            logger.error(f"Timeout when extracting titles: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Unexpected error during title extraction: {e}")
            return [], []


class SectionScraper:
    """Scrapes and saves the HTML content of each guideline section."""

    def __init__(self, config: Config):
        """Initialize scraper with configuration."""
        self.config = config
        self.extracted_guidelines_folder = Path(config.extracted_guidelines_folder)
        self.extracted_guidelines_folder.mkdir(parents=True, exist_ok=True)

    def scrape_page(self, title: str, context) -> str:
        """Scrapes and returns HTML content, explicitly handling special cases."""
        try:
            page = context.new_page()
            logger.info(f"Scraping page with title  {title} by simulated click.")
            page.goto(self.config.guidelines_url, timeout=60000)
            page.click(f'text="{title}"', force=True)
            page.wait_for_load_state("networkidle")

            content = page.content()
            page.close()
            logger.info(f"Successfully scraped page: {title}")
            return content

        except TimeoutError as e:
            logger.error(f"Timeout scraping {title}: {e}")
        except Exception as e:
            logger.error(f"Error scraping {title}: {e}")
        return ""

    def scrape_all_sections(self) -> dict:
        """Scrapes HTML content from all guideline sections and
        returns a dictionary of titles and content."""
        guidelines_dict = {}
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()

            url_extractor = GetSectionUrls(self.config, browser)
            section_titles = url_extractor.get_section_title()

            for title in section_titles:
                html_content = self.scrape_page(title, context)
                if html_content:
                    file_path = (
                        self.extracted_guidelines_folder
                        / f"{format_title_for_url(title)}.html"
                    )
                    file_path.write_text(html_content, encoding="utf-8")
                    guidelines_dict[title] = html_content
                else:
                    logger.warning(f"Empty content for URL: {title}")

            context.close()
            browser.close()
        return guidelines_dict


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    section_scraper = SectionScraper(config)
    guidelines_dict = section_scraper.scrape_all_sections()
    print(guidelines_dict)
