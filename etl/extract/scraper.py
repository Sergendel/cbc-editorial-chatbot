import logging
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_title_for_url(title: str) -> str:
    return (
        title.lower()
        .replace(" ", "-")
        .replace(",", "")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .replace("’", "")
    )


class GetSectionUrls:
    def __init__(self, config, browser):
        if not config or not hasattr(config, "guidelines_url"):
            raise ValueError("Invalid configuration provided.")
        self.main_url = config.guidelines_url
        self.browser = browser

    def get_section_urls(self):
        page = self.browser.new_page()
        page.goto(self.main_url, timeout=60000)
        # Explicitly wait for the titles to load
        page.wait_for_selector(
            "td.policy-title-container h3.policy-title", timeout=30000
        )
        html = page.content()
        page.close()

        # Explicitly parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Explicitly extract section titles
        title_elements = soup.select("td.policy-title-container h3.policy-title")
        section_titles = [element.get_text(strip=True) for element in title_elements]

        logger.info(f"Extracted section titles: {section_titles}")

        # Create URLs
        section_urls = [
            f"{self.main_url}/{format_title_for_url(title)}" for title in section_titles
        ]

        # matching between url and titles
        seen_urls = set()
        unique_urls, unique_titles = [], []

        for url, title in zip(section_urls, section_titles):
            if url not in seen_urls:
                unique_urls.append(url)
                unique_titles.append(title)
                seen_urls.add(url)
        section_urls, section_titles = unique_urls, unique_titles

        return section_urls, section_titles


class SectionScraper:
    def __init__(self, config):
        self.guidelines_dict = {}
        self.config = config
        self.raw_guidelines_folder = Path(config.raw_guidelines_folder)

        # Ensure folder exists
        self.raw_guidelines_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def scrape_page(url, context):
        try:
            page = context.new_page()
            page.goto(url, timeout=60000)
            html = page.content()
            page.close()
            return html
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""

    def scrape_all_sections(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()

            url_extractor = GetSectionUrls(self.config, browser)
            section_urls, section_titles = url_extractor.get_section_urls()

            for url, title in zip(section_urls, section_titles):
                html_content = self.scrape_page(url, context)

                # Save to file
                file_path = (
                    self.raw_guidelines_folder / f"{format_title_for_url(title)}.html"
                )
                with file_path.open("w", encoding="utf-8") as f:
                    f.write(html_content)

                self.guidelines_dict[title] = html_content

            context.close()
            browser.close()
        return self.guidelines_dict


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    section_scraper = SectionScraper(config)
    guidelines_dict = section_scraper.scrape_all_sections()
    print(guidelines_dict)
