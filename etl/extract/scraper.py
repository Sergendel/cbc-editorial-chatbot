import logging
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError, sync_playwright

from config.config import Config

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
        self.main_url = config.guidelines_url
        self.browser = browser

    def get_section_urls(self):
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
            urls = [
                f"{self.main_url}/{format_title_for_url(title)}" for title in titles
            ]

            logger.info(f"Extracted URLs: {urls}")
            return urls, titles
        except TimeoutError as e:
            logger.error(f"Timeout when extracting URLs: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Unexpected error during URL extraction: {e}")
            return [], []


class SectionScraper:
    def __init__(self, config):
        self.config = config
        self.extracted_guidelines_folder = Path(config.extracted_guidelines_folder)
        self.extracted_guidelines_folder.mkdir(parents=True, exist_ok=True)

    def scrape_page(self, url, context):
        try:
            page = context.new_page()
            page.goto(url, timeout=60000)
            content = page.content()
            page.close()
            logger.info(f"Successfully scraped page: {url}")
            return content
        except TimeoutError as e:
            logger.error(f"Timeout scraping {url}: {e}")
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        return ""

    def scrape_all_sections(self):
        guidelines_dict = {}
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()

            url_extractor = GetSectionUrls(self.config, browser)
            section_urls, section_titles = url_extractor.get_section_urls()

            for url, title in zip(section_urls, section_titles):
                html_content = self.scrape_page(url, context)
                if html_content:
                    file_path = (
                        self.extracted_guidelines_folder
                        / f"{format_title_for_url(title)}.html"
                    )
                    file_path.write_text(html_content, encoding="utf-8")
                    guidelines_dict[title] = html_content
                else:
                    logger.warning(f"Empty content for URL: {url}")

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
