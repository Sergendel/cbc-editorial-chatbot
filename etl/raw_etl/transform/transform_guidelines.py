import logging
import os
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from config.config import Config
from etl.raw_etl.transform.transform_base import TransformBase

project_root = Path(__file__).parent.parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "transform_guidelines.log"),
        logging.StreamHandler(),
    ],
)


class TransformGuidelines(TransformBase):
    """Transforms raw HTML guideline files into structured JSON."""

    def __init__(self, config):
        """Initialize paths and URLs from configuration."""
        self.processed_guidelines_path = config.processed_guidelines_path
        self.extracted_guidelines_folder = config.extracted_guidelines_folder
        self.guidelines_base_url = config.guidelines_url

    def transform(self):
        """Process all HTML files and save structured JSON."""
        all_files_result = {}
        html_files = Path(self.extracted_guidelines_folder).glob("*.html")

        for html_file in html_files:
            logging.info(f"Starting processing: {html_file.name}")
            try:
                main_page_title, single_file_result = self.transform_single_file(
                    html_file
                )
                if main_page_title:
                    original_url = self.construct_url(html_file.stem)
                    timestamp = self.get_file_timestamp(html_file)
                    all_files_result[main_page_title] = {
                        "metadata": {
                            "original_url": original_url,
                            "timestamp": timestamp,
                        },
                        **single_file_result,
                    }
                    logging.info(f"Successfully processed: {html_file.name}")
                else:
                    logging.warning(f"No title found for file: {html_file.name}")
            except Exception as e:
                logging.error(f"Error processing file {html_file.name}: {str(e)}")
                continue

        # will be performed by dedicated wtl/raw_etl/load/load_guidelines.py
        # try:
        #     with open(
        #         self.processed_guidelines_path, "w", encoding="utf-8"
        #     ) as json_file:
        #         json.dump(all_files_result, json_file, ensure_ascii=False, indent=4)
        #     logging.info("All files processed and JSON output created successfully.")
        # except Exception as e:
        #     logging.error(f"Failed to write JSON file: {e}")

        return all_files_result

    def construct_url(self, filename_stem):
        """Construct URL for a guideline based on filename."""
        return f"{self.guidelines_base_url}/{filename_stem}"

    def get_file_timestamp(self, file_path):
        """Get the file's last modification timestamp as ISO format."""
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).isoformat()

    def transform_single_file(self, input_html_path):
        """Transform a single HTML file into structured JSON."""
        single_file_result = {}
        try:
            with open(input_html_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
        except Exception as e:
            logging.error(f"Failed to parse HTML {input_html_path.name}: {e}")
            return None, {}

        main_page_title = self.find_main_title(soup)
        if not main_page_title:
            main_page_title = input_html_path.stem.replace("-", " ").title()

        panels = soup.select("div.panel.panel-default")
        for panel in panels:
            first_level_title_tag = panel.find("h2", class_="panel-title")
            if not first_level_title_tag:
                continue
            first_level_title = first_level_title_tag.get_text(strip=True)
            panel_body = panel.find("div", class_="panel-body")
            if not panel_body:
                continue

            nested_dict, general_content = {}, ""
            elements = [el for el in panel_body.contents if el.name]
            idx = 0

            while idx < len(elements):
                element = elements[idx]
                if element.name == "h4":
                    current_subsection = element.get_text(strip=True)
                    idx += 1
                    subsection_content = ""
                    while idx < len(elements) and elements[idx].name == "p":
                        subsection_content += " " + elements[idx].get_text(strip=True)
                        idx += 1
                    if idx < len(elements) and elements[idx].name == "ul":
                        ul_elem = elements[idx]
                        sub_nested = {}
                        for li in ul_elem.find_all("li", recursive=False):
                            strong_tag = li.find("strong")
                            if strong_tag:
                                key = strong_tag.get_text(strip=True)
                                strong_tag.decompose()
                                value = " ".join(
                                    p.get_text(strip=True) for p in li.find_all("p")
                                )
                                sub_nested[key] = value.strip()
                        nested_dict[current_subsection] = {
                            "description": subsection_content.strip(),
                            **sub_nested,
                        }
                    else:
                        nested_dict[current_subsection] = subsection_content.strip()
                elif element.name == "p":
                    general_content += " " + element.get_text(strip=True)
                idx += 1

            # Simplified logic (remove redundant "General")
            if nested_dict:
                single_file_result[first_level_title] = nested_dict
            elif general_content.strip():
                single_file_result[first_level_title] = general_content.strip()

        return main_page_title, single_file_result

    @staticmethod
    def find_main_title(soup):
        """Extract the main page title from HTML soup."""
        title_tag = soup.find("title")
        if title_tag and "-" in title_tag.text:
            return title_tag.text.split("-", 1)[0].strip()
        return title_tag.text.strip() if title_tag else None


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    transformer = TransformGuidelines(config)
    transformed_data = transformer.transform()
    print(transformed_data)
