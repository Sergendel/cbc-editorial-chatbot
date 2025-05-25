import json
import logging
import re
from datetime import datetime
from pathlib import Path

from config.config import Config
from etl.transform.transform_base import TransformBase

# Setup logging
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "transform_news.log"),
        logging.StreamHandler(),
    ],
)


class TransformNews(TransformBase):
    """Transforms raw news JSON into optimized structured JSON for embedding."""

    def __init__(self, config):
        """Initialize raw and processed news data paths."""
        self.raw_news_path = config.raw_news_path
        self.processed_news_path = config.processed_news_path

    def transform(self):
        """Execute the full transformation on news dataset."""
        try:
            with open(self.raw_news_path, "r", encoding="utf-8") as file:
                raw_news_data = json.load(file)
        except Exception as e:
            logging.error(f"Failed to load raw news data: {e}")
            return []

        processed_news = []
        for article in raw_news_data:
            processed_article = self.transform_single_article(article)
            if processed_article:
                processed_news.append(processed_article)

        try:
            with open(self.processed_news_path, "w", encoding="utf-8") as file:
                json.dump(processed_news, file, ensure_ascii=False, indent=4)
            logging.info("News dataset transformation completed successfully.")
        except Exception as e:
            logging.error(f"Failed to write processed news data: {e}")

        return processed_news

    def transform_single_article(self, article):
        """Clean and structure individual article fields ."""
        try:
            article_id = (article.get("content_id") or "").strip()
            title = (article.get("content_headline") or "").strip()
            content = self.clean_text(article.get("body") or "")
            publish_time = self.normalize_datetime(article.get("content_publish_time"))
            last_update = self.normalize_datetime(article.get("content_last_update"))
            word_count = int(article.get("content_word_count") or 0)
            department = (article.get("content_department_path") or "").strip()

            categories = [
                c.get("content_category", "").strip()
                for c in article.get("content_categories", [])
                if c.get("content_category")
            ]

            tags = {}
            for tag in article.get("content_tags", []):
                tag_type = (tag.get("type") or "generic").strip()
                tag_name = (tag.get("name") or "").strip()
                if not tag_name:
                    continue
                if tag_type not in tags:
                    tags[tag_type] = []
                tags[tag_type].append(tag_name)

            if not (article_id and title and content):
                logging.warning(f"Incomplete data for article ID {article_id}")
                return None

            return {
                "id": article_id,
                "title": title,
                "content": content,
                "publish_time": publish_time,
                "last_update": last_update,
                "word_count": word_count,
                "categories": categories,
                "tags": tags,
                "department": department,
            }
        except Exception as e:
            logging.error(
                f"Error transforming article ID "
                f"{article.get('content_id', 'unknown')}: {e}"
            )
            return None

    def clean_text(self, text):
        """Clean article text ."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize_datetime(self, date_str):
        """Normalize date to ISO 8601 format ."""
        try:
            return datetime.fromisoformat(date_str).isoformat()
        except (ValueError, TypeError) as e:
            logging.warning(f"Date normalization error '{date_str}': {e}")
            return "Unknown"


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    transformer = TransformNews(config)
    transformer.transform()
