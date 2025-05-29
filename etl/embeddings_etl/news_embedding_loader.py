import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config.config import Config
from etl.embeddings_etl.embedding_loader_base import EmbeddingLoaderBase

# Logging Setup
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / "news_embedding_loader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

DEFAULT_TIMESTAMP = "Unknown timestamp"


class DataHolder:
    def __init__(self, config: Config):
        self.json_file_path = config.processed_news_path
        self.data = self.load_data(self.json_file_path)

    @staticmethod
    def load_data(file_path: str) -> List[Dict[str, Any]]:
        """Load JSON data from file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logger.info(f"Successfully loaded news articles from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load news articles from {file_path}: {e}")
            raise


class NewsArticlesEmbeddingLoader(EmbeddingLoaderBase):
    def __init__(self, config: Config):
        self.batch_size = config.embedding_loader_batch_size
        self.embeddings_with_metadata: List[Dict[str, Any]] = []

    def load_embeddings(self, dataholder: DataHolder) -> List[Dict[str, Any]]:
        logger.info("Starting embedding loading process for news articles...")
        for article in dataholder.data:
            text_to_embed = f"{article['title']}\n\n{article['content']}"
            chunk_metadata = {
                "id": article["id"],
                "title": article["title"],
                "publish_time": article.get("publish_time", DEFAULT_TIMESTAMP),
                "last_update": article.get("last_update", DEFAULT_TIMESTAMP),
                "categories": article.get("categories", []),
                "tags": article.get("tags", {}),
                "department": article.get("department", "Unknown department"),
                "chunk_text": text_to_embed,  # Include the chunk text
            }
            self.embeddings_with_metadata.append(
                {"text": text_to_embed, "metadata": chunk_metadata}
            )

            logger.info(f"Prepared embedding chunk for article: {article['id']}")

        logger.info("Completed embedding loading process.")
        return self.embeddings_with_metadata


if __name__ == "__main__":
    from etl.embeddings_etl.batch_embedder import BatchEmbedder
    from etl.embeddings_etl.embedding_saver import EmbeddingSaver
    from models.embedding_model import embedding_model_function

    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    dataholder = DataHolder(config)
    loader = NewsArticlesEmbeddingLoader(config)
    embedding_chunks = loader.load_embeddings(dataholder)

    batch_embedder = BatchEmbedder(embedding_model_function, loader.batch_size)
    final_embeddings = batch_embedder.embed_chunks(embedding_chunks)

    # Save embeddings
    storage_path = project_root / "data" / "embeddings_storage"
    saver = EmbeddingSaver(storage_path)
    saver.save_embeddings_and_metadata(
        final_embeddings,
        embedding_filename="news_embeddings.npy",
        metadata_filename="news_metadata.json",
    )

    for emb in final_embeddings[:3]:
        logger.info(f"Embedding metadata: {emb['metadata']}")
        logger.info(f"Embedding vector (preview): {emb['embedding'][:5]}...")
