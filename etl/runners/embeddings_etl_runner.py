# embeddings_etl_runner.py
"""
embeddings_etl_runner.py  orchestrates embedding generation and saving
for both editorial guidelines and news articles, ensuring modularity,
explicitness, and adherence to SOLID principles.
"""

import logging
from pathlib import Path

from config.config import Config

# Common embedding tools
from etl.embeddings_etl.batch_embedder import BatchEmbedder
from etl.embeddings_etl.embedding_saver import EmbeddingSaver

# Import for guidelines loader
from etl.embeddings_etl.guidelines_embedding_loader import (
    DataHolder as GuidelinesDataHolder,
)
from etl.embeddings_etl.guidelines_embedding_loader import (
    EmbeddingLoader as GuidelinesEmbeddingLoader,
)

# Import for news articles loader
from etl.embeddings_etl.news_embedding_loader import DataHolder as NewsDataHolder
from etl.embeddings_etl.news_embedding_loader import NewsArticlesEmbeddingLoader
from models.embedding_model import embedding_model_function

# Setup explicit logging
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / "etl_runner.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def run_guidelines_etl(config: Config, storage_path: Path):
    logger.info("Starting ETL for Editorial Guidelines...")
    dataholder = GuidelinesDataHolder(config)
    loader = GuidelinesEmbeddingLoader(config)
    embedding_chunks = loader.load_embeddings(dataholder)

    batch_embedder = BatchEmbedder(embedding_model_function, loader.batch_size)
    final_embeddings = batch_embedder.embed_chunks(embedding_chunks)

    saver = EmbeddingSaver(storage_path)
    saver.save_embeddings_and_metadata(
        final_embeddings,
        embedding_filename="guidelines_embeddings.npy",
        metadata_filename="guidelines_metadata.json",
    )

    logger.info("ETL for Editorial Guidelines completed successfully.")


def run_news_articles_etl(config: Config, storage_path: Path):
    logger.info("Starting ETL for News Articles...")
    dataholder = NewsDataHolder(config)
    loader = NewsArticlesEmbeddingLoader(config)
    embedding_chunks = loader.load_embeddings(dataholder)

    batch_embedder = BatchEmbedder(embedding_model_function, loader.batch_size)
    final_embeddings = batch_embedder.embed_chunks(embedding_chunks)

    saver = EmbeddingSaver(storage_path)
    saver.save_embeddings_and_metadata(
        final_embeddings,
        embedding_filename="news_embeddings.npy",
        metadata_filename="news_metadata.json",
    )

    logger.info("ETL for News Articles completed successfully.")


def main():
    try:
        # config path
        project_root = Path(__file__).parent.parent.parent.resolve()
        config_path = project_root / "config" / "config.yml"
        config = Config(str(config_path))

        # embedding storage path
        storage_path = project_root / "data" / "embeddings_storage"

        #  run ETL for guidelines
        run_guidelines_etl(config, storage_path)

        #  run ETL for news articles
        run_news_articles_etl(config, storage_path)

    except Exception as e:
        logger.error(f"ETL Runner  failed due to: {e}")
        raise


if __name__ == "__main__":
    main()
