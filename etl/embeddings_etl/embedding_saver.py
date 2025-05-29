import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingSaver:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"EmbeddingSaver initialized at {self.storage_path}")

    def save_embeddings_and_metadata(
        self,
        embeddings_with_metadata: List[Dict[str, Any]],
        embedding_filename: str,
        metadata_filename: str,
    ) -> None:
        """
        Saves embeddings as NumPy array and metadata as JSON file.

        Args:
            embeddings_with_metadata: List of dicts with 'embedding' and 'metadata'.
            embedding_filename: Filename for saving embeddings (.npy).
            metadata_filename: Filename for saving metadata (.json).
        """
        logger.info("Separating embeddings and metadata...")
        embeddings = [item["embedding"] for item in embeddings_with_metadata]
        metadata = [item["metadata"] for item in embeddings_with_metadata]

        # Save embeddings
        embeddings_array = np.array(embeddings)
        embeddings_path = self.storage_path / embedding_filename
        np.save(embeddings_path, embeddings_array)
        logger.info(f"Embeddings saved to {embeddings_path}")

        # Save metadata
        metadata_path = self.storage_path / metadata_filename
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    # from etl.load.embedding_saver import EmbeddingSaver
    from config.config import Config
    from etl.embeddings_etl.batch_embedder import BatchEmbedder
    from etl.embeddings_etl.guidelines_embedding_loader import (
        DataHolder,
        EmbeddingLoader,
    )
    from models.embedding_model import embedding_model_function

    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    dataholder = DataHolder(config)
    loader = EmbeddingLoader(config)
    embedding_chunks = loader.load_embeddings(dataholder)

    batch_embedder = BatchEmbedder(embedding_model_function, loader.batch_size)
    final_embeddings = batch_embedder.embed_chunks(embedding_chunks)

    # Saving embeddings and metadata
    storage_path = project_root / "data" / "embeddings_storage"
    saver = EmbeddingSaver(storage_path)

    saver.save_embeddings_and_metadata(
        final_embeddings,
        embedding_filename="guidelines_embeddings.npy",
        metadata_filename="guidelines_metadata.json",
    )
    saver.save_embeddings_and_metadata(
        final_embeddings,
        embedding_filename="news_embeddings.npy",
        metadata_filename="news_metadata.json",
    )
