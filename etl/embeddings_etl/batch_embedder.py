# batch_embedder.py

import logging
import time
from typing import Any, Callable, Dict, List

from openai import RateLimitError

logger = logging.getLogger(__name__)


class BatchEmbedder:
    def __init__(
        self, embedding_model: Callable[[List[str]], List[List[float]]], batch_size: int
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def embed_chunks(
        self, embeddings_with_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        texts = [item["text"] for item in embeddings_with_metadata]
        metadata_list = [item["metadata"] for item in embeddings_with_metadata]
        embeddings = []

        logger.info(f"Starting batch embedding of {len(texts)} chunks...")

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_metadata = metadata_list[i : i + self.batch_size]
            logger.info(
                f"Embedding batch {i // self.batch_size + 1}/"
                f"{(len(texts) - 1) // self.batch_size + 1}"
            )

            while True:
                try:
                    batch_embeddings = self.embedding_model(batch_texts)
                    logger.info("Batch embedding succeeded.")
                    break
                except RateLimitError:
                    logger.warning("Rate limit reached, sleeping for 60 seconds...")
                    time.sleep(60)

            embeddings.extend(
                [
                    {"embedding": emb, "metadata": meta}
                    for emb, meta in zip(batch_embeddings, batch_metadata)
                ]
            )

        logger.info("Completed batch embedding.")
        return embeddings
