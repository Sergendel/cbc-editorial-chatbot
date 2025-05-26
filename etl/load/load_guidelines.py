import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List

from openai import RateLimitError

from config.config import Config
from models.embedding_model import embedding_model_function

# Setup logging
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

log_file_path = log_dir / "embedding_loader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),  # to continue logging to console as well
    ],
)

logger = logging.getLogger(__name__)


# Constants for chunk levels
CHUNK_LEVEL_PARAGRAPH = "paragraph"
CHUNK_LEVEL_PREFIX = "level_"
DEFAULT_URL = "Unknown URL"
DEFAULT_TIMESTAMP = "Unknown timestamp"


class EmbeddingLoaderBase(ABC):
    @abstractmethod
    def load_embeddings(self, dataholder: "DataHolder") -> List[Dict[str, Any]]:
        """Abstract method to load embeddings from a DataHolder."""
        pass


class DataHolder:
    def __init__(self, config: Config):
        self.json_file_path = config.processed_guidelines_path
        self.data = self.load_data(self.json_file_path)

    @staticmethod
    def load_data(file_path: str) -> Dict[str, Any]:
        """Loads JSON data from a given file path."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logger.info(f"Successfully loaded guidelines from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load guidelines from {file_path}: {e}")
            raise


class EmbeddingLoader(EmbeddingLoaderBase):
    def __init__(
        self, config: Config, embedding_model: Callable[[List[str]], List[List[float]]]
    ):
        self.embedding_model = embedding_model
        self.batch_size = config.embedding_loader_batch_size
        self.embeddings_with_metadata: List[Dict[str, Any]] = []

    def load_embeddings(self, dataholder: DataHolder) -> List[Dict[str, Any]]:
        """Traverses data and prepares chunks with metadata for embedding."""
        logger.info("Starting embedding loading process...")
        for document_title, document_content in dataholder.data.items():
            logger.info(f"Processing document: {document_title}")
            document_metadata = document_content.get("metadata", {})
            self._dfs(document_content, [document_title], document_metadata)
        logger.info("Completed embedding loading process.")
        return self.embeddings_with_metadata

    def _dfs(
        self, current_data: Any, path: List[str], parent_metadata: Dict[str, Any]
    ) -> str:
        """Recursive DFS traversal to generate embedding chunks
         with metadata inheritance."""
        current_metadata = parent_metadata.copy()
        if isinstance(current_data, dict):
            node_metadata = current_data.get("metadata", {})
            current_metadata.update(node_metadata)

            texts = []
            for key, value in current_data.items():
                if key == "metadata":
                    continue
                texts.append(self._dfs(value, path + [key], current_metadata))

            aggregated_text = "\n\n".join(filter(None, texts)).strip()
            if aggregated_text:
                chunk_level = f"{CHUNK_LEVEL_PREFIX}{len(path)}"
                self._create_chunk(aggregated_text, path, current_metadata, chunk_level)
            return aggregated_text

        elif isinstance(current_data, str):
            paragraphs = [p.strip() for p in current_data.split("\n\n") if p.strip()]
            for para in paragraphs:
                self._create_chunk(para, path, current_metadata, CHUNK_LEVEL_PARAGRAPH)
            return current_data.strip()

    def _create_chunk(
        self, text: str, path: List[str], metadata: Dict[str, Any], chunk_level: str
    ) -> None:
        """Creates and stores an embedding chunk with associated metadata."""
        chunk_metadata = {
            "chunk_level": chunk_level,
            "document_title": path[0],
            "section_path": path,
            "content_snippet": text[:100],
            "url": metadata.get("original_url", DEFAULT_URL),
            "timestamp": metadata.get("timestamp", DEFAULT_TIMESTAMP),
        }
        logger.info(f"Created chunk at {' > '.join(path)} [{chunk_level}]")
        self.embeddings_with_metadata.append({"text": text, "metadata": chunk_metadata})


class BatchEmbedder:
    def __init__(
        self, embedding_model: Callable[[List[str]], List[List[float]]], batch_size: int
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def embed_chunks(
        self, embeddings_with_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Embeds text chunks in batches and attaches embeddings to metadata."""
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


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    dataholder = DataHolder(config)
    loader = EmbeddingLoader(config, embedding_model_function)

    embeddings_with_metadata = loader.load_embeddings(dataholder)

    batch_embedder = BatchEmbedder(
        embedding_model_function, config.embedding_loader_batch_size
    )
    final_embeddings = batch_embedder.embed_chunks(embeddings_with_metadata)

    for emb in final_embeddings[:3]:
        logger.info(f"Embedding metadata: {emb['metadata']}")
        logger.info(f"Embedding vector (preview): {emb['embedding'][:5]}...")
