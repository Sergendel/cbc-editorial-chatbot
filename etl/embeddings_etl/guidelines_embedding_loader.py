import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config.config import Config
from etl.embeddings_etl.batch_embedder import BatchEmbedder
from etl.embeddings_etl.embedding_loader_base import EmbeddingLoaderBase
from models.embedding_model import embedding_model_function

# Logging setup
project_root = Path(__file__).parent.parent.parent.resolve()
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / "guidelines_embedding_loader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Constants
CHUNK_LEVEL_PARAGRAPH = "paragraph"
CHUNK_LEVEL_PREFIX = "level_"
DEFAULT_URL = "Unknown URL"
DEFAULT_TIMESTAMP = "Unknown timestamp"


class DataHolder:
    def __init__(self, config: Config):
        self.json_file_path = config.processed_guidelines_path
        self.data = self.load_data(self.json_file_path)

    @staticmethod
    def load_data(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            logger.info(f"Successfully loaded guidelines from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load guidelines from {file_path}: {e}")
            raise


class EmbeddingLoader(EmbeddingLoaderBase):
    def __init__(self, config: Config):
        self.batch_size = config.embedding_loader_batch_size
        self.embeddings_with_metadata: List[Dict[str, Any]] = []

    def load_embeddings(self, dataholder: DataHolder) -> List[Dict[str, Any]]:
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
        chunk_metadata = {
            "chunk_level": chunk_level,
            "document_title": path[0],
            "section_path": path,
            "content_snippet": text[:100],
            "url": metadata.get("original_url", DEFAULT_URL),
            "timestamp": metadata.get("timestamp", DEFAULT_TIMESTAMP),
            "chunk_text": text,  # Include the full chunk text
        }
        logger.info(f"Created chunk at {' > '.join(path)} [{chunk_level}]")
        self.embeddings_with_metadata.append({"text": text, "metadata": chunk_metadata})


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    dataholder = DataHolder(config)
    loader = EmbeddingLoader(config)

    embeddings_with_metadata = loader.load_embeddings(dataholder)

    batch_embedder = BatchEmbedder(
        embedding_model_function, config.embedding_loader_batch_size
    )
    final_embeddings = batch_embedder.embed_chunks(embeddings_with_metadata)

    for emb in final_embeddings[:3]:
        logger.info(f"Embedding metadata: {emb['metadata']}")
        logger.info(f"Embedding vector (preview): {emb['embedding'][:5]}...")
