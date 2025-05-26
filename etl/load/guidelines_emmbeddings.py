import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

from config.config import Config
from etl.load.load_base import EmbeddingLoader
from models.embedding_model import embedding_model_function

logger = logging.getLogger(__name__)


class GuidelineEmbeddingLoader(EmbeddingLoader):
    def __init__(
        self, config: Config, embedding_model: Callable[[List[str]], List[List[float]]]
    ):
        self.embedding_model = embedding_model
        self.batch_size = self.batch_size = config.embedding_loader_batch_size

        try:
            with open(config.processed_guidelines_path, "r", encoding="utf-8") as file:
                self.data = json.load(file)
            logger.info(f"Loaded guidelines from {config.processed_guidelines_path}")
        except Exception as e:
            logger.error(f"Failed to load guidelines: {e}")
            raise

    def generate_embeddings(self) -> List[Dict[str, Any]]:
        texts_to_embed, metadata_list = [], []

        for document_title, content in self.data.items():
            document_metadata = content.get("metadata", {})

            self._add_document_embedding(
                document_title,
                content,
                document_metadata,
                texts_to_embed,
                metadata_list,
            )
            self._add_section_embeddings(
                document_title,
                content,
                document_metadata,
                texts_to_embed,
                metadata_list,
            )
            self._add_paragraph_embeddings(
                document_title,
                content,
                document_metadata,
                texts_to_embed,
                metadata_list,
            )

        return self._batch_embed(texts_to_embed, metadata_list)

    def _add_document_embedding(
        self,
        document_title: str,
        content: dict,
        metadata: dict,
        texts: List[str],
        meta: List[dict],
    ):
        document_text = self._combine_all_sections(content)
        texts.append(document_text.strip())
        meta.append(
            self._create_metadata(
                "document", document_title, None, document_text, metadata
            )
        )
        logger.info(f"Added document embedding: '{document_title}'")

    def _add_section_embeddings(
        self,
        document_title: str,
        content: dict,
        metadata: dict,
        texts: List[str],
        meta: List[dict],
    ):
        for section_title, section_content in content.items():
            if section_title == "metadata":
                continue
            texts.append(section_content.strip())
            meta.append(
                self._create_metadata(
                    "section", document_title, section_title, section_content, metadata
                )
            )
        logger.info(f"Added section embeddings for document: '{document_title}'")

    def _add_paragraph_embeddings(
        self,
        document_title: str,
        content: dict,
        metadata: dict,
        texts: List[str],
        meta: List[dict],
    ):
        paragraph_count = 0
        for section_title, section_content in content.items():
            if section_title == "metadata":
                continue
            paragraphs = [p.strip() for p in section_content.split("\n\n") if p.strip()]
            for para in paragraphs:
                texts.append(para)
                meta.append(
                    self._create_metadata(
                        "paragraph", document_title, section_title, para, metadata
                    )
                )
                paragraph_count += 1
        logger.info(
            f"Added {paragraph_count}"
            f" paragraph embeddings for document: '{document_title}'"
        )

    def _create_metadata(
        self,
        chunk_level: str,
        document_title: str,
        section_title: str,
        content: str,
        document_metadata: dict,
    ) -> dict:
        return {
            "chunk_level": chunk_level,
            "type": "guideline",
            "document_title": document_title,
            "section_title": section_title,
            "subsection_title": None,
            "content_snippet": content[:100],
            "url": document_metadata.get("original_url"),
            "timestamp": document_metadata.get("timestamp"),
        }

    def _combine_all_sections(self, document_content: dict) -> str:
        combined_text = ""
        for title, content in document_content.items():
            if title != "metadata":
                combined_text += f"{title}\n\n{content}\n\n"
        return combined_text

    def _batch_embed(self, texts: List[str], metadata: List[dict]) -> List[dict]:
        embeddings_with_metadata = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.info(f"Total texts to embed: {len(texts)} in {total_batches} batches")

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_metadata = metadata[i : i + self.batch_size]
            current_batch = (i // self.batch_size) + 1
            logger.info(
                f"Embedding batch {current_batch}/"
                f"{total_batches}: {len(batch_texts)} items"
            )

            try:
                batch_embeddings = self.embedding_model(batch_texts)
                embeddings_with_metadata.extend(
                    [
                        {"embedding": emb, "metadata": meta}
                        for emb, meta in zip(batch_embeddings, batch_metadata)
                    ]
                )
            except Exception as e:
                logger.error(f"Embedding failed for batch {current_batch}: {e}")
                raise

        logger.info(f"Generated total embeddings: {len(embeddings_with_metadata)}")
        return embeddings_with_metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    loader = GuidelineEmbeddingLoader(config, embedding_model_function)
    embeddings = loader.generate_embeddings()
    print(embeddings)
