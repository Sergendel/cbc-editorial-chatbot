from pathlib import Path

from embeddings import load_embeddings
from indexing import create_faiss_index

BASE_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    # Explicit guidelines embeddings
    guidelines_embeddings_path = (
        BASE_DIR / "data/embeddings_storage/guidelines_embeddings.npy"
    )
    guidelines_metadata_path = (
        BASE_DIR / "data/embeddings_storage/guidelines_metadata.json"
    )
    guidelines_index_path = BASE_DIR / "data/vector_indexes/guidelines_faiss_index"

    texts, embeddings, metadata = load_embeddings(
        guidelines_embeddings_path, guidelines_metadata_path
    )
    create_faiss_index(texts, embeddings, metadata, guidelines_index_path)

    # Explicit news embeddings
    news_embeddings_path = BASE_DIR / "data/embeddings_storage/news_embeddings.npy"
    news_metadata_path = BASE_DIR / "data/embeddings_storage/news_metadata.json"
    news_index_path = BASE_DIR / "data/vector_indexes/news_faiss_index"

    texts, embeddings, metadata = load_embeddings(
        news_embeddings_path, news_metadata_path
    )
    create_faiss_index(texts, embeddings, metadata, news_index_path)
