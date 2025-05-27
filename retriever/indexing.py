from pathlib import Path

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS

from embeddings import load_embeddings


class PrecomputedEmbeddings(Embeddings):
    def __init__(self, embeddings_array):
        self.embeddings = embeddings_array

    def embed_documents(self, texts):
        raise NotImplementedError("Embeddings precomputed.")

    def embed_query(self, text):
        raise NotImplementedError("Queries handled separately.")


def create_faiss_index(embeddings, metadata, save_path):
    precomputed_embeddings = PrecomputedEmbeddings(embeddings)
    faiss_index = FAISS.from_embeddings(
        text_embeddings=embeddings,
        embedding=precomputed_embeddings,
        metadatas=metadata,  # pass metadata directly
    )
    faiss_index.save_local(save_path)
    print(f"FAISS index saved at {save_path}.")


#  define project base directory
BASE_DIR = (
    Path(__file__).resolve().parent.parent
)  # Adjust if indexing.py is nested deeper

if __name__ == "__main__":
    # Explicit Guidelines embeddings paths
    guidelines_embeddings_path = (
        BASE_DIR / "data/embeddings_storage/guidelines_embeddings.npy"
    )
    guidelines_metadata_path = (
        BASE_DIR / "data/embeddings_storage/guidelines_metadata.json"
    )
    guidelines_index_path = BASE_DIR / "data/vector_indexes/guidelines_faiss_index"

    embeddings, metadata = load_embeddings(
        guidelines_embeddings_path, guidelines_metadata_path
    )
    create_faiss_index(embeddings, metadata, guidelines_index_path)

    # Explicit News embeddings paths
    news_embeddings_path = BASE_DIR / "data/embeddings_storage/news_embeddings.npy"
    news_metadata_path = BASE_DIR / "data/embeddings_storage/news_metadata.json"
    news_index_path = BASE_DIR / "data/vector_indexes/news_faiss_index"

    embeddings, metadata = load_embeddings(news_embeddings_path, news_metadata_path)
    create_faiss_index(embeddings, metadata, news_index_path)
