from pathlib import Path

import numpy as np
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

from models.embedding_model import embedding_model_function

BASE_DIR = Path(__file__).resolve().parent.parent


class QueryEmbeddings(Embeddings):
    def embed_query(self, text):
        return embedding_model_function(text)

    def embed_documents(self, texts):
        # Explicit placeholder implementation (not used explicitly here)
        return [embedding_model_function(text) for text in texts]


def demo_retrieval(index_name: str, query: str, k: int = 3):
    embedding_model = QueryEmbeddings()
    embedding_vector = embedding_model.embed_query(query)
    embedding_vector = np.array(embedding_vector).flatten()

    index_path = BASE_DIR / f"data/vector_indexes/{index_name}"

    faiss_index = FAISS.load_local(
        str(index_path), embedding_model, allow_dangerous_deserialization=True
    )

    retrieved_docs = faiss_index.similarity_search_by_vector(embedding_vector, k=k)

    print(f"\nTop {k} retrieval results for query: '{query}'\n")
    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"Result {idx}:")
        print(f"Chunk Text: {doc.page_content}\n")
        print(f"Metadata: {doc.metadata}\n")
        print("-" * 50)


if __name__ == "__main__":
    demo_retrieval(
        index_name="guidelines_faiss_index",
        query="What does CBC say about privacy?",
        k=3,
    )

    demo_retrieval(
        index_name="news_faiss_index", query="Recent CBC news about food banks", k=3
    )
