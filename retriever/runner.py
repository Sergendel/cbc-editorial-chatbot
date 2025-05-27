from indexing import create_faiss_index

from embeddings import load_embeddings

if __name__ == "__main__":
    # Explicit guidelines embeddings
    embeddings, metadata = load_embeddings(
        "data/embeddings_storage/guidelines_embeddings.npy",
        "data/embeddings_storage/guidelines_metadata.json",
    )
    create_faiss_index(
        embeddings, metadata, "data/vector_indexes/guidelines_faiss_index"
    )

    # Explicit news embeddings
    embeddings, metadata = load_embeddings(
        "data/embeddings_storage/news_embeddings.npy",
        "data/embeddings_storage/news_metadata.json",
    )
    create_faiss_index(embeddings, metadata, "data/vector_indexes/news_faiss_index")
