import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Explicit embedding wrapper (confirmed working from retrieval_demo.py)
class QueryEmbeddings(Embeddings):
    def embed_query(self, text):
        from models.embedding_model import embedding_model_function

        embedding_vector = embedding_model_function(text)
        return np.array(embedding_vector).flatten()

    def embed_documents(self, texts):
        from models.embedding_model import embedding_model_function

        return [np.array(embedding_model_function(text)).flatten() for text in texts]


# Explicitly confirmed retriever function (from retrieval_demo.py)
def get_retriever(index_name: str, k: int = 3):
    embedding_model = QueryEmbeddings()
    index_path = BASE_DIR / f"data/vector_indexes/{index_name}"
    faiss_index = FAISS.load_local(
        str(index_path), embedding_model, allow_dangerous_deserialization=True
    )
    return faiss_index.as_retriever(search_kwargs={"k": k})


# Explicitly defined RAG chain integrating the confirmed retriever and generative model
def get_rag_chain(index_name="guidelines_faiss_index", k=3):
    retriever = get_retriever(index_name, k)
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    return rag_chain


# Explicit test of the full RAG pipeline
if __name__ == "__main__":
    rag_chain = get_rag_chain()
    query = "What does CBC say about privacy?"
    result = rag_chain.invoke({"query": query})

    print("🚀 Explicitly Generated Answer:\n", result["result"])
    print("\n📚 Explicitly Retrieved Sources:")
    for doc in result["source_documents"]:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)
