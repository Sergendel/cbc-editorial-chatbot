import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class QueryEmbeddings(Embeddings):
    def embed_query(self, text):
        from models.embedding_model import embedding_model_function

        embedding_vector = embedding_model_function(text)
        return np.array(embedding_vector).flatten()

    def embed_documents(self, texts):
        from models.embedding_model import embedding_model_function

        return [np.array(embedding_model_function(text)).flatten() for text in texts]


def load_faiss_retriever(index_name, k=3):
    index_path = BASE_DIR / "data" / "vector_indexes" / index_name
    embedding_model = QueryEmbeddings()
    faiss_index = FAISS.load_local(
        str(index_path), embedding_model, allow_dangerous_deserialization=True
    )
    return faiss_index.as_retriever(search_kwargs={"k": k})


def get_combined_retriever(k=3):
    guidelines_retriever = load_faiss_retriever("guidelines_faiss_index", k)
    news_retriever = load_faiss_retriever("news_faiss_index", k)
    return MergerRetriever(retrievers=[guidelines_retriever, news_retriever])


rag_prompt = PromptTemplate(
    template="""
    You are the CBC Editorial Assistant Chatbot. Answer explicitly the single question below using ONLY the provided context. 

    Follow these rules explicitly:
    - If asked explicitly for an SEO-optimized headline for a CBC news article, provide explicitly a concise, keyword-rich headline ONLY.
    - If asked explicitly about CBC editorial guidelines, provide explicitly a brief, accurate summary ONLY.
    - If context explicitly does NOT provide enough information, explicitly respond ONLY "I don't know".
    - If the query explicitly requests an SEO headline or recent news, prioritize news dataset retrievals explicitly and ignore guidelines retrieval explicitly.  
    - If the query explicitly references CBC guidelines, prioritize guidelines dataset explicitly.
    Context:
    {context}

    Question:
    {question}

    Explicit Answer:""",
    input_variables=["context", "question"],
)


def get_rag_chain(k=3):
    retriever = get_combined_retriever(k)
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="text-generation",
        temperature=0.05,
        max_new_tokens=100,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt},
    )


if __name__ == "__main__":
    rag_chain = get_rag_chain()
    query = "Suggest an SEO-optimized headline explicitly for article 1.6272172"
    result = rag_chain.invoke({"query": query})

    print("🚀 Explicitly Generated Answer:\n", result["result"])
    print("\n📚 Explicitly Retrieved Sources:")
    for doc in result["source_documents"]:
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("-" * 50)

# query = "Suggest an SEO-optimized headline explicitly for article 1.6272172"
# rag_prompt = PromptTemplate(
#     template="""
#     You are the CBC Editorial Assistant Chatbot. Answer explicitly the single question below using ONLY the provided context.
#
#     Follow these rules explicitly:
#     - If asked explicitly for an SEO-optimized headline for a CBC news article, provide explicitly a concise, keyword-rich headline ONLY.
#     - If asked explicitly about CBC editorial guidelines, provide explicitly a brief, accurate summary ONLY.
#     - If context explicitly does NOT provide enough information, explicitly respond ONLY "I don't know".
#     - If the query explicitly requests an SEO headline or recent news, prioritize news dataset retrievals explicitly and ignore guidelines retrieval explicitly.
#     - If the query explicitly references CBC guidelines, prioritize guidelines dataset explicitly.
#     Context:
#     {context}
#
#     Question:
#     {question}
#
#     Explicit Answer:""",
#     input_variables=["context", "question"],
# )
