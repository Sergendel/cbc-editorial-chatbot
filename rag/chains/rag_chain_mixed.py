import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpoint

from rag.chains.query_classifier import classify_query  # Explicitly import classifier

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


def get_retrievers():
    embedding_model = QueryEmbeddings()

    guidelines_index = FAISS.load_local(
        BASE_DIR / "data/vector_indexes/guidelines_faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True,
    ).as_retriever()

    news_index = FAISS.load_local(
        BASE_DIR / "data/vector_indexes/news_faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True,
    ).as_retriever()

    return guidelines_index, news_index


rag_prompt = PromptTemplate(
    template="""
    You are the CBC Editorial Assistant Chatbot. Answer explicitly the question below using ONLY the provided context.

    Explicit Rules:
    - SEO-optimized headlines → concise, keyword-rich headline ONLY.
    - CBC guidelines queries → brief accurate summary ONLY.
    - Not enough context → explicitly "I don't know".

    Context:
    {context}

    Question:
    {question}

    Explicit Answer:""",
    input_variables=["context", "question"],
)


def get_rag_chain(selected_retriever):
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="text-generation",
        temperature=0.05,
        max_new_tokens=150,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=selected_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": rag_prompt},
    )


def explicit_rag(query: str):
    guidelines_retriever, news_retriever = get_retrievers()

    # Explicitly classify query first
    query_category = classify_query(query)
    print(f"🚩 Query classified explicitly as: {query_category}")

    # Explicit dataset selection
    if query_category == "guidelines":
        retriever = guidelines_retriever
        print(f"🚩 guidelines_retriever is used")
    elif query_category == "news":
        retriever = news_retriever
        print(f"🚩 news_retriever is used")
    else:  # Mixed
        retriever = guidelines_retriever.merge(news_retriever)
        print(f"🚩 mixed_retriever is used")

    rag_chain = get_rag_chain(retriever)

    response = rag_chain.invoke(query)

    result = {
        "query": response["query"],
        "classification": query_category,
        "result": response["result"],
        "source_documents": response["source_documents"],
    }

    return result


if __name__ == "__main__":
    test_queries = [
        'Retrieve explicitly news from "British Columbia"',
        # "Suggest an SEO-optimized headline explicitly for article 1.6272172",
        # "What are CBC's editorial guidelines about language?",
        # "What recent CBC news articles are there about heat waves?",
        # "Suggest an SEO headline based on CBC guidelines for news article on extreme heat."
    ]

    for query in test_queries:
        print(f"\n📌 Query: {query}")
        result = explicit_rag(query)
        print(f"📚 result:\n{result}")
