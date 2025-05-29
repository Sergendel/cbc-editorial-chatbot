import json
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from config.config import Config
from models.embedding_model import embedding_model_function
from models.generative_model import get_generative_model
from rag.chains.intent_classifier import classify_intent
from rag.utils.prompt_router import get_prompt
from rag.utils.query_metadata_extractor import extract_metadata_from_query

# Conversation memory tracking
conversation_state = {
    "clarification_attempts": 0,
    "last_intent": None,
    "last_query": None,
}


# Embedding model for runtime embedding
class QueryEmbeddings(Embeddings):
    def embed_query(self, text):
        embedding_vector = embedding_model_function(text)
        return np.array(embedding_vector).flatten()

    def embed_documents(self, texts):
        return [np.array(embedding_model_function(text)).flatten() for text in texts]


# Load FAISS indexes once (singleton style)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

embedding_model = QueryEmbeddings()

guidelines_index = FAISS.load_local(
    BASE_DIR / "data/vector_indexes/guidelines_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)

news_index = FAISS.load_local(
    BASE_DIR / "data/vector_indexes/news_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)


# Retrieval based on exact metadata match
def retrieve_doc_by_metadata(vectorstore: object, key: object, value: object) -> object:
    matched_docs = [
        doc
        for doc in vectorstore.docstore._dict.values()
        if doc.metadata.get(key) == value
    ]
    return matched_docs[:1] if matched_docs else []


def retrieve_context_with_fallback(
    index, query, fallback_keywords, initial_k=3, fallback_k=5
):
    initial_docs = index.similarity_search(query, k=initial_k)
    initial_context = "\n".join([doc.page_content for doc in initial_docs])

    # Check if initial retrieval failed or was insufficient
    if (
        "not mentioned" in initial_context.lower()
        or "i don't know" in initial_context.lower()
        or not initial_context.strip()
    ):
        refined_docs = index.similarity_search(fallback_keywords, k=fallback_k)
        refined_context = "\n".join([doc.page_content for doc in refined_docs])
        return refined_context

    return initial_context


def retrieve_relevant_context(query, intent, metadata):
    context = ""

    if intent == "policy_query":
        context = retrieve_context_with_fallback(
            guidelines_index, query, fallback_keywords=" ".join(query.split()[-2:])
        )

    elif intent in ["headline_request", "summary_request", "full_article_request"]:
        context = retrieve_context_with_fallback(
            news_index, query, fallback_keywords=" ".join(query.split()[-2:])
        )

    elif metadata:
        retrieved_docs = []
        if metadata.get("id"):
            retrieved_docs = retrieve_doc_by_metadata(news_index, "id", metadata["id"])
        elif metadata.get("timestamp"):
            retrieved_docs = retrieve_doc_by_metadata(
                guidelines_index, "timestamp", metadata["timestamp"]
            )
        elif metadata.get("url"):
            retrieved_docs = retrieve_doc_by_metadata(
                guidelines_index, "url", metadata["url"]
            )

        if retrieved_docs:
            context = retrieved_docs[0].page_content

    return context


def metadata_exists(key: str, value: str) -> bool:
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))
    metadata_lookup_path = project_root / config.metadata_lookup_path

    with open(metadata_lookup_path, "r", encoding="utf-8") as file:
        metadata_lookup = json.load(file)

    return value in metadata_lookup.get(key, [])


def handle_unknown_intent(user_query, conversation_state):
    if conversation_state["last_intent"] == "unknown":
        conversation_state["clarification_attempts"] += 1
    else:
        conversation_state["clarification_attempts"] = 1

    conversation_state["last_intent"] = "unknown"
    conversation_state["last_query"] = user_query

    if conversation_state["clarification_attempts"] > 2:
        return "I don't know."
    else:
        return (
            "I'm not sure I understand your request clearly."
            " Could you please clarify or rephrase your question?"
        )


def intent_driven_rag_chain(user_query):
    intent = classify_intent(user_query)
    if intent == "unknown":
        return handle_unknown_intent(user_query, conversation_state)

    metadata = extract_metadata_from_query(user_query)

    retrieved_docs, retrieved_context = [], ""

    # Metadata-based retrieval (your robust previous logic)
    if metadata.get("url") and metadata_exists("urls", metadata["url"]):
        retrieved_docs = retrieve_doc_by_metadata(
            guidelines_index, "url", metadata["url"]
        )

    elif metadata.get("id") and metadata_exists("ids", metadata["id"]):
        retrieved_docs = retrieve_doc_by_metadata(news_index, "id", metadata["id"])

    elif metadata.get("timestamp") and metadata_exists(
        "timestamps", metadata["timestamp"]
    ):
        retrieved_docs = retrieve_doc_by_metadata(
            guidelines_index, "timestamp", metadata["timestamp"]
        )

    elif metadata.get("publish_time") and metadata_exists(
        "publish_times", metadata["publish_time"]
    ):
        retrieved_docs = retrieve_doc_by_metadata(
            news_index, "publish_time", metadata["publish_time"]
        )

    elif metadata.get("last_update") and metadata_exists(
        "last_update_times", metadata["last_update"]
    ):
        retrieved_docs = retrieve_doc_by_metadata(
            news_index, "last_update", metadata["last_update"]
        )

    # Fallback if no docs retrieved by metadata
    if not retrieved_docs:
        retrieved_context = retrieve_relevant_context(user_query, intent, metadata)
    else:
        retrieved_context = retrieved_docs[0].page_content

    # Handling if still no context found
    if not retrieved_context.strip():
        return "I don't know."

    # Combine intent and retrieved context for accurate prompt construction
    prompt = get_prompt(intent, retrieved_context, user_query)
    llm = get_generative_model()

    response_raw = llm.invoke(prompt)

    # EHandle both AIMessage (OpenAI) and str (HuggingFace) responses
    if hasattr(response_raw, "content"):
        response = response_raw.content.strip()
    else:
        response = response_raw.strip()

    # Gather source document metadata
    sources = []

    for doc in retrieved_docs:
        metadata = doc.metadata

        # Guidelines documents explicitly have 'url' and 'timestamp'
        if "url" in metadata and "timestamp" in metadata:
            sources.append(
                {
                    "type": "Guidelines",
                    "document_title": metadata.get("document_title", "N/A"),
                    "section_path": " > ".join(metadata.get("section_path", [])),
                    "url": metadata.get("url", "N/A"),
                    "timestamp": metadata.get("timestamp", "N/A"),
                    "content_snippet": metadata.get("content_snippet", "N/A"),
                }
            )

        # News articles explicitly have 'id' and 'title'
        elif "id" in metadata and "title" in metadata:
            sources.append(
                {
                    "type": "News",
                    "id": metadata.get("id", "N/A"),
                    "title": metadata.get("title", "N/A"),
                    "publish_time": metadata.get("publish_time", "N/A"),
                    "last_update": metadata.get("last_update", "N/A"),
                    "categories": ", ".join(metadata.get("categories", [])),
                    "chunk_text_snippet": metadata.get("chunk_text", "")[:100] + "...",
                }
            )

        # Explicit fallback for unexpected structures
        else:
            sources.append({"type": "Unknown", "metadata": metadata})

    return {"response": response, "sources": sources}


if __name__ == "__main__":
    test_queries = [
        "Give me the document with timestamp 2025-05-29T15:25:30.625595",
        "Summarize  https://cbc.radio-canada.ca/en/vision/governance/"
        "journalistic-standards-and-practices/children-and-youth",
        "What’s CBC’s guideline on citing anonymous sources?",
        "Suggest an SEO-optimized headline for this article:1.7346111",
        "Summarize this article for a Twitter post.",
        "Show me the details of article ID 1.6272172",
        "Give me the document with timestamp 2025-05-29T15:24:08.648385",
        "Suggest an SEO-optimized headline for article 1.6272172",
        "Summarize article ID 1.6272172 for Twitter.",
        "Show me the details of article ID 1.6272172",
        "Give me the document with timestamp 2025-05-29T15:25:30.625595",
        "What's the weather today?",
        "Please find latest news about weather and summarize",
    ]

    for query in test_queries:
        response = intent_driven_rag_chain(query)
        print(f"\n📌 Query: {query}\n🚩 Response: {response}\n{'-' * 50}")
