# news_articles_json is loaded from your full file 'news_articles.json'



def generate_news_embeddings(article_json: dict,
                             embedding_model: Callable[[str], List[float]]) -> List[dict]:
    embeddings_with_metadata = []

    # Document-level chunk (full article)
    article_text = article_json["title"] + "\n\n" + article_json["content"]

    embeddings_with_metadata.append({
        "embedding": embedding_model(article_text.strip()),
        "metadata": {
            "chunk_level": "document",
            "type": "news",
            "id": article_json["id"],
            "title": article_json["title"],
            "content_snippet": article_text[:100],
            "publish_time": article_json.get("publish_time"),
            "last_update": article_json.get("last_update"),
            "word_count": article_json.get("word_count"),
            "categories": article_json.get("categories", []),
            "tags": article_json.get("tags", {}),
            "department": article_json.get("department"),
            "url": article_json.get("url")
        }
    })

    # Sentence-level chunks for precise retrieval
    sentences = re.split(r'(?<=[.!?])\s+', article_json["content"])
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if len(clean_sentence) < 20:  # filter very short/noisy sentences
            continue

        embeddings_with_metadata.append({
            "embedding": embedding_model(clean_sentence),
            "metadata": {
                "chunk_level": "sentence",
                "type": "news",
                "id": article_json["id"],
                "title": article_json["title"],
                "content_snippet": clean_sentence[:100],
                "publish_time": article_json.get("publish_time"),
                "last_update": article_json.get("last_update"),
                "word_count": article_json.get("word_count"),
                "categories": article_json.get("categories", []),
                "tags": article_json.get("tags", {}),
                "department": article_json.get("department"),
                "url": article_json.get("url")
            }
        })

    return embeddings_with_metadata



all_embeddings = []

for article_json in news_articles.json:  # each `article_json` is one news article
    embeddings = generate_news_embeddings(article_json, embedding_model)
    all_embeddings.extend(embeddings)

import re
from typing import Callable, List
