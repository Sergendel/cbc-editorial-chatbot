# CBC/rag/chains/intent_classifier.py

from models.generative_model import get_generative_model

#  intent categories
INTENTS = [
    "policy_query",
    "headline_request",
    "summary_request",
    "full_article_request",
    "metadata_specific_request",
    "unknown",
]

#  classification prompt template
INTENT_CLASSIFICATION_PROMPT = """
Classify the user's intent . Choose ONLY one of the following intents:

- policy_query: The user  asks about CBC editorial policy or guidelines.
- headline_request: The user  asks for a headline or SEO-optimized title.
- summary_request: The user  requests a summary
 or brief overview of an article.
- full_article_request: The user  requests detailed information, content,
 or full details from an article or document.
- unknown: The user's intent is unclear or
 does not match any  defined intents.

User Query:
"{user_query}"

Intent:
"""


#  Intent classification function
def classify_intent(user_query: str) -> str:
    llm = get_generative_model()

    prompt = INTENT_CLASSIFICATION_PROMPT.format(user_query=user_query)
    response = llm.invoke(prompt)

    # AIMessage (OpenAI) and str (HuggingFace)
    if hasattr(response, "content"):
        intent = response.content.strip().lower()
    else:
        intent = response.strip().lower()

    if intent in INTENTS:
        return intent
    else:
        return "unknown"


#  test to verify intent classification
if __name__ == "__main__":
    test_queries = [
        "What's CBC's guideline on citing anonymous sources?",
        "Suggest an SEO-optimized headline for article 1.6272172",
        "Summarize this article for a Twitter post.",
        "Show me the details of article ID 1.6272172",
        "Give me the document with timestamp 2025-05-27T11:16:14",
        "What's the weather today?",
    ]

    for query in test_queries:
        intent = classify_intent(query)
        print(f"Query: '{query}' \n→  classified as: {intent}\n")
