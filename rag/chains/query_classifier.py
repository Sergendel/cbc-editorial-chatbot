from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

from models.generative_model import get_generative_model

load_dotenv()


def classify_query(query: str):
    classification_llm = get_generative_model()

    classification_prompt = PromptTemplate(
        template="""
        Classify the following user query into exactly one category from:
        - Guidelines
        - News
        - Mixed

        Query: "{query}"
        Category:
        """,
        input_variables=["query"],
    )

    structured_prompt = classification_prompt.format(query=query)
    classification_result = (
        classification_llm.invoke(structured_prompt).content.strip().lower()
    )

    if "guidelines" in classification_result:
        return "guidelines"
    elif "news" in classification_result:
        return "news"
    else:
        return "mixed"


if __name__ == "__main__":
    test_queries = [
        "What does CBC say about journalistic accuracy?",
        "Show me recent CBC news about politics.",
        "Give me an SEO headline for a climate change article.",
    ]

    for query in test_queries:
        category = classify_query(query)
        print(f"Query: '{query}'\nClassified as: '{category}'\n")
