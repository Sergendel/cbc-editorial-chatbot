import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()


def classify_query(query: str):
    classification_llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        task="text-generation",
        temperature=0.01,
        max_new_tokens=5,
    )

    classification_prompt = PromptTemplate(
        template="""
        Classify explicitly the following user query into exactly one category from:
        - Guidelines
        - News
        - Mixed

        Query: "{query}"
        Explicit Category:
        """,
        input_variables=["query"],
    )

    structured_prompt = classification_prompt.format(query=query)
    classification_result = classification_llm.invoke(structured_prompt).strip().lower()

    if "guidelines" in classification_result:
        return "guidelines"
    elif "news" in classification_result:
        return "news"
    else:
        return "mixed"
