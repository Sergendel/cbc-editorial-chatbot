import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# Explicitly load environment variables from .env file
load_dotenv()


def get_generative_model():
    """
    Explicitly initializes and returns a Hugging Face generative model
    (Llama-3.1-8B-Instruct) configured explicitly for concise, deterministic responses.
    """
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not huggingfacehub_api_token:
        raise ValueError("Explicit error: HUGGINGFACEHUB_API_TOKEN is not set in .env")

    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=huggingfacehub_api_token,
        task="text-generation",
        temperature=0.05,  # explicitly lower for deterministic responses
        max_new_tokens=100,  # explicitly concise responses
    )

    return llm


# Explicit standalone test
if __name__ == "__main__":
    model = get_generative_model()
    test_prompt = "Explain briefly: What is Retrieval-Augmented Generation (RAG)?"
    response = model.invoke(test_prompt)
    print("🚀 Explicit Model Test Response:\n", response)
