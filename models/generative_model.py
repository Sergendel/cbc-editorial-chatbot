import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

#  load environment variables from .env file
load_dotenv()

# Singleton instances for each model
_GENERATIVE_MODELS = {}


def get_generative_model_Llama():
    """
    Initializes and returns a Hugging Face generative model
    (Llama-3.1-8B-Instruct) configured  for concise, deterministic responses.
    """
    if "llama" not in _GENERATIVE_MODELS:
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not huggingfacehub_api_token:
            raise ValueError("Error: HUGGINGFACEHUB_API_TOKEN is not set in .env")

        _GENERATIVE_MODELS["llama"] = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/"
            "models/meta-llama/Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=huggingfacehub_api_token,
            task="text-generation",
            temperature=0.05,
            max_new_tokens=100,
        )

    return _GENERATIVE_MODELS["llama"]


def get_generative_model():
    """
     initializes and returns a Hugging Face generative model
    (Mistral-7B-Instruct-v0.2) configured  for concise, deterministic responses.
    """
    if "mistral" not in _GENERATIVE_MODELS:
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not huggingfacehub_api_token:
            raise ValueError("Error: HUGGINGFACEHUB_API_TOKEN is not set in .env")

        _GENERATIVE_MODELS["mistral"] = HuggingFaceEndpoint(
            endpoint_url="https://api-inference.huggingface.co/models/"
            "mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=huggingfacehub_api_token,
            task="text-generation",
            temperature=0.05,
            max_new_tokens=100,
        )

    return _GENERATIVE_MODELS["mistral"]


def get_generative_model_openai():
    """
    Initializes and returns an OpenAI generative model (GPT-3.5-turbo)
    configured  for concise, deterministic responses.
    """
    if "openai" not in _GENERATIVE_MODELS:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Error: OPENAI_API_KEY is not set in .env")

        _GENERATIVE_MODELS["openai"] = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.05,
            max_tokens=150,
        )

    return _GENERATIVE_MODELS["openai"]


if __name__ == "__main__":
    model = get_generative_model()
    test_prompt = "Explain briefly: What is Retrieval-Augmented Generation (RAG)?"
    response = model.invoke(test_prompt)
    print("🚀 Model Test Response:\n", response)
