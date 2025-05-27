import os

from langchain.llms import HuggingFaceHub


def get_generative_model():
    return HuggingFaceHub(
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        repo_id="meta-llama/Llama-3-8B-Instruct",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
    )
