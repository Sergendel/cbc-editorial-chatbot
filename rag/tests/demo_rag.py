import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation",
    temperature=0.1,
    max_new_tokens=512,
)

response = llm.invoke("Explain briefly: What is Retrieval-Augmented Generation (RAG)?")
print("Model Response:", response)
