import os
from typing import List, Union

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embedding_model_function(texts: Union[str, List[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]  # Explicitly ensure texts is a list

    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = [data.embedding for data in response.data]

    return embeddings  # returns a list of embedding vectors
