# rag_schemas.py explicitly defined
from pydantic import BaseModel


class RAGRequest(BaseModel):
    query: str
    top_k: int = 3


class RAGResponse(BaseModel):
    answer: str
    sources: list
