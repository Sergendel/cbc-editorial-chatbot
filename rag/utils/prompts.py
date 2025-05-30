from langchain.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    template="""Use the provided context to answer the question clearly and explicitly.

    Context:
    {context}

    Question:
    {question}

    Answer explicitly and clearly:""",
    input_variables=["context", "question"],
)
