from langchain.prompts import PromptTemplate

#  prompt templates per intent
PROMPTS = {
    "policy_query": PromptTemplate(
        template="""You are the CBC Editorial Assistant Chatbot.
         Clearly summarize the relevant editorial policy based ONLY
         on the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:""",
        input_variables=["context", "question"],
    ),
    "headline_request": PromptTemplate(
        template="""You are an SEO assistant. Generate a concise, keyword-rich headline
        based on the provided context.

        Context:
        {context}

        Headline:""",
        input_variables=["context"],
    ),
    "summary_request": PromptTemplate(
        template="""You are a journalist assistant. Provide a clear, concise,
         one-sentence summary based on the provided context.

        Context:
        {context}

        Summary:""",
        input_variables=["context"],
    ),
    "full_article_request": PromptTemplate(
        template="""You are an editorial assistant. Provide a clear and concise summary
         or highlight key details from the provided article content.

        Context:
        {context}

        Details:""",
        input_variables=["context"],
    ),
    "unknown": PromptTemplate(
        template="""I'm not sure I understand your request clearly.
         Could you please clarify or rephrase your question?

        Original Question:
        {question}

        Request for Clarification:""",
        input_variables=["question"],
    ),
}


# Prompt selection function based on intent
def get_prompt(intent: str, context: str, question: str) -> str:
    prompt_template = PROMPTS.get(intent, PROMPTS["unknown"])
    return prompt_template.format(context=context, question=question)


# Standalone testing/demo
if __name__ == "__main__":
    test_intents_and_queries = [
        ("policy_query", "CBC's policy on anonymous sources."),
        ("headline_request", ""),
        ("summary_request", ""),
        ("full_article_request", ""),
        ("metadata_specific_request", "Give details for article ID 1.6272172."),
        ("unknown", "What's the weather?"),
    ]

    context = "Example context for testing purposes."

    for intent, query in test_intents_and_queries:
        prompt = get_prompt(intent, context, query)
        print(f"Intent: {intent}\nPrompt:\n{prompt}\n{'-'*40}\n")
