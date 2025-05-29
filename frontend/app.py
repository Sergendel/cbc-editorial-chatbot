import sys
from pathlib import Path

import streamlit as st

# Add project root  to PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from rag.chains.IntentDrivenRAGChain import intent_driven_rag_chain

st.set_page_config(page_title="CBC Editorial Assistant", page_icon="📚", layout="wide")

st.title("📚 CBC Editorial Assistant Chatbot")
st.markdown(
    """
This chatbot leverages advanced Retrieval-Augmented Generation (RAG) to assist with CBC editorial tasks, including policy queries, SEO headlines, article summaries, and detailed article retrieval.

Enter your query below and receive intent-driven responses along with source documents.
"""
)

user_query = st.text_area("🔎 Enter your query  here:", height=150)

if st.button("🚀 Generate Response"):
    if user_query.strip():
        with st.spinner("Generating response..."):
            try:
                result = intent_driven_rag_chain(user_query)
                response, sources = result["response"], result["sources"]

                st.markdown("### 📝 Response:")
                st.write(response)

                if sources:
                    st.markdown("### 📌 Source Documents:")
                    for source in sources:
                        if source["type"] == "Guidelines":
                            st.info(
                                f"""
                            **Type:** Guidelines  
                            **Title:** {source['document_title']}  
                            **Section Path:** {source['section_path']}  
                            **URL:** [{source['url']}]({source['url']})  
                            **Timestamp:** {source['timestamp']}  
                            **Snippet:** {source['content_snippet']}  
                            """
                            )

                        elif source["type"] == "News":
                            st.success(
                                f"""
                            **Type:** News  
                            **Title:** {source['title']}  
                            **ID:** {source['id']}  
                            **Published:** {source['publish_time']}  
                            **Last Updated:** {source['last_update']}  
                            **Categories:** {source['categories']}  
                            **Snippet:** {source['chunk_text_snippet']}  
                            """
                            )

                        else:
                            st.warning(
                                f"""
                            **Type:** Unknown  
                            **Metadata:** {source['metadata']}  
                            """
                            )

                else:
                    st.info("No source documents  available.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query  to get a response.")

# Explicit footer
txt = """
---
**CBC Editorial Assistant Chatbot**  
Powered by Mistral-7B-Instruct-v0.3 & FAISS vector indexing  
Made  for accurate, reliable editorial assistance.
"""
st.markdown(txt)
