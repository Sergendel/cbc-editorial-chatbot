import sys
from pathlib import Path

# Explicitly add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from rag.chains.rag_chain import get_rag_chain

st.title("CBC Editorial Assistant Chatbot 📰")

query = st.text_input("Enter your query explicitly here:")

if st.button("Get Explicit Answer") and query:
    with st.spinner("Retrieving and generating explicitly..."):
        rag_chain = get_rag_chain()
        result = rag_chain.invoke({"query": query})

        st.markdown("### 📌 Generated Answer")
        st.write(result["result"])

        st.markdown("### 📚 Source Documents")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata.get('url', 'No URL provided')}")
