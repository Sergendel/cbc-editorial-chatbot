import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from rag.chains.rag_chain_mixed import explicit_rag

st.title("📰 CBC Editorial Assistant Chatbot")
st.write("### Explicitly enter your query below:")

query = st.text_area("Query", placeholder="Enter your explicit query here...")

if st.button("Get Explicit Answer"):
    if query.strip():
        with st.spinner("🔎 Classifying and retrieving relevant data..."):
            result = explicit_rag(query)
            st.success(
                f"✅ Explicit Answer (classified as: {result['classification']}):"
            )
            st.write(result["result"])

            if result.get("source_documents"):
                st.write("\n📚 **Explicitly Retrieved Sources:**")
                for i, doc in enumerate(result["source_documents"], 1):
                    st.write(f"**Source {i}:**")
                    st.write(doc.metadata)
                    st.write("-" * 30)
    else:
        st.error("Please explicitly enter a valid query!")
