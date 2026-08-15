#!/usr/bin/env python
"""
Streamlit UI for Programming Documentation Assistant
"""

import streamlit as st
from app import ask_rag
import time

st.set_page_config(
    page_title="Programming Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #3498db;
    }
    .code-block {
        background-color: #2d2d2d;
        color: #f8f8f2;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📚 Programming Documentation Assistant</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This AI-powered assistant helps you understand programming concepts using:
    
    - **📚 Documentation**: Python, LangChain, FastAPI, PyTorch
    - **💡 Examples**: Code examples from official docs
    - **🔍 Search**: Semantic search through documentation
    
    All processing is done locally. Your data stays private!
    """)
    
    st.markdown("## 🚀 Supported Docs")
    st.markdown("""
    - **Python** - Standard library, builtins
    - **LangChain** - Framework for LLMs
    - **FastAPI** - Modern web framework
    - **PyTorch** - Deep learning library
    
    More coming soon!
    """)
    
    st.markdown("## 📝 Note")
    st.markdown("""
    This assistant uses a local vector database of documentation.
    Responses are based on the ingested documentation.
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a programming question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documentation..."):
            start_time = time.time()
            result = ask_rag(prompt)
            elapsed = time.time() - start_time
            
            st.markdown(result["Response"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Response Time", f"{elapsed:.2f}s")
            with col2:
                st.metric("📚 Sources", len(result["sources"]))
            with col3:
                status = "✅ Found" if result["source_type"] != "none" else "❌ Not Found"
                st.metric("Status", status)
            
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    st.markdown(f"📄 File: {doc.metadata.get('source_file', 'N/A')}")
                    st.markdown(f"📂 Type: {doc.metadata.get('doc_type', 'N/A')}")
                    if doc.metadata.get('heading'):
                        st.markdown(f"📌 Heading: {doc.metadata.get('heading')}")
                    
                    preview = doc.page_content[:300]
                    st.markdown(f"```\n{preview}...\n```")
                    st.divider()
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["Response"]
    })