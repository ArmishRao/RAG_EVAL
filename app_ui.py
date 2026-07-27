#!/usr/bin/env python
"""
Enhanced Streamlit UI for Pakistan Legal Advisor
"""

import streamlit as st
from app import ask_rag
import time
import pandas as pd

st.set_page_config(
    page_title="Pakistan Legal Advisor",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        color: #1a5276;
        font-size: 2.5rem;
        font-weight: bold;
        border-bottom: 3px solid #2e86c1;
        padding-bottom: 10px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #2e86c1;
    }
    .web-source {
        border-left-color: #e67e22;
    }
    .legal-term {
        color: #1a5276;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #fdf2e9;
        border-radius: 5px;
        padding: 10px;
        border-left: 4px solid #e67e22;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header"> Pakistan Legal Advisor</p>', unsafe_allow_html=True)

# Sidebar with detailed information
with st.sidebar:
    st.markdown("##  About")
    st.markdown("""
    This AI-powered legal assistant provides information about Pakistani law using:
    
    - ** Local Legal Database**: Pre-indexed legal documents from Pakistani law
    - ** Web Search Fallback**: Searches the web when local sources are insufficient
    - ** Smart Query Understanding**: Analyzes your question for better results
    
    All processing is done locally. Your data stays private!
    """)
    
    st.markdown("##  Features")
    st.markdown("""
     **Hybrid Search** - Vector + keyword search  
     **Query Expansion** - Better understanding of legal questions  
     **Document Reranking** - Most relevant sources first  
     **Multi-source** - Local + Web + Wikipedia  
     **Source Citations** - Transparent references  
     **No API Keys** - Fully self-contained  
    """)
    
    st.markdown("##  Disclaimer")
    st.markdown("""
    <div class="disclaimer">
    This is an AI-powered legal assistant for <strong>informational purposes only</strong>.
    For legal advice, please consult a qualified lawyer.
    </div>
    """, unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source_type" in message:
            if message["source_type"] == "web":
                st.caption(" Source: Web Search (fallback)")
            elif message["source_type"] == "local":
                st.caption(" Source: Legal Database")
            else:
                st.caption(" No sources found")

# User input
if prompt := st.chat_input("Ask a legal question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner(" Analyzing and searching for legal information..."):
            start_time = time.time()
            result = ask_rag(prompt)
            elapsed = time.time() - start_time
            
            response = result["Response"]
            st.markdown(response)
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(" Response Time", f"{elapsed:.2f}s")
            with col2:
                source_label = " Web" if result["source_type"] == "web" else " Local"
                st.metric(" Source", source_label)
            with col3:
                st.metric(" Documents", len(result["sources"]))
            with col4:
                status = " Found" if result["source_type"] != "none" else " Not Found"
                st.metric("Status", status)
            
            # Show sources in expander
            with st.expander(" View Sources"):
                for i, doc in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    
                    if doc.metadata.get('source') == 'web':
                        st.markdown(f" **Type:** Web Search")
                        st.markdown(f"**Title:** {doc.metadata.get('title', 'N/A')}")
                        st.markdown(f"**URL:** {doc.metadata.get('source_url', 'N/A')}")
                    else:
                        st.markdown(f" **Type:** Legal Database")
                        st.markdown(f"**Book:** {doc.metadata.get('book', 'N/A')}")
                        st.markdown(f"**Section:** {doc.metadata.get('section', 'N/A')}")
                        st.markdown(f"**Heading:** {doc.metadata.get('heading', 'N/A')}")
                    
                    # Show content preview
                    content_preview = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                    with st.expander(" View Content"):
                        st.text(content_preview)
                    st.divider()
            
            # Show query analysis in expander
            if "query_analysis" in result:
                with st.expander("🔍 Query Analysis"):
                    analysis = result["query_analysis"]
                    st.write("**Topics Detected:**", ", ".join(analysis.get("topics", ["None"])) or "None")
                    st.write("**Sections Detected:**", ", ".join(analysis.get("sections", ["None"])) or "None")
    
    # Store message with source info
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "source_type": result["source_type"]
    })