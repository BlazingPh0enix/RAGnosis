"""
DocuLens - Multi-Modal RAG Streamlit Application

This is the main Streamlit application for DocuLens, providing an
interactive chat interface with source inspection capabilities.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from app.components.chat_interface import (
    init_chat_state,
    add_message,
    display_chat_history,
    clear_chat_history,
    render_welcome_message,
)
from app.components.source_inspector import (
    render_source_inspector,
    render_source_stats,
)
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title="DocuLens - Multi-Modal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_app_state():
    """Initialize all application state."""
    init_chat_state()
    
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    
    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False
    
    if "error_message" not in st.session_state:
        st.session_state.error_message = None


@st.cache_resource
def load_query_engine(collection_name: Optional[str] = None):
    """
    Load the query engine (cached to avoid reloading).
    
    Args:
        collection_name: Optional Qdrant collection name
        
    Returns:
        DocuLensQueryEngine instance or None on error
    """
    try:
        from retrieval.query_engine import load_query_engine
        
        return load_query_engine(collection_name=collection_name)
    except Exception as e:
        st.error(f"Failed to load query engine: {e}")
        return None


def handle_query(query: str) -> dict:
    """
    Handle a user query and return the response.
    
    Args:
        query: User's question
        
    Returns:
        Dict with 'response' and 'sources'
    """
    if st.session_state.query_engine is None:
        return {
            "response": "❌ Query engine not loaded. Please check the index configuration.",
            "sources": [],
        }
    
    try:
        # Query the engine
        result = st.session_state.query_engine.query(query)
        
        return {
            "response": result.response,
            "sources": result.sources,
        }
    except Exception as e:
        return {
            "response": f"❌ Error processing query: {str(e)}",
            "sources": [],
        }


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("🔍 DocuLens")
        st.caption("Multi-Modal RAG System")
        
        st.divider()
        
        # Index configuration
        st.subheader("⚙️ Configuration")
        
        collection_name = st.text_input(
            "Collection Name",
            value=settings.QDRANT_COLLECTION or "doculens",
            help="Qdrant collection to query",
        )
        
        # Load index button
        if st.button("🔄 Load Index", use_container_width=True):
            with st.spinner("Loading index..."):
                engine = load_query_engine(collection_name)
                if engine:
                    st.session_state.query_engine = engine
                    st.session_state.index_loaded = True
                    st.success("✅ Index loaded successfully!")
                else:
                    st.session_state.index_loaded = False
                    st.error("Failed to load index")
        
        # Status indicator
        if st.session_state.index_loaded:
            st.success("✅ Index Active")
        else:
            st.warning("⚠️ Index Not Loaded")
        
        st.divider()
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()
        
        st.divider()
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **DocuLens** is a multi-modal RAG system that can:
            - 📄 Answer questions from documents
            - 📊 Understand tables and charts
            - 🖼️ Describe and reference images
            - 📝 Provide source citations
            """
        )
        
        st.divider()
        st.caption("Made with ❤️ using LlamaIndex + Qdrant")


def render_main_content():
    """Render the main content area with chat and sources."""
    # Two-column layout
    chat_col, source_col = st.columns([3, 2])
    
    with chat_col:
        st.header("💬 Chat with Documents")
        
        # Welcome message if no chat history
        if not st.session_state.get("messages"):
            render_welcome_message()
        
        # Chat history
        display_chat_history()
        
        # Processing indicator
        if st.session_state.get("is_processing"):
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching and generating response..."):
                    pass
        
        # Chat input
        if not st.session_state.get("is_processing"):
            if prompt := st.chat_input(
                "Ask a question about your documents...",
                disabled=not st.session_state.index_loaded,
            ):
                # Add user message
                add_message("user", prompt)
                
                # Process query
                with st.spinner(""):
                    result = handle_query(prompt)
                
                # Add assistant response
                add_message("assistant", result["response"], result.get("sources", []))
                
                # Update current sources
                st.session_state.current_sources = result.get("sources", [])
                
                st.rerun()
        
        # Show message if index not loaded
        if not st.session_state.index_loaded:
            st.warning("⚠️ Please load an index from the sidebar to start chatting.")
    
    with source_col:
        st.header("📚 Sources")
        
        # Get current sources
        sources = st.session_state.get("current_sources", [])
        
        # Render source inspector
        render_source_inspector(
            sources=sources,
            title="Retrieved Context",
            show_stats=True,
        )


def main():
    """Main application entry point."""
    # Initialize state
    init_app_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
