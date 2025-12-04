"""DocuLens - Multi-Modal RAG Streamlit Application

Interactive chat interface with file upload and source inspection.
Uses FastAPI backend for document processing.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import streamlit as st
import json

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
from app.components.file_upload import (
    init_upload_state,
    render_upload_section,
    render_api_status,
    check_api_health,
)
from config.settings import settings
from config.logging_config import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# API URL
API_URL = settings.API_URL


# Page configuration
st.set_page_config(
    page_title="DocuLens - Multi-Modal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_app_state():
    """Initialize all application state."""
    logger.debug("Initializing application state")
    init_chat_state()
    init_upload_state()
    
    if "query_ready" not in st.session_state:
        st.session_state.query_ready = False
    
    if "error_message" not in st.session_state:
        st.session_state.error_message = None


def handle_query(query: str) -> dict:
    """
    Handle a user query via the FastAPI backend.
    
    Args:
        query: User's question
        
    Returns:
        Dict with 'response' and 'sources'
    """
    collection = st.session_state.get("current_collection")
    
    if not collection:
        logger.warning("Query attempted with no collection")
        return {
            "response": "❌ No document loaded. Please upload and process a document first.",
            "sources": [],
        }
    
    logger.info(f"Querying collection '{collection}': '{query[:50]}...'")
    
    try:
        response = requests.post(
            f"{API_URL}/api/query",
            json={
                "query": query,
                "collection_name": collection,
                "top_k": 5,
            },
            timeout=60,
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Query successful, {len(data.get('sources', []))} sources")
            
            # Format sources for display
            sources = [
                {
                    "page_number": s.get("page_number", 1),
                    "content_type": s.get("content_type", "text"),
                    "source_document": s.get("source_document", "unknown"),
                    "score": s.get("score", 0),
                    "text_preview": s.get("text_preview", ""),
                    "image_name": s.get("image_name"),
                }
                for s in data.get("sources", [])
            ]
            
            return {
                "response": data.get("response", "No response generated"),
                "sources": sources,
            }
        else:
            error = response.json().get("detail", "Query failed")
            logger.error(f"Query failed: {error}")
            return {
                "response": f"❌ Error: {error}",
                "sources": [],
            }
            
    except requests.exceptions.ConnectionError:
        logger.error("API connection error")
        return {
            "response": "❌ Cannot connect to API server. Please ensure it's running.",
            "sources": [],
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {
            "response": f"❌ Error: {str(e)}",
            "sources": [],
        }


def handle_query_streaming(query: str, response_placeholder) -> dict:
    """
    Handle a user query via the FastAPI streaming backend.
    
    Args:
        query: User's question
        
    Returns:
        Dict with 'response' and 'sources'
    """
    collection = st.session_state.get("current_collection")
    
    if not collection:
        logger.warning("Query attempted with no collection")
        return {
            "response": "❌ No document loaded. Please upload and process a document first.",
            "sources": [],
        }
    
    logger.info(f"Streaming query: '{query[:50]}...' on collection '{collection}'")
    
    try:
        response = requests.post(
            f"{API_URL}/api/query/stream",
            json={
                "query": query,
                "collection_name": collection,
                "top_k": 5,
            },
            stream=True,
            timeout=120,
        )
        
        if response.status_code != 200:
            error = response.text
            logger.error(f"Streaming query failed: {error}")
            return {
                "response": f"❌ Error: {error}",
                "sources": [],
            }
        
        # Create a placeholder for the streaming response
        full_response = ""
        sources = []
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: '
                    if data == '[DONE]':
                        break
                    try:
                        item = json.loads(data)
                        if item["type"] == "sources":
                            sources = item["data"]
                        elif item["type"] == "chunk":
                            chunk = item["data"]
                            full_response += chunk
                            response_placeholder.markdown(full_response)
                    except json.JSONDecodeError:
                        continue
        
        return {
            "response": full_response,
            "sources": sources,
        }
        
    except requests.exceptions.ConnectionError:
        logger.error("API connection error")
        return {
            "response": "❌ Cannot connect to API server. Please ensure it's running.",
            "sources": [],
        }
    except Exception as e:
        logger.error(f"Streaming query error: {e}")
        return {
            "response": f"❌ Error: {str(e)}",
            "sources": [],
        }


def render_sidebar():
    """Render the sidebar with upload and configuration."""
    with st.sidebar:
        st.title("🔍 DocuLens")
        st.caption("Multi-Modal RAG System")
        
        st.divider()
        
        # API Status
        render_api_status()
        
        st.divider()
        
        # File Upload Section
        render_upload_section()
        
        st.divider()
        
        # Chat controls (only show if document is ready)
        if st.session_state.get("processing_complete"):
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
        st.caption("Made with ❤️ using LlamaIndex + FastAPI + Qdrant")


def render_main_content():
    """Render the main content area with chat and sources."""
    # Check if API is available
    api_available = check_api_health()
    document_ready = st.session_state.get("processing_complete", False)
    
    # Three-column layout: chat, spacer, sources (right sidebar)
    chat_col, spacer_col, source_col = st.columns([4, 1, 2])
    
    with chat_col:
        st.header("💬 Chat with Documents")
        
        # Show status messages
        if not api_available:
            st.error(
                "⚠️ **API Server Not Running**\n\n"
                "Start the FastAPI backend:\n"
                "```\nuvicorn api.main:app --reload --port 8000\n```"
            )
        elif not document_ready:
            st.info("📄 **Upload a document** in the sidebar to start chatting.")
        
        # Welcome message if no chat history
        if not st.session_state.get("messages"):
            render_welcome_message()
        
        # Chat history
        display_chat_history()
        
        # Chat input (enabled only when document is ready)
        chat_enabled = api_available and document_ready
        
        if prompt := st.chat_input(
            "Ask a question about your document...",
            disabled=not chat_enabled,
        ):
            # Add user message
            add_message("user", prompt)
            
            # Create placeholder for streaming response
            response_placeholder = st.empty()
            
            # Process query with streaming
            result = handle_query_streaming(prompt, response_placeholder)
            
            # Clear placeholder and add final message
            response_placeholder.empty()
            add_message("assistant", result["response"], result.get("sources", []))
            
            # Update current sources
            st.session_state.current_sources = result.get("sources", [])
            
            st.rerun()
        
        # Helper text
        if not chat_enabled:
            if api_available:
                st.caption("👆 Upload and process a document to enable chat")
            else:
                st.caption("👆 Start the API server first")
    
    with spacer_col:
        # Empty spacer column
        pass
    
    with source_col:
        st.header("📚 Sources")
        
        # Get current sources
        sources = st.session_state.get("current_sources", [])
        
        if sources:
            # Render source inspector
            render_source_inspector(
                sources=sources,
                title="Retrieved Context",
                show_stats=True,
            )
        else:
            st.caption("Sources will appear here after you ask a question.")
            
            # Show collection info if available
            if document_ready:
                collection = st.session_state.get("current_collection", "")
                st.info(f"📁 Active collection: `{collection}`")


def main():
    """Main application entry point."""
    logger.info("Starting DocuLens Streamlit application")
    
    # Initialize state
    init_app_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
