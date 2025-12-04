"""
Chat interface component for DocuLens Streamlit app.

This module provides the chat UI component with message input,
chat history display, and loading states.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Callable


def init_chat_state():
    """Initialize chat-related session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False


def add_message(role: str, content: str, sources: Optional[List[Dict]] = None):
    """
    Add a message to the chat history.
    
    Args:
        role: "user" or "assistant"
        content: Message content
        sources: Optional source information for assistant messages
    """
    message: Dict[str, Any] = {
        "role": role,
        "content": content,
    }
    if sources:
        message["sources"] = sources
    
    st.session_state.messages.append(message)


def clear_chat_history():
    """Clear all chat history."""
    st.session_state.messages = []
    st.session_state.current_sources = []


def display_message(message: Dict[str, Any]):
    """
    Display a single chat message.
    
    Args:
        message: Message dict with role and content
    """
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)


def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        display_message(message)


def render_chat_input(
    on_submit: Callable[[str], None],
    placeholder: str = "Ask a question about your documents...",
    disabled: bool = False,
):
    """
    Render the chat input field.
    
    Args:
        on_submit: Callback function when user submits a message
        placeholder: Input placeholder text
        disabled: Whether input is disabled
    """
    if prompt := st.chat_input(placeholder, disabled=disabled):
        on_submit(prompt)


def render_chat_interface(
    query_handler: Callable[[str], Dict[str, Any]],
    show_clear_button: bool = True,
):
    """
    Render the complete chat interface.
    
    Args:
        query_handler: Function that takes a query string and returns
                      a dict with 'response' and optional 'sources'
        show_clear_button: Whether to show the clear history button
    """
    # Initialize state
    init_chat_state()
    
    # Header with clear button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Chat")
    with col2:
        if show_clear_button:
            if st.button("🗑️ Clear", use_container_width=True):
                clear_chat_history()
                st.rerun()
    
    # Chat history container
    chat_container = st.container()
    
    with chat_container:
        display_chat_history()
        
        # Show processing indicator
        if st.session_state.is_processing:
            with st.chat_message("assistant"):
                st.markdown("🔍 Searching documents and generating response...")
    
    # Chat input
    def handle_input(prompt: str):
        # Add user message
        add_message("user", prompt)
        
        # Set processing state
        st.session_state.is_processing = True
        st.rerun()
    
    # Only allow input when not processing
    if not st.session_state.is_processing:
        render_chat_input(handle_input)
    
    # Process pending query
    if st.session_state.is_processing:
        # Get the last user message
        last_user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if last_user_msg:
            try:
                # Call query handler
                result = query_handler(last_user_msg)
                
                # Add assistant response
                response = result.get("response", "I couldn't generate a response.")
                sources = result.get("sources", [])
                
                add_message("assistant", response, sources)
                
                # Update current sources for the inspector
                st.session_state.current_sources = sources
                
            except Exception as e:
                add_message("assistant", f"❌ Error: {str(e)}")
                st.session_state.current_sources = []
            
            finally:
                st.session_state.is_processing = False
                st.rerun()


def render_welcome_message():
    """Render a welcome message when chat is empty."""
    if not st.session_state.get("messages"):
        st.info(
            "👋 Welcome to DocuLens! Ask questions about your indexed documents.\n\n"
            "**Example questions:**\n"
            "- What is the main topic of this document?\n"
            "- Summarize the key findings from page 5\n"
            "- What does the chart on page 3 show?\n"
            "- Compare the revenue figures across quarters"
        )
