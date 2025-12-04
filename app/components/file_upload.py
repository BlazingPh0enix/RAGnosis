"""
File upload component for DocuLens Streamlit app.

Provides file upload UI with progress tracking via FastAPI backend.
"""

import time
from typing import Optional, Callable
import requests
import streamlit as st

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

# API base URL
API_URL = settings.API_URL


def init_upload_state():
    """Initialize upload-related session state."""
    if "current_job_id" not in st.session_state:
        st.session_state.current_job_id = None
    
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = None
    
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    
    if "upload_error" not in st.session_state:
        st.session_state.upload_error = None


def upload_file(file) -> Optional[dict]:
    """
    Upload a file to the FastAPI backend.
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        Upload response dict or None on error
    """
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/api/documents/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"File uploaded successfully, job_id: {data['job_id']}")
            return data
        else:
            error = response.json().get("detail", "Upload failed")
            logger.error(f"Upload failed: {error}")
            st.session_state.upload_error = error
            return None
            
    except requests.exceptions.ConnectionError:
        error = "Cannot connect to API server. Is it running?"
        logger.error(error)
        st.session_state.upload_error = error
        return None
    except Exception as e:
        logger.error(f"Upload error: {e}")
        st.session_state.upload_error = str(e)
        return None


def get_job_status(job_id: str) -> Optional[dict]:
    """
    Get processing status from the API.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Status dict or None on error
    """
    try:
        response = requests.get(f"{API_URL}/api/documents/{job_id}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return None


def cancel_job(job_id: str) -> bool:
    """Cancel a processing job."""
    try:
        response = requests.post(f"{API_URL}/api/documents/{job_id}/cancel", timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Cancel error: {e}")
        return False


def render_file_uploader():
    """Render the file upload widget."""
    st.subheader("📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help=f"Maximum file size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        key="pdf_uploader",
    )
    
    if uploaded_file is not None:
        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.caption(f"📎 {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Check file size
        if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
            st.error(f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB")
            return None
        
        # Upload button
        if st.button("🚀 Process Document", use_container_width=True, type="primary"):
            return uploaded_file
    
    return None


def render_processing_progress(job_id: str) -> bool:
    """
    Render processing progress with live updates.
    
    Args:
        job_id: Job identifier to track
        
    Returns:
        True if processing completed successfully
    """
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    steps_placeholder = st.empty()
    cancel_placeholder = st.empty()
    
    # Cancel button
    if cancel_placeholder.button("❌ Cancel Processing", key="cancel_btn"):
        if cancel_job(job_id):
            st.warning("Processing cancelled")
            return False
    
    while True:
        status = get_job_status(job_id)
        
        if status is None:
            st.error("Failed to get job status")
            return False
        
        # Update progress bar
        progress_placeholder.progress(
            status["progress"] / 100,
            text=f"{status['progress']}% - {status['current_step']}"
        )
        
        # Update status message
        status_text = f"**Status:** {status['status'].replace('_', ' ').title()}"
        if status.get("page_count"):
            status_text += f" | **Pages:** {status['page_count']}"
        if status.get("image_count"):
            status_text += f" | **Images:** {status['image_count']}"
        if status.get("chunk_count"):
            status_text += f" | **Chunks:** {status['chunk_count']}"
        status_placeholder.markdown(status_text)
        
        # Update steps
        with steps_placeholder.container():
            for step in status["steps"]:
                if step["status"] == "completed":
                    st.success(f"✅ {step['name']}: {step.get('message', 'Done')}")
                elif step["status"] == "in_progress":
                    st.info(f"⏳ {step['name']}: {step.get('message', 'Processing...')}")
                elif step["status"] == "failed":
                    st.error(f"❌ {step['name']}: {step.get('message', 'Failed')}")
                else:
                    st.caption(f"⬜ {step['name']}")
        
        # Check completion
        if status["status"] == "completed":
            cancel_placeholder.empty()
            progress_placeholder.progress(1.0, text="✅ Processing complete!")
            st.session_state.current_collection = status.get("collection_name")
            st.session_state.processing_complete = True
            st.balloons()
            return True
        
        if status["status"] in ["failed", "cancelled"]:
            cancel_placeholder.empty()
            if status.get("error"):
                st.error(f"Processing failed: {status['error']}")
            return False
        
        # Wait before next poll
        time.sleep(1)


def render_upload_section():
    """Render the complete upload section with progress tracking."""
    init_upload_state()
    
    # Show error if any
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
        if st.button("Clear Error"):
            st.session_state.upload_error = None
            st.rerun()
    
    # If already processing, show progress
    if st.session_state.current_job_id and not st.session_state.processing_complete:
        st.info(f"Processing job: {st.session_state.current_job_id}")
        if render_processing_progress(st.session_state.current_job_id):
            st.success("✅ Document ready! You can now chat with it.")
            st.rerun()
        return
    
    # If processing complete, show success and option to upload new
    if st.session_state.processing_complete:
        st.success(f"✅ Document indexed to: `{st.session_state.current_collection}`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Upload New Document", use_container_width=True):
                st.session_state.current_job_id = None
                st.session_state.current_collection = None
                st.session_state.processing_complete = False
                st.rerun()
        return
    
    # Show upload widget
    uploaded_file = render_file_uploader()
    
    if uploaded_file:
        with st.spinner("Uploading file..."):
            result = upload_file(uploaded_file)
        
        if result:
            st.session_state.current_job_id = result["job_id"]
            st.session_state.upload_error = None
            st.rerun()


def check_api_health() -> bool:
    """Check if the FastAPI backend is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def render_api_status():
    """Render API connection status."""
    if check_api_health():
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Unavailable")
        st.caption(f"Start the API server: `uvicorn api.main:app --port 8000`")
