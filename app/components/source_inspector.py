"""
Source inspector component for DocuLens Streamlit app.

This module provides the sources panel that displays retrieved chunks
with content type badges, relevance scores, and original images.
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.settings import settings


# Content type styling
CONTENT_TYPE_BADGES = {
    "text": ("📝", "Text", "#4A90D9"),
    "table": ("📊", "Table", "#50C878"),
    "image_summary": ("🖼️", "Image", "#FFB347"),
}


def get_content_badge(content_type: str) -> tuple:
    """
    Get badge info for a content type.
    
    Returns:
        Tuple of (emoji, label, color)
    """
    return CONTENT_TYPE_BADGES.get(content_type, ("📄", "Unknown", "#808080"))


def format_score(score: Optional[float]) -> str:
    """Format relevance score as percentage."""
    if score is None:
        return "N/A"
    return f"{score * 100:.1f}%"


def render_source_card(
    source: Dict[str, Any],
    index: int,
    show_full_text: bool = False,
    images_dir: Optional[Path] = None,
):
    """
    Render a single source card.
    
    Args:
        source: Source dictionary with metadata
        index: Source index for display
        show_full_text: Whether to show full text or preview
        images_dir: Directory containing extracted images
    """
    content_type = source.get("content_type", "text")
    emoji, label, color = get_content_badge(content_type)
    page_num = source.get("page_number", "?")
    score = source.get("score")
    document = source.get("source_document", "Unknown")
    text_preview = source.get("text_preview", "")
    
    # Card container
    with st.container():
        # Header row
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown(f"**#{index + 1}**")
        
        with col2:
            st.markdown(
                f'<span style="background-color: {color}; padding: 2px 8px; '
                f'border-radius: 4px; color: white; font-size: 0.8em;">'
                f'{emoji} {label}</span>',
                unsafe_allow_html=True,
            )
        
        with col3:
            st.markdown(f"**{format_score(score)}**")
        
        # Page and document info
        st.caption(f"📄 Page {page_num} | 📁 {document}")
        
        # Image name for image summaries
        if content_type == "image_summary":
            image_name = source.get("image_name", "")
            if image_name:
                st.caption(f"🖼️ {image_name}")
                
                # Try to display the actual image
                if images_dir:
                    image_path = images_dir / image_name
                    if image_path.exists():
                        try:
                            st.image(str(image_path), caption=image_name, width=200)
                        except Exception:
                            pass
        
        # Text content
        if text_preview:
            if show_full_text:
                st.markdown(text_preview)
            else:
                # Truncate for preview
                preview = text_preview[:300] + "..." if len(text_preview) > 300 else text_preview
                with st.expander("View content"):
                    st.markdown(preview)
        
        st.divider()


def render_source_inspector(
    sources: List[Dict[str, Any]],
    title: str = "📚 Sources",
    show_stats: bool = True,
    images_dir: Optional[str] = None,
):
    """
    Render the complete source inspector panel.
    
    Args:
        sources: List of source dictionaries
        title: Panel title
        show_stats: Whether to show statistics
        images_dir: Directory containing extracted images
    """
    st.subheader(title)
    
    if not sources:
        st.info("No sources retrieved yet. Ask a question to see relevant sources.")
        return
    
    # Statistics
    if show_stats:
        render_source_stats(sources)
    
    # Images directory
    img_dir = Path(images_dir) if images_dir else Path(settings.DATA_DIR) / "images"
    
    # Render each source
    for i, source in enumerate(sources):
        render_source_card(
            source=source,
            index=i,
            images_dir=img_dir,
        )


def render_source_stats(sources: List[Dict[str, Any]]):
    """
    Render statistics about retrieved sources.
    
    Args:
        sources: List of source dictionaries
    """
    if not sources:
        return
    
    # Count by type
    type_counts = {}
    pages = set()
    documents = set()
    
    for source in sources:
        content_type = source.get("content_type", "text")
        type_counts[content_type] = type_counts.get(content_type, 0) + 1
        pages.add(source.get("page_number", 0))
        documents.add(source.get("source_document", ""))
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sources", len(sources))
    
    with col2:
        st.metric("Pages", len(pages))
    
    with col3:
        st.metric("Documents", len(documents))
    
    # Type breakdown
    type_str = " | ".join([
        f"{get_content_badge(t)[0]} {count}"
        for t, count in type_counts.items()
    ])
    st.caption(f"Types: {type_str}")
    
    st.divider()


def render_cited_pages(cited_pages: List[int], all_pages: List[int]):
    """
    Render a visualization of cited vs available pages.
    
    Args:
        cited_pages: List of page numbers cited in the response
        all_pages: List of all page numbers in context
    """
    if not all_pages:
        return
    
    st.caption("📖 **Page Citations**")
    
    # Create a visual representation
    page_display = []
    for page in sorted(set(all_pages)):
        if page in cited_pages:
            page_display.append(f"**[{page}]**")
        else:
            page_display.append(f"{page}")
    
    st.markdown(" • ".join(page_display))


def render_compact_sources(sources: List[Dict[str, Any]], max_display: int = 3):
    """
    Render a compact view of sources for inline display.
    
    Args:
        sources: List of source dictionaries
        max_display: Maximum number of sources to show
    """
    if not sources:
        return
    
    displayed = sources[:max_display]
    remaining = len(sources) - max_display
    
    for source in displayed:
        emoji, label, _ = get_content_badge(source.get("content_type", "text"))
        page = source.get("page_number", "?")
        score = format_score(source.get("score"))
        
        st.caption(f"{emoji} Page {page} ({score})")
    
    if remaining > 0:
        st.caption(f"... and {remaining} more sources")
