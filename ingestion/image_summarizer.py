"""
Image summarizer using GPT-5-nano-2025-08-07 for generating detailed descriptions.

This module provides functionality to generate detailed text descriptions
of images (charts, diagrams, figures) extracted from documents. These
descriptions are used for RAG indexing and retrieval.
"""

import base64
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

from openai import OpenAI

from config.settings import settings


# Default prompt for image summarization
DEFAULT_IMAGE_PROMPT = """Describe this chart/image in extreme detail for a blind user. 
Include all of the following if present:
- Main subject and type of visualization (chart, diagram, photo, etc.)
- All text, labels, titles, and annotations
- For charts: axes labels, scales, data points, trends, and patterns
- For tables: column headers, row labels, and key data values
- Colors and visual elements that convey meaning
- Key insights and takeaways from the visual
- Any legends, footnotes, or source citations

Be thorough and precise. Your description will be used for document search and retrieval."""


@dataclass
class ImageSummary:
    """Represents a summarized image with its description and metadata."""
    image_name: str
    page_number: int
    summary: str
    model_used: str
    source_document: str = ""
    
    @property
    def node_id(self) -> str:
        """Generate a unique identifier for this image summary."""
        return f"img_p{self.page_number}_{self.image_name}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "image_name": self.image_name,
            "page_number": self.page_number,
            "summary": self.summary,
            "model_used": self.model_used,
            "source_document": self.source_document,
            "node_id": self.node_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ImageSummary":
        """Create from dictionary."""
        return cls(
            image_name=data["image_name"],
            page_number=data["page_number"],
            summary=data["summary"],
            model_used=data["model_used"],
            source_document=data.get("source_document", ""),
        )


class ImageSummarizer:
    """
    Generates detailed text descriptions of images using GPT-4o-mini.
    
    This class uses OpenAI's vision capabilities to analyze images and
    produce detailed descriptions suitable for RAG indexing.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano-2025-08-07",
        prompt: Optional[str] = None,
        max_tokens: int = 500,
    ):
        """
        Initialize the image summarizer.
        
        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Vision model to use (default: gpt-4o-mini for cost efficiency).
            prompt: Custom prompt for image description.
            max_tokens: Maximum tokens for the response.
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )
        
        self.model = model
        self.prompt = prompt or DEFAULT_IMAGE_PROMPT
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self._client = OpenAI(api_key=self.api_key)
    
    def summarize_image(
        self,
        image_data: bytes,
        image_name: str,
        page_number: int,
        source_document: str = "",
        additional_context: Optional[str] = None,
    ) -> ImageSummary:
        """
        Generate a detailed summary of an image.
        
        Args:
            image_data: Raw image bytes.
            image_name: Name of the image file.
            page_number: Page number where the image was found.
            source_document: Name of the source document.
            additional_context: Optional context about the document/image.
            
        Returns:
            ImageSummary object with the generated description.
        """
        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image format from name or default to png
        image_format = "png"
        if "." in image_name:
            ext = image_name.rsplit(".", 1)[-1].lower()
            if ext in ("jpg", "jpeg", "png", "gif", "webp"):
                image_format = ext if ext != "jpg" else "jpeg"
        
        # Build the prompt with additional context if provided
        prompt = self.prompt
        if additional_context:
            prompt = f"{prompt}\n\nAdditional context: {additional_context}"
        
        prompt += f"\n\nThis image is from page {page_number} of the document."
        
        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            ) # type: ignore
            
            summary_text = response.choices[0].message.content or "No description generated."
            
        except Exception as e:
            # Return error message as summary to avoid breaking the pipeline
            summary_text = f"[Error generating image description: {str(e)}]"
        
        return ImageSummary(
            image_name=image_name,
            page_number=page_number,
            summary=summary_text,
            model_used=self.model,
            source_document=source_document,
        )
    
    def summarize_image_from_file(
        self,
        image_path: str,
        page_number: int = 0,
        source_document: str = "",
    ) -> ImageSummary:
        """
        Generate a summary from an image file.
        
        Args:
            image_path: Path to the image file.
            page_number: Page number where the image was found.
            source_document: Name of the source document.
            
        Returns:
            ImageSummary object with the generated description.
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(path, "rb") as f:
            image_data = f.read()
        
        return self.summarize_image(
            image_data=image_data,
            image_name=path.name,
            page_number=page_number,
            source_document=source_document or path.parent.name,
        )
    
    def summarize_batch(
        self,
        images: List[dict],
        source_document: str = "",
    ) -> List[ImageSummary]:
        """
        Summarize multiple images.
        
        Args:
            images: List of dicts with keys: 'data', 'name', 'page_number'
            source_document: Name of the source document.
            
        Returns:
            List of ImageSummary objects.
        """
        summaries = []
        
        for img in images:
            try:
                summary = self.summarize_image(
                    image_data=img["data"],
                    image_name=img["name"],
                    page_number=img["page_number"],
                    source_document=source_document,
                )
                summaries.append(summary)
            except Exception as e:
                print(f"Warning: Failed to summarize image {img.get('name')}: {e}")
                # Create a placeholder summary
                summaries.append(ImageSummary(
                    image_name=img.get("name", "unknown"),
                    page_number=img.get("page_number", 0),
                    summary=f"[Error: {str(e)}]",
                    model_used=self.model,
                    source_document=source_document,
                ))
        
        return summaries


def create_summarizer(
    model: str = "gpt-5-nano-2025-08-07",
    max_tokens: int = 500,
) -> ImageSummarizer:
    """
    Factory function to create an ImageSummarizer with default settings.
    
    Args:
        model: Vision model to use.
        max_tokens: Maximum tokens for responses.
        
    Returns:
        Configured ImageSummarizer instance.
    """
    return ImageSummarizer(model=model, max_tokens=max_tokens)
