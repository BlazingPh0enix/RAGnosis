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
from config.logging_config import get_logger, log_execution_time

logger = get_logger(__name__)


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
    """Represents a summarized image with its description."""
    image_name: str
    page_number: int
    summary: str
    model_used: str
    source_document: str = ""
    
    @property
    def node_id(self) -> str:
        return f"img_p{self.page_number}_{self.image_name}"
    
    def to_dict(self) -> dict:
        return {"image_name": self.image_name, "page_number": self.page_number,
                "summary": self.summary, "model_used": self.model_used,
                "source_document": self.source_document, "node_id": self.node_id}
    
    @classmethod
    def from_dict(cls, data: dict) -> "ImageSummary":
        return cls(image_name=data["image_name"], page_number=data["page_number"],
                   summary=data["summary"], model_used=data["model_used"],
                   source_document=data.get("source_document", ""))


class ImageSummarizer:
    """Generates detailed text descriptions of images using gpt-5-nano-2025-08-07."""
    
    def __init__(
        self, api_key: Optional[str] = None, model: str = "gpt-5-nano-2025-08-07",
        prompt: Optional[str] = None, max_tokens: int = 500,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        self.model, self.prompt, self.max_tokens = model, prompt or DEFAULT_IMAGE_PROMPT, max_tokens
        self._client = OpenAI(api_key=self.api_key)
        logger.debug(f"ImageSummarizer initialized (model={model})")
    
    def summarize_image(
        self, image_data: bytes, image_name: str, page_number: int,
        source_document: str = "", additional_context: Optional[str] = None,
    ) -> ImageSummary:
        """Generate a detailed summary of an image."""
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine format from extension
        ext = image_name.rsplit(".", 1)[-1].lower() if "." in image_name else "png"
        img_fmt = ext if ext != "jpg" else "jpeg"
        if img_fmt not in ("jpeg", "png", "gif", "webp"):
            img_fmt = "png"
        
        prompt = self.prompt
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        prompt += f"\n\nThis image is from page {page_number}."
        
        try:
            response = self._client.responses.create(
                model=self.model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_image", "image_url": {"url": f"data:image/{img_fmt};base64,{base64_image}"}},
                    ],
                }],
                max_tokens=self.max_tokens,
            )  # type: ignore
            summary_text = response.choices[0].message.content or "No description generated."
            logger.info(f"Generated summary for {image_name} ({len(summary_text)} chars)")
        except Exception as e:
            logger.error(f"Failed to summarize {image_name}: {e}")
            summary_text = f"[Error: {str(e)}]"
        
        return ImageSummary(
            image_name=image_name, page_number=page_number, summary=summary_text,
            model_used=self.model, source_document=source_document,
        )
    
    def summarize_image_from_file(
        self, image_path: str, page_number: int = 0, source_document: str = "",
    ) -> ImageSummary:
        """Generate a summary from an image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.summarize_image(
            image_data=path.read_bytes(), image_name=path.name,
            page_number=page_number, source_document=source_document or path.parent.name,
        )
    
    def summarize_batch(self, images: List[dict], source_document: str = "") -> List[ImageSummary]:
        """Summarize multiple images. Each dict needs: 'data', 'name', 'page_number'."""
        logger.info(f"Batch summarizing {len(images)} images")
        summaries = []
        for img in images:
            try:
                summaries.append(self.summarize_image(
                    image_data=img["data"], image_name=img["name"],
                    page_number=img["page_number"], source_document=source_document,
                ))
            except Exception as e:
                logger.warning(f"Failed to summarize {img.get('name')}: {e}")
                summaries.append(ImageSummary(
                    image_name=img.get("name", "unknown"), page_number=img.get("page_number", 0),
                    summary=f"[Error: {str(e)}]", model_used=self.model, source_document=source_document,
                ))
        return summaries


def create_summarizer(model: str = "gpt-5-nano-2025-08-07", max_tokens: int = 500) -> ImageSummarizer:
    """Factory function to create an ImageSummarizer."""
    return ImageSummarizer(model=model, max_tokens=max_tokens)
