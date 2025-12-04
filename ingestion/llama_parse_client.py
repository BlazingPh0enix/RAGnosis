"""
LlamaParse client wrapper with credit-optimized settings.
"""

import os
from pathlib import Path
from typing import Optional

from llama_cloud_services import LlamaParse

from config.settings import settings

def create_parser(
    api_key: Optional[str] = None,
    auto_mode: bool = True,
    save_images: bool = True,
) -> LlamaParse:
    api_key = api_key or settings.LLAMA_CLOUD_API_KEY
    
    if not api_key:
        raise ValueError(
            "LlamaCloud API key is required. Set LLAMA_CLOUD_API_KEY environment variable."
        )
    
    parser_kwargs = {
        "api_key": api_key,
        "save_images": save_images,
        "page_prefix": settings.PAGE_PREFIX,
        # Output tables as HTML for better structure preservation
        "output_tables_as_HTML": True,
        # Merge tables that span multiple pages
        "merge_tables_across_pages_in_markdown": True,
    }
    
    if auto_mode:
        parser_kwargs.update({
            "auto_mode": True,
            "auto_mode_trigger_on_image_in_page": True,
            "auto_mode_trigger_on_table_in_page": True,
        })
    else:
        # Use cost-effective preset if auto_mode is disabled
        parser_kwargs["preset"] = "cost_effective"
    
    return LlamaParse(**parser_kwargs)


def parse_document(file_path: str, parser: Optional[LlamaParse] = None):
    """
    Parse a PDF document and return the JobResult.
    """
    file_path = str(Path(file_path).resolve())
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if parser is None:
        parser = create_parser()
    
    return parser.parse(file_path)
