#!/usr/bin/env python3
"""
CLI script to ingest PDF documents through the DocuLens pipeline.

This script processes PDF documents using LlamaParse for parsing
and GPT-4o-mini for image summarization, saving all outputs to
the data/ directory.

Usage:
    python scripts/ingest_documents.py <pdf_path> [pdf_path2 ...]
    python scripts/ingest_documents.py --directory <dir_path>
    python scripts/ingest_documents.py --help
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pipeline import create_pipeline, IngestionPipeline
from config.settings import settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents for the DocuLens RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single PDF
    python scripts/ingest_documents.py data/raw/report.pdf

    # Process multiple PDFs
    python scripts/ingest_documents.py doc1.pdf doc2.pdf doc3.pdf

    # Process all PDFs in a directory
    python scripts/ingest_documents.py --directory data/raw/

    # Process without generating image summaries (faster, cheaper)
    python scripts/ingest_documents.py --no-summaries data/raw/report.pdf

    # Force reprocessing even if outputs exist
    python scripts/ingest_documents.py --force data/raw/report.pdf
        """,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "files",
        nargs="*",
        default=[],
        help="Path(s) to PDF file(s) to process",
    )
    input_group.add_argument(
        "-d", "--directory",
        type=str,
        help="Process all PDFs in this directory",
    )
    
    # Processing options
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip image summarization (faster, uses no OpenAI credits)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force reprocessing even if outputs already exist",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {settings.DATA_DIR})",
    )
    
    # Verbosity
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Handle the case where files is empty but not using --directory
    if not args.directory and not args.files:
        parser.error("Please provide PDF file paths or use --directory")
    
    return args


def validate_inputs(args: argparse.Namespace) -> list[Path]:
    """
    Validate input paths and return list of PDF files to process.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        List of Path objects for PDF files to process.
        
    Raises:
        SystemExit: If validation fails.
    """
    pdf_files = []
    
    if args.directory:
        dir_path = Path(args.directory)
        if not dir_path.exists():
            print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
            sys.exit(1)
        if not dir_path.is_dir():
            print(f"Error: Not a directory: {args.directory}", file=sys.stderr)
            sys.exit(1)
        
        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))
        if not pdf_files:
            print(f"Error: No PDF files found in {args.directory}", file=sys.stderr)
            sys.exit(1)
    else:
        for file_path in args.files:
            path = Path(file_path)
            if not path.exists():
                print(f"Error: File not found: {file_path}", file=sys.stderr)
                sys.exit(1)
            if not path.is_file():
                print(f"Error: Not a file: {file_path}", file=sys.stderr)
                sys.exit(1)
            if path.suffix.lower() != ".pdf":
                print(f"Warning: Skipping non-PDF file: {file_path}", file=sys.stderr)
                continue
            pdf_files.append(path)
    
    if not pdf_files:
        print("Error: No valid PDF files to process", file=sys.stderr)
        sys.exit(1)
    
    return pdf_files


def main() -> int:
    """
    Main entry point for the ingestion script.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()
    
    # Validate inputs
    pdf_files = validate_inputs(args)
    
    if not args.quiet:
        print("=" * 60)
        print("DocuLens Document Ingestion Pipeline")
        print("=" * 60)
        print(f"Files to process: {len(pdf_files)}")
        print(f"Image summaries: {'Disabled' if args.no_summaries else 'Enabled'}")
        print(f"Output directory: {args.output_dir or settings.DATA_DIR}")
        print(f"Skip existing: {not args.force}")
        print("=" * 60)
        print()
    
    # Create the pipeline
    try:
        pipeline = create_pipeline(
            summarize_images=not args.no_summaries,
            output_dir=args.output_dir,
        )
    except ValueError as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        print("\nMake sure you have configured your API keys in .env file:", file=sys.stderr)
        print("  LLAMA_CLOUD_API_KEY=llx-...", file=sys.stderr)
        print("  OPENAI_API_KEY=sk-...", file=sys.stderr)
        return 1
    
    # Process files
    successful = 0
    failed = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        if not args.quiet:
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        try:
            document = pipeline.process(
                str(pdf_path),
                save_outputs=True,
                skip_existing=not args.force,
            )
            
            if not args.quiet:
                print(f"  ✓ Pages: {document.page_count}")
                print(f"  ✓ Images: {len(document.images)}")
                print(f"  ✓ Summaries: {len(document.image_summaries)}")
                if document.is_cache_hit:
                    print(f"  ✓ Cache hit (no credits used)")
            
            successful += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            failed += 1
    
    # Summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(pdf_files)}")
        
        if successful > 0:
            output_dir = args.output_dir or settings.DATA_DIR
            print(f"\nOutputs saved to:")
            print(f"  Markdown/JSON: {output_dir}/parsed/")
            print(f"  Images: {output_dir}/images/")
            print(f"  Summaries: {output_dir}/summaries/")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
