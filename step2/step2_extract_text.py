#!/usr/bin/env python3
"""
Step 2: PDF Text Extraction with GPU Acceleration
Purpose: Extract text from PDFs (digital + OCR for scanned docs)
Supports: PyMuPDF (fast), PaddleOCR (GPU-accelerated), metadata preservation
"""

import os
import sys
import json
import logging
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import re

# ----------------------------
# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PDFS_DIR = BASE_DIR / "pdfs"
EXTRACTED_DIR = BASE_DIR / "extracted_text"
METADATA_FILE = BASE_DIR / "arxiv_metadata.csv"
OUTPUT_METADATA = BASE_DIR / "extraction_metadata.csv"
LOGS_DIR = BASE_DIR / "logs"

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Logging Setup
LOG_FILE = LOGS_DIR / f"text_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
MAX_WORKERS = 4  # Parallel processing threads
MIN_TEXT_LENGTH = 100  # Minimum characters to consider valid extraction
OCR_ENABLED = False  # Set to True to enable OCR for scanned PDFs (requires PaddleOCR)
CLEAN_TEXT = True  # Enable text cleaning and normalization

# Try to import PaddleOCR if OCR is enabled
ocr_reader = None
if OCR_ENABLED:
    try:
        from paddleocr import PaddleOCR
        ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
        logger.info("✓ PaddleOCR initialized with GPU support")
    except ImportError:
        logger.warning("PaddleOCR not installed. OCR will be disabled.")
        logger.warning("Install with: pip install paddleocr paddlepaddle-gpu")
        OCR_ENABLED = False

# ----------------------------
def clean_text(text):
    """
    Clean and normalize extracted text.
    Removes headers, footers, extra whitespace, normalizes punctuation.
    """
    if not CLEAN_TEXT:
        return text

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common PDF artifacts
    text = re.sub(r'\x00', '', text)  # Remove null characters
    text = re.sub(r'[\x01-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)  # Remove control characters

    # Normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double

    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove URLs (optional - keep if needed for citations)
    # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Normalize punctuation spacing
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([^\s\d])', r'\1 \2', text)

    # Remove hyphenation at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    return text.strip()

def extract_text_with_ocr(pdf_path):
    """
    Extract text using PaddleOCR for scanned PDFs.
    GPU-accelerated OCR.
    """
    if not ocr_reader:
        return None

    try:
        # Convert PDF to images and run OCR
        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
            img_data = pix.tobytes("png")

            # Run OCR
            result = ocr_reader.ocr(img_data, cls=True)

            # Extract text from OCR results
            if result and result[0]:
                page_text = ' '.join([line[1][0] for line in result[0]])
                all_text.append(page_text)

        doc.close()
        return '\n\n'.join(all_text)

    except Exception as e:
        logger.error(f"OCR failed for {pdf_path.name}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF with fallback to OCR.
    Returns: dict with extracted text and metadata
    """
    result = {
        'pdf_path': str(pdf_path),
        'success': False,
        'text': '',
        'num_pages': 0,
        'num_characters': 0,
        'num_words': 0,
        'extraction_method': 'pymupdf',
        'error': None
    }

    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        result['num_pages'] = len(doc)

        # Extract text from all pages
        text_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_pages.append(text)

        # Combine all pages
        full_text = '\n\n'.join(text_pages)
        doc.close()

        # Check if extraction was successful
        if len(full_text.strip()) < MIN_TEXT_LENGTH:
            # Try OCR if enabled
            if OCR_ENABLED:
                logger.info(f"Low text content, trying OCR for {pdf_path.name}")
                ocr_text = extract_text_with_ocr(pdf_path)
                if ocr_text and len(ocr_text.strip()) >= MIN_TEXT_LENGTH:
                    full_text = ocr_text
                    result['extraction_method'] = 'paddleocr'
                    logger.info(f"✓ OCR extracted {len(ocr_text.split())} words from {pdf_path.name}")
                else:
                    result['success'] = False
                    result['error'] = 'Insufficient text extracted (OCR also failed)'
                    return result
            else:
                result['success'] = False
                result['error'] = 'Insufficient text extracted (possible scanned PDF)'
                logger.warning(f"Low text content in {pdf_path.name}: {len(full_text)} chars")
                return result

        # Clean the text
        full_text = clean_text(full_text)

        result['text'] = full_text
        result['num_characters'] = len(full_text)
        result['num_words'] = len(full_text.split())
        result['success'] = True
        logger.info(f"✓ Extracted {result['num_words']:,} words from {pdf_path.name}")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Failed to extract from {pdf_path.name}: {e}")

    return result

def save_extracted_text(paper_id, text, metadata):
    """Save extracted text to file with JSON metadata."""
    # Create text file
    text_file = EXTRACTED_DIR / f"{paper_id}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text)

    # Create metadata sidecar
    meta_file = EXTRACTED_DIR / f"{paper_id}.meta.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return text_file

def load_arxiv_metadata():
    """Load ArXiv metadata to get paper details."""
    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row['id']] = row
    return metadata

def process_single_pdf(pdf_path, arxiv_metadata):
    """Process a single PDF file."""
    paper_id = pdf_path.stem.split('_')[0]  # Extract ID from filename

    # Check if already processed
    text_file = EXTRACTED_DIR / f"{paper_id}.txt"
    if text_file.exists():
        logger.info(f"⊘ Already processed: {paper_id}")
        return {
            'paper_id': paper_id,
            'status': 'skipped',
            'reason': 'already_processed'
        }

    logger.info(f"Processing: {pdf_path.name}")

    # Extract text
    extraction = extract_text_from_pdf(pdf_path)

    if not extraction['success']:
        return {
            'paper_id': paper_id,
            'status': 'failed',
            'reason': extraction['error']
        }

    # Get ArXiv metadata
    paper_meta = arxiv_metadata.get(paper_id, {})

    # Prepare full metadata
    full_metadata = {
        'paper_id': paper_id,
        'title': paper_meta.get('title', 'Unknown'),
        'authors': paper_meta.get('authors', 'Unknown'),
        'categories': paper_meta.get('categories', 'Unknown'),
        'published': paper_meta.get('published', 'Unknown'),
        'extraction_date': datetime.now().isoformat(),
        'num_pages': extraction['num_pages'],
        'num_characters': extraction['num_characters'],
        'num_words': extraction['num_words'],
        'extraction_method': extraction['extraction_method']
    }

    # Save extracted text
    save_extracted_text(paper_id, extraction['text'], full_metadata)

    return {
        'paper_id': paper_id,
        'status': 'success',
        'num_words': extraction['num_words'],
        'num_pages': extraction['num_pages']
    }

def main():
    """Main extraction workflow."""
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("PDF Text Extraction Started")
    logger.info(f"OCR Enabled: {OCR_ENABLED}")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info("=" * 70)

    # Load ArXiv metadata
    logger.info("Loading ArXiv metadata...")
    arxiv_metadata = load_arxiv_metadata()
    logger.info(f"Loaded metadata for {len(arxiv_metadata):,} papers")

    # Get list of PDFs to process
    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files):,} PDF files")

    if not pdf_files:
        logger.warning("No PDF files found!")
        return

    # Process PDFs
    results = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_words': 0,
        'total_pages': 0
    }

    logger.info(f"\nStarting extraction with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_pdf, pdf, arxiv_metadata): pdf
            for pdf in pdf_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if result['status'] == 'success':
                results['success'] += 1
                results['total_words'] += result.get('num_words', 0)
                results['total_pages'] += result.get('num_pages', 0)
            elif result['status'] == 'failed':
                results['failed'] += 1
            elif result['status'] == 'skipped':
                results['skipped'] += 1

            # Progress update
            if i % 50 == 0 or i == len(pdf_files):
                logger.info(f"Progress: {i}/{len(pdf_files)} | ✓ {results['success']} | ✗ {results['failed']} | ⊘ {results['skipped']}")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 70)
    logger.info("Text Extraction Completed")
    logger.info("=" * 70)
    logger.info(f"Processed: {len(pdf_files):,} PDFs")
    logger.info(f"Success: {results['success']:,}")
    logger.info(f"Failed: {results['failed']:,}")
    logger.info(f"Skipped: {results['skipped']:,}")
    logger.info(f"Total Words Extracted: {results['total_words']:,}")
    logger.info(f"Total Pages: {results['total_pages']:,}")
    logger.info(f"Elapsed Time: {elapsed/60:.1f} minutes")
    logger.info(f"Output Directory: {EXTRACTED_DIR}")
    logger.info(f"Log File: {LOG_FILE}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
