#!/usr/bin/env python3
"""
Fast Parallel PDF Downloader for ArXiv
Uses concurrent downloads to maximize speed
"""

import os
import csv
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys

# ----------------------------
# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PDFS_DIR = BASE_DIR / "pdfs"
LOGS_DIR = BASE_DIR / "logs"
METADATA_FILE = BASE_DIR / "arxiv_metadata.csv"

PDFS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Logging Setup
LOG_FILE = LOGS_DIR / f"fast_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
MAX_WORKERS = 50  # Increased from 20 to 50 for maximum speed
TIMEOUT = 60  # Reduced from 120 to 60 seconds
RETRY_ATTEMPTS = 2  # Reduced from 3 to 2 for faster failures
CHUNK_SIZE = 16384  # Increased from 8192 for faster writes

# Progress tracking
progress_lock = Lock()
stats = {
    'downloaded': 0,
    'failed': 0,
    'skipped': 0,
    'total': 0
}

# Session with connection pooling for better performance
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=0
)
session.mount('http://', adapter)
session.mount('https://', adapter)

def sanitize_filename(text, max_length=60):
    """Create a safe filename from text."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '')
    text = text.strip()[:max_length]
    return text

def download_single_pdf(paper_info):
    """
    Download a single PDF file.
    Returns: (paper_id, success, filepath or error_msg)
    """
    paper_id = paper_info['id']
    title = paper_info.get('title', '')
    pdf_url = paper_info['pdf_url']

    # Create filename
    safe_title = sanitize_filename(title)
    if safe_title:
        filename = f"{paper_id}_{safe_title}.pdf"
    else:
        filename = f"{paper_id}.pdf"

    filepath = PDFS_DIR / filename

    # Skip if already exists and has content
    if filepath.exists() and filepath.stat().st_size > 10000:  # At least 10KB
        with progress_lock:
            stats['skipped'] += 1
        return paper_id, True, str(filepath.relative_to(BASE_DIR))

    # Download with retries
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = session.get(pdf_url, timeout=TIMEOUT, stream=True)
            response.raise_for_status()

            # Write to file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

            # Verify file size
            if filepath.stat().st_size > 1000:  # At least 1KB
                with progress_lock:
                    stats['downloaded'] += 1
                    current = stats['downloaded'] + stats['failed'] + stats['skipped']
                    if current % 50 == 0:  # Progress update every 50 papers
                        logger.info(f"Progress: {current}/{stats['total']} | "
                                  f"✓ {stats['downloaded']} | "
                                  f"✗ {stats['failed']} | "
                                  f"⊘ {stats['skipped']}")
                return paper_id, True, str(filepath.relative_to(BASE_DIR))
            else:
                filepath.unlink(missing_ok=True)
                raise ValueError("Downloaded file too small")

        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                with progress_lock:
                    stats['failed'] += 1
                logger.warning(f"Failed to download {paper_id}: {str(e)[:100]}")
                return paper_id, False, str(e)[:200]

    return paper_id, False, "Max retries exceeded"

def load_metadata():
    """Load metadata CSV to get list of papers to download."""
    if not METADATA_FILE.exists():
        logger.error(f"Metadata file not found: {METADATA_FILE}")
        return []

    papers = []
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                papers.append(row)
        logger.info(f"Loaded {len(papers)} papers from metadata")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return []

    return papers

def update_metadata_with_results(papers, results):
    """Update metadata CSV with download results."""
    # Create a results dictionary for quick lookup
    results_dict = {paper_id: (success, path) for paper_id, success, path in results}

    # Update papers with download status
    for paper in papers:
        paper_id = paper['id']
        if paper_id in results_dict:
            success, path = results_dict[paper_id]
            paper['downloaded'] = 'yes' if success else 'no'
            if success:
                paper['local_path'] = path
            else:
                paper['local_path'] = ''
        else:
            # Paper wasn't processed (shouldn't happen)
            paper['downloaded'] = paper.get('downloaded', 'no')
            paper['local_path'] = paper.get('local_path', '')

    # Save updated metadata
    fieldnames = ['id', 'title', 'authors', 'abstract', 'categories',
                  'published', 'updated', 'pdf_url', 'arxiv_url',
                  'downloaded', 'local_path']

    try:
        with open(METADATA_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for paper in papers:
                writer.writerow(paper)
        logger.info(f"✓ Metadata updated with download results")
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")

def main():
    """Main parallel download workflow."""
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("🚀 FAST PARALLEL PDF DOWNLOADER")
    logger.info(f"Workers: {MAX_WORKERS} parallel downloads")
    logger.info(f"Timeout: {TIMEOUT}s per download")
    logger.info("=" * 70)

    # Load papers from metadata
    papers = load_metadata()
    if not papers:
        logger.error("No papers to download. Run step1b first to generate metadata.")
        return

    stats['total'] = len(papers)
    logger.info(f"Total papers to process: {stats['total']}")

    # Filter papers that need downloading (optional - can download all)
    papers_to_download = [p for p in papers if p.get('downloaded') != 'yes' or not p.get('local_path')]

    if not papers_to_download:
        logger.info("All papers already downloaded!")
        return

    logger.info(f"Papers to download: {len(papers_to_download)}")
    logger.info(f"Already downloaded: {len(papers) - len(papers_to_download)}")
    logger.info(f"\n🔄 Starting parallel download with {MAX_WORKERS} workers...\n")

    # Parallel download
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_paper = {
            executor.submit(download_single_pdf, paper): paper
            for paper in papers_to_download
        }

        # Process completed downloads
        for future in as_completed(future_to_paper):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

    # Update metadata with results
    logger.info("\n📝 Updating metadata...")
    update_metadata_with_results(papers, results)

    # Final statistics
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("✅ DOWNLOAD COMPLETE")
    logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    logger.info(f"Total papers: {stats['total']}")
    logger.info(f"✓ Downloaded: {stats['downloaded']}")
    logger.info(f"⊘ Skipped (already exists): {stats['skipped']}")
    logger.info(f"✗ Failed: {stats['failed']}")

    if stats['downloaded'] > 0:
        rate = stats['downloaded'] / elapsed
        logger.info(f"⚡ Download rate: {rate:.2f} papers/second")

    logger.info(f"📁 PDFs location: {PDFS_DIR}")
    logger.info(f"📄 Metadata: {METADATA_FILE}")
    logger.info(f"🧾 Log file: {LOG_FILE}")
    logger.info("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
