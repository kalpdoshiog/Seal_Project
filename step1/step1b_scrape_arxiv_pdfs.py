#!/usr/bin/env python3
"""
Step 1b: Scrape ArXiv PDFs and Metadata
Purpose: Download recent ArXiv papers from specified categories
Outputs: pdfs/, logs/, arxiv_metadata.csv
"""

import os
import sys
import csv
import time
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlencode

# ----------------------------
# Paths (project-level)
BASE_DIR = Path(__file__).resolve().parents[1]
PDFS_DIR = BASE_DIR / "pdfs"
LOGS_DIR = BASE_DIR / "logs"
METADATA_FILE = BASE_DIR / "arxiv_metadata.csv"

PDFS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Logging Setup
LOG_FILE = LOGS_DIR / f"arxiv_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
# Multiple AI-related categories to get 10,000+ papers
CATEGORIES = [
    "cs.AI",      # Artificial Intelligence
    "cs.LG",      # Machine Learning
    "cs.CL",      # Computation and Language (NLP)
    "cs.CV",      # Computer Vision
    "cs.NE",      # Neural and Evolutionary Computing
    "stat.ML"     # Machine Learning (Statistics)
]
MAX_RESULTS_PER_CATEGORY = 3000  # Get ~3000 per category = 18,000+ total papers
BATCH_SIZE = 500  # ArXiv API returns max 2000 per query, we'll use batches
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds
REQUEST_DELAY = 3  # seconds between API calls to respect rate limits

# ----------------------------
def sanitize_filename(text, max_length=60):
    """Create a safe filename from text."""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '')

    # Truncate and clean up
    text = text.strip()[:max_length]
    return text

def query_arxiv(category, max_results=10, start_index=0):
    """
    Query ArXiv API for papers in a specific category.
    Returns list of paper metadata dictionaries.
    """
    params = {
        'search_query': f'cat:{category}',
        'start': start_index,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    url = f"{ARXIV_API_BASE}?{urlencode(params)}"
    logger.info(f"Querying ArXiv API: {category} (start={start_index}, max={max_results})")

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return parse_arxiv_response(response.text)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to query ArXiv after {RETRY_ATTEMPTS} attempts")
                return []

    return []

def parse_arxiv_response(xml_text):
    """Parse ArXiv API XML response."""
    papers = []

    try:
        root = ET.fromstring(xml_text)
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}

        entries = root.findall('atom:entry', ns)
        logger.info(f"Found {len(entries)} papers in response")

        for entry in entries:
            try:
                # Extract paper ID from the id URL
                id_url = entry.find('atom:id', ns).text
                paper_id = id_url.split('/abs/')[-1]

                # Extract basic metadata
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')

                # Extract authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)
                authors_str = ', '.join(authors)

                # Extract dates
                published = entry.find('atom:published', ns).text
                updated = entry.find('atom:updated', ns).text

                # Extract categories
                categories = []
                for cat in entry.findall('atom:category', ns):
                    term = cat.get('term')
                    if term:
                        categories.append(term)
                categories_str = ', '.join(categories)

                # PDF URL
                pdf_url = f"http://arxiv.org/pdf/{paper_id}"
                arxiv_url = f"http://arxiv.org/abs/{paper_id}"

                papers.append({
                    'id': paper_id,
                    'title': title,
                    'authors': authors_str,
                    'abstract': summary,
                    'categories': categories_str,
                    'published': published,
                    'updated': updated,
                    'pdf_url': pdf_url,
                    'arxiv_url': arxiv_url
                })

            except Exception as e:
                logger.error(f"Error parsing entry: {e}")
                continue

    except Exception as e:
        logger.error(f"Error parsing XML response: {e}")

    return papers

def download_pdf(paper, output_dir):
    """
    Download a single PDF with retry logic.
    Returns (success: bool, local_path: str or None)
    """
    paper_id = paper['id']
    title = paper['title']
    pdf_url = paper['pdf_url']

    # Create filename
    safe_title = sanitize_filename(title)
    filename = f"{paper_id}_{safe_title}.pdf"
    filepath = output_dir / filename

    # Skip if already downloaded
    if filepath.exists():
        logger.info(f"Already exists: {filename}")
        return True, str(filepath.relative_to(BASE_DIR))

    logger.info(f"Downloading: {filename}")

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()

            # Write PDF to file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Successfully downloaded: {filename}")
            return True, str(filepath.relative_to(BASE_DIR))

        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1}/{RETRY_ATTEMPTS} failed for {paper_id}: {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to download {paper_id} after {RETRY_ATTEMPTS} attempts")
                return False, None

    return False, None

def load_existing_metadata():
    """Load existing metadata CSV if it exists."""
    if not METADATA_FILE.exists():
        return {}

    existing = {}
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row['id']] = row
        logger.info(f"Loaded {len(existing)} existing metadata entries")
    except Exception as e:
        logger.warning(f"Could not load existing metadata: {e}")

    return existing

def save_metadata(papers_dict):
    """Save all metadata to CSV."""
    if not papers_dict:
        logger.warning("No metadata to save")
        return

    fieldnames = ['id', 'title', 'authors', 'abstract', 'categories',
                  'published', 'updated', 'pdf_url', 'arxiv_url',
                  'downloaded', 'local_path']

    try:
        with open(METADATA_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Sort by ID for consistency
            for paper_id in sorted(papers_dict.keys()):
                writer.writerow(papers_dict[paper_id])

        logger.info(f"Saved metadata for {len(papers_dict)} papers to {METADATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")

def query_arxiv_batch(category, max_total_results, batch_size=500):
    """
    Query ArXiv API in batches to get large number of papers.
    Returns list of all paper metadata dictionaries.
    """
    all_papers = []
    start_index = 0

    while len(all_papers) < max_total_results:
        # Calculate how many to fetch in this batch
        remaining = max_total_results - len(all_papers)
        current_batch_size = min(batch_size, remaining)

        logger.info(f"Fetching batch: start={start_index}, size={current_batch_size}, total so far={len(all_papers)}")

        papers = query_arxiv(category, max_results=current_batch_size, start_index=start_index)

        if not papers:
            logger.info(f"No more papers found for {category}")
            break

        all_papers.extend(papers)
        start_index += len(papers)

        # If we got fewer papers than requested, we've reached the end
        if len(papers) < current_batch_size:
            logger.info(f"Reached end of available papers for {category}")
            break

        # Rate limiting between batches
        time.sleep(REQUEST_DELAY)

    return all_papers

def main():
    """Main scraping workflow."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("ArXiv PDF Scraper Started - Large Scale Download")
    logger.info(f"Target: 10,000+ AI-related papers")
    logger.info(f"Categories: {', '.join(CATEGORIES)}")
    logger.info(f"Max results per category: {MAX_RESULTS_PER_CATEGORY}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info("=" * 60)

    # Load existing metadata
    all_papers = load_existing_metadata()
    initial_count = len(all_papers)

    # Query ArXiv for each category using batch processing
    new_papers = []
    for category in CATEGORIES:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing category: {category}")
        logger.info(f"{'=' * 60}")

        papers = query_arxiv_batch(category, max_total_results=MAX_RESULTS_PER_CATEGORY, batch_size=BATCH_SIZE)
        logger.info(f"Retrieved {len(papers)} total papers from {category}")

        # Add new papers to our collection
        category_new = 0
        for paper in papers:
            if paper['id'] not in all_papers:
                new_papers.append(paper)
                all_papers[paper['id']] = {
                    **paper,
                    'downloaded': False,
                    'local_path': ''
                }
                category_new += 1

        logger.info(f"New papers from {category}: {category_new}")
        logger.info(f"Running total of new papers: {len(new_papers)}")

        # Respect API rate limits between categories
        time.sleep(REQUEST_DELAY)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Metadata Collection Complete")
    logger.info(f"Total unique papers found: {len(new_papers)}")
    logger.info(f"{'=' * 60}")

    # Save metadata before starting downloads
    save_metadata(all_papers)
    logger.info("Metadata saved. Starting PDF downloads...")

    # Download PDFs
    download_success = 0
    download_failed = 0
    download_skipped = 0

    for i, paper in enumerate(new_papers, 1):
        logger.info(f"\n[{i}/{len(new_papers)}] Processing: {paper['id']}")
        success, local_path = download_pdf(paper, PDFS_DIR)

        if success:
            all_papers[paper['id']]['downloaded'] = True
            all_papers[paper['id']]['local_path'] = local_path
            download_success += 1
        else:
            download_failed += 1

        # Save metadata periodically (every 100 downloads)
        if i % 100 == 0:
            save_metadata(all_papers)
            logger.info(f"Progress checkpoint: {i}/{len(new_papers)} processed")

        # Rate limiting
        if i < len(new_papers):
            time.sleep(REQUEST_DELAY)

    # Final save
    save_metadata(all_papers)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("ArXiv PDF Scraper Completed")
    logger.info("=" * 60)
    logger.info(f"Total papers in database: {len(all_papers)} (was {initial_count})")
    logger.info(f"New papers found: {len(new_papers)}")
    logger.info(f"Successfully downloaded: {download_success}")
    logger.info(f"Failed downloads: {download_failed}")
    logger.info(f"Elapsed time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Metadata file: {METADATA_FILE}")
    logger.info(f"PDFs directory: {PDFS_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
