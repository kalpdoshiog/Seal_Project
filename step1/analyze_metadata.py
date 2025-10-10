#!/usr/bin/env python3
"""
Analyze ArXiv Metadata
Provides insights into the collected papers
"""

import csv
from pathlib import Path
from collections import Counter
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_FILE = BASE_DIR / "arxiv_metadata.csv"

def analyze_metadata():
    """Analyze the metadata CSV and display statistics."""

    if not METADATA_FILE.exists():
        print("❌ Metadata file not found!")
        return

    papers = []
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        papers = list(reader)

    if not papers:
        print("❌ No papers found in metadata!")
        return

    print("=" * 70)
    print(f"ArXiv Metadata Analysis - {len(papers):,} Papers")
    print("=" * 70)
    print()

    # Category distribution
    print("📚 Papers by Category:")
    all_categories = []
    for paper in papers:
        cats = paper.get('categories', '').split(', ')
        all_categories.extend([c.strip() for c in cats if c.strip()])

    cat_counts = Counter(all_categories)
    for cat, count in cat_counts.most_common(15):
        bar = "█" * int(count / 100)
        print(f"   {cat:12} {count:5,} {bar}")
    print()

    # Year distribution
    print("📅 Papers by Year:")
    years = []
    for paper in papers:
        pub_date = paper.get('published', '')
        if pub_date:
            try:
                year = pub_date.split('-')[0]
                years.append(year)
            except:
                pass

    year_counts = Counter(years)
    for year, count in sorted(year_counts.items(), reverse=True)[:10]:
        bar = "█" * int(count / 100)
        print(f"   {year}: {count:5,} {bar}")
    print()

    # Download status
    downloaded = sum(1 for p in papers if p.get('downloaded', '').lower() == 'true')
    pending = len(papers) - downloaded

    print("📥 Download Status:")
    print(f"   Total: {len(papers):,}")
    print(f"   Downloaded: {downloaded:,} ({downloaded/len(papers)*100:.1f}%)")
    print(f"   Pending: {pending:,} ({pending/len(papers)*100:.1f}%)")
    print()

    # Top authors (by paper count)
    print("👥 Most Prolific Authors (in this dataset):")
    all_authors = []
    for paper in papers:
        authors = paper.get('authors', '').split(', ')
        all_authors.extend([a.strip() for a in authors if a.strip()])

    author_counts = Counter(all_authors)
    for author, count in author_counts.most_common(10):
        if len(author) > 40:
            author = author[:37] + "..."
        print(f"   {count:3} papers - {author}")
    print()

    # Sample recent papers
    print("🆕 Sample of Most Recent Papers:")
    sorted_papers = sorted(papers, key=lambda x: x.get('published', ''), reverse=True)
    for i, paper in enumerate(sorted_papers[:5], 1):
        title = paper.get('title', 'N/A')
        if len(title) > 65:
            title = title[:62] + "..."
        pub_date = paper.get('published', 'N/A')[:10]
        print(f"   {i}. [{pub_date}] {title}")

    print()
    print("=" * 70)

if __name__ == "__main__":
    analyze_metadata()
#!/usr/bin/env python3
"""
Progress Monitor for ArXiv PDF Downloads
Displays real-time statistics on download progress
"""

import csv
import time
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_FILE = BASE_DIR / "arxiv_metadata.csv"
PDFS_DIR = BASE_DIR / "pdfs"
LOGS_DIR = BASE_DIR / "logs"

def get_latest_log():
    """Get the most recent log file."""
    log_files = list(LOGS_DIR.glob("arxiv_scraper_*.log"))
    if not log_files:
        return None
    return max(log_files, key=lambda x: x.stat().st_mtime)

def read_metadata():
    """Read current metadata and count statistics."""
    if not METADATA_FILE.exists():
        return None

    total = 0
    downloaded = 0
    failed = 0

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get('downloaded', '').lower() == 'true':
                downloaded += 1
            elif row.get('local_path', '') == '' and row.get('downloaded', '').lower() == 'false':
                failed += 1

    return {
        'total': total,
        'downloaded': downloaded,
        'pending': total - downloaded - failed,
        'failed': failed
    }

def count_pdf_files():
    """Count actual PDF files in directory."""
    if not PDFS_DIR.exists():
        return 0
    return len(list(PDFS_DIR.glob("*.pdf")))

def get_log_tail(n=5):
    """Get last n lines from the latest log."""
    log_file = get_latest_log()
    if not log_file:
        return []

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-n:]]
    except:
        return []

def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def main():
    """Display progress statistics."""
    print("=" * 70)
    print("ArXiv PDF Download Progress Monitor")
    print("=" * 70)
    print()

    # Metadata stats
    stats = read_metadata()
    if stats:
        print("📊 Metadata Statistics:")
        print(f"   Total papers in database: {stats['total']:,}")
        print(f"   ✅ Downloaded: {stats['downloaded']:,}")
        print(f"   ⏳ Pending: {stats['pending']:,}")
        print(f"   ❌ Failed: {stats['failed']:,}")

        if stats['total'] > 0:
            progress_pct = (stats['downloaded'] / stats['total']) * 100
            print(f"   📈 Progress: {progress_pct:.1f}%")

            # Estimate time remaining
            if stats['downloaded'] > 0:
                avg_time_per_paper = 3  # seconds (conservative estimate)
                remaining_time = stats['pending'] * avg_time_per_paper
                print(f"   ⏱️  Estimated time remaining: {format_time(remaining_time)}")
        print()

    # Physical file count
    pdf_count = count_pdf_files()
    print(f"📁 PDF Files on Disk: {pdf_count:,}")
    print()

    # Latest log activity
    print("📝 Latest Log Activity:")
    log_lines = get_log_tail(8)
    if log_lines:
        for line in log_lines:
            # Shorten long lines
            if len(line) > 100:
                line = line[:97] + "..."
            print(f"   {line}")
    else:
        print("   No log file found")
    print()

    # Log file info
    latest_log = get_latest_log()
    if latest_log:
        log_size = latest_log.stat().st_size / 1024 / 1024  # MB
        log_modified = datetime.fromtimestamp(latest_log.stat().st_mtime)
        print(f"📄 Log File: {latest_log.name}")
        print(f"   Size: {log_size:.2f} MB")
        print(f"   Last Modified: {log_modified.strftime('%Y-%m-%d %H:%M:%S')}")

    print()
    print("=" * 70)
    print("💡 Tip: Run this script periodically to check progress")
    print("   The download saves checkpoints every 100 papers")
    print("=" * 70)

if __name__ == "__main__":
    main()

