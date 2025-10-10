# Step 1: Data Collection

This folder contains scripts for collecting data from two sources:
1. **Hugging Face Dataset** (arxiv-summarization)
2. **ArXiv API** (recent papers with PDFs)

## Prerequisites

```powershell
# Create and activate virtual environment (from project root)
python -m venv .venv
.venv\Scripts\activate

# Install required packages
pip install datasets pandas pyarrow requests feedparser
```

## Scripts Overview

### 1a. Download ArXiv Summarization Dataset
**Script:** `step1a_download_full_dataset.py`

**Purpose:** Download the ccdv/arxiv-summarization dataset from Hugging Face (~203K papers).

**Outputs:**
- `data/train.csv` - Training split (~203K papers)
- `data/validation.csv` - Validation split (~6.4K papers)
- `data/test.csv` - Test split (~6.4K papers)
- `metadata/full_stats.txt` - Dataset statistics

**Features:**
- Automatic retry logic for network issues
- Memory-efficient (no pandas conversion)
- Verification disabled to handle network interruptions
- ~7GB download, requires stable internet

**Usage:**
```powershell
cd "d:\Final Project"
.venv\Scripts\activate
python step1/step1a_download_full_dataset.py
```

### 1b. Scrape Recent ArXiv PDFs
**Script:** `step1b_scrape_arxiv_pdfs.py`

**Purpose:** Collect recent ArXiv papers in AI/ML categories with PDFs and metadata.

**Outputs:**
- `pdfs/` - Downloaded PDF files
- `arxiv_metadata.csv` - Paper metadata (title, authors, abstract, URLs, etc.)
- `logs/arxiv_scraper_*.log` - Detailed execution logs

**Features:**
- Scrapes from cs.CV, cs.AI, cs.LG, cs.CL categories
- Configurable max results (default: 10)
- Retry logic for failed downloads (up to 3 attempts)
- Skips already downloaded PDFs
- Rate limiting (2-second delay between downloads)
- Safe filename generation
- Comprehensive logging

**Usage:**
```powershell
cd "d:\Final Project"
.venv\Scripts\activate
python step1/step1b_scrape_arxiv_pdfs.py
```

**Configuration:**
Edit the `main()` function in the script to modify:
- `categories` - ArXiv categories to scrape (default: ["cs.CV", "cs.AI", "cs.LG", "cs.CL"])
- `max_results` - Total papers to fetch (default: 10)

## Quick Start

Run both scripts in sequence:

```powershell
# From project root
cd "d:\Final Project"
.venv\Scripts\activate

# Download Hugging Face dataset (large, takes time)
python step1/step1a_download_full_dataset.py

# Scrape recent ArXiv papers (quick)
python step1/step1b_scrape_arxiv_pdfs.py
```

Or use the batch file:
```powershell
cd "d:\Final Project\step1"
.\run_step1.bat
```

## Project Structure After Completion

```
Final Project/
├── data/                          # Hugging Face dataset
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── metadata/                      # Dataset statistics
│   └── full_stats.txt
├── pdfs/                          # ArXiv PDFs
│   ├── 2510.03215v1_Cache-to-Cache....pdf
│   ├── 2510.03216v1_Wave-GMS....pdf
│   └── ...
├── logs/                          # Scraping logs
│   ├── arxiv_scraper_20251006_185052.log
│   └── ...
├── arxiv_metadata.csv            # ArXiv paper metadata
└── step1/
    ├── step1a_download_full_dataset.py
    ├── step1a_download_small_sample.py
    ├── step1b_scrape_arxiv_pdfs.py
    ├── run_step1.bat
    └── README.md
```

## Troubleshooting

### Issue: `ImportError: cannot import name 'load_dataset'`
**Solution:** Make sure you don't have a local folder named `datasets` that shadows the Python package.

### Issue: Dataset download fails with network errors
**Solution:** The script has retry logic and uses `verification_mode='no_checks'`. If download consistently fails:
1. Check your internet connection
2. Try downloading during off-peak hours
3. Ensure you have ~10GB free disk space

### Issue: ArXiv returns no papers
**Solution:** The current version doesn't use date filters to maximize results.

### Issue: PDF download times out
**Solution:** The script retries up to 3 times per PDF. If it fails, re-run the script - it skips already downloaded files.

## Dataset Information

### arxiv-summarization (Hugging Face)
- **Source:** https://huggingface.co/datasets/ccdv/arxiv-summarization
- **Papers:** ~215K scientific papers from ArXiv
- **Fields:** article (full text), abstract (summary)
- **Use case:** Training summarization models

### ArXiv API Papers
- **Source:** http://export.arxiv.org/api/
- **Categories:** Computer Science (CV, AI, LG, CL)
- **Freshness:** Most recent submissions
- **Use case:** Testing on latest research, custom corpus

## Next Steps

After completing Step 1:
- Verify data integrity
- Explore sample papers
- Proceed to Step 2: Data Preprocessing

## Next Steps

After completing Step 1, you'll have:
1. A large dataset of pre-existing summaries for training/testing
2. Recent ArXiv PDFs for real-world document processing
3. Metadata for all collected papers

These will be used in subsequent steps for:
- PDF text extraction
- OCR for scanned documents
- Semantic embedding generation
- Model training and evaluation

