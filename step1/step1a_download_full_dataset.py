#!/usr/bin/env python3
"""
Step 1a: Download arxiv-summarization dataset from Hugging Face
Outputs: datasets/{train,validation,test}.csv, metadata/full_stats.txt
"""

from pathlib import Path
from datetime import datetime
import time

# Import datasets package with error handling
try:
    from datasets import load_dataset
except ImportError as e:
    print(f"Error importing datasets: {e}")
    print("Make sure you have installed the datasets package:")
    print("pip install datasets")
    print("\nNote: If you have a local 'datasets' folder, it may be shadowing the package.")
    exit(1)

# ----------------------------
# Paths (project-level)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"  # Changed from "datasets" to avoid naming conflict
META_DIR = BASE_DIR / "metadata"
DATA_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
def download_with_retry(dataset_name, max_retries=3, retry_delay=30):
    """Download dataset with retry logic for network issues."""
    for attempt in range(max_retries):
        try:
            print(f"Downloading {dataset_name} (attempt {attempt + 1}/{max_retries})...")
            return load_dataset(dataset_name, verification_mode='no_checks')
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("All download attempts failed.")
                raise

def main():
    dataset = download_with_retry("ccdv/arxiv-summarization")

    # Save splits to CSV without loading entire splits into pandas
    print("Saving CSV files...")
    dataset["train"].to_csv(str(DATA_DIR / "train.csv"))
    dataset["validation"].to_csv(str(DATA_DIR / "validation.csv"))
    dataset["test"].to_csv(str(DATA_DIR / "test.csv"))
    print(f"CSV files saved in {DATA_DIR}/")

    # Save dataset stats without materializing to pandas
    stats = {
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
        "columns": list(dataset["train"].features.keys()),
        "download_time": datetime.now().isoformat(timespec="seconds"),
    }

    with open(META_DIR / "full_stats.txt", "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print(f"Dataset stats saved in {META_DIR}/full_stats.txt")
    print("Step 1a completed successfully!")

if __name__ == "__main__":
    main()
