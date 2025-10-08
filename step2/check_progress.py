#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 Progress Checker
Run this anytime to check extraction status and progress
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PDFS_DIR = BASE_DIR / "pdfs"
EXTRACTED_DIR = BASE_DIR / "extracted_text"
TABLES_DIR = BASE_DIR / "extracted_tables"
LOGS_DIR = BASE_DIR / "logs"
EXTRACTION_LOG = BASE_DIR / "extraction_results.csv"

def get_color(percentage):
    """Get color based on completion percentage"""
    if percentage < 25:
        return "🔴"
    elif percentage < 50:
        return "🟡"
    elif percentage < 75:
        return "🟠"
    elif percentage < 100:
        return "🟢"
    else:
        return "✅"

def check_progress():
    """Check and display extraction progress"""

    print("=" * 80)
    print("📊 STEP 2: PDF TEXT EXTRACTION - PROGRESS CHECK")
    print("=" * 80)
    print(f"⏰ Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Count PDFs
    if PDFS_DIR.exists():
        total_pdfs = len(list(PDFS_DIR.glob("*.pdf")))
        print(f"📥 Total PDFs Available: {total_pdfs:,}")
    else:
        print("❌ PDFs folder not found!")
        total_pdfs = 0

    # Count extracted files
    if EXTRACTED_DIR.exists():
        txt_files = list(EXTRACTED_DIR.glob("*.txt"))
        json_files = list(EXTRACTED_DIR.glob("*.meta.json"))
        extracted_count = len(txt_files)
        metadata_count = len(json_files)

        # Get extraction start time from oldest file
        if txt_files:
            oldest_file = min(txt_files, key=lambda x: x.stat().st_mtime)
            start_time = datetime.fromtimestamp(oldest_file.stat().st_mtime)
        else:
            start_time = None
    else:
        print("⚠️  extracted_text folder doesn't exist yet - extraction not started or just starting")
        extracted_count = 0
        metadata_count = 0
        start_time = None

    # Count tables
    if TABLES_DIR.exists():
        table_files = list(TABLES_DIR.glob("*_tables.json"))
        tables_count = len(table_files)
    else:
        tables_count = 0

    # Calculate progress
    if total_pdfs > 0:
        progress_percentage = (extracted_count / total_pdfs) * 100
        remaining = total_pdfs - extracted_count
    else:
        progress_percentage = 0
        remaining = 0

    print()
    print("-" * 80)
    print("📈 EXTRACTION PROGRESS")
    print("-" * 80)

    # Progress bar
    bar_length = 50
    filled_length = int(bar_length * progress_percentage / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    status_emoji = get_color(progress_percentage)
    print(f"\n{status_emoji} Progress: [{bar}] {progress_percentage:.1f}%")
    print()
    print(f"✅ Extracted: {extracted_count:,} / {total_pdfs:,} PDFs")
    print(f"⏳ Remaining: {remaining:,} PDFs")
    print(f"📄 Metadata files: {metadata_count:,}")
    print(f"📊 Tables extracted: {tables_count:,}")

    # DYNAMIC time estimation based on actual speed
    if extracted_count > 0 and remaining > 0 and start_time:
        elapsed_time = (datetime.now() - start_time).total_seconds()
        elapsed_minutes = elapsed_time / 60

        # Calculate ACTUAL speed
        actual_speed = extracted_count / elapsed_minutes if elapsed_minutes > 0 else 0

        if actual_speed > 0:
            minutes_remaining = remaining / actual_speed
            hours_remaining = minutes_remaining / 60

            # Calculate ETA
            eta = datetime.now().timestamp() + (minutes_remaining * 60)
            eta_datetime = datetime.fromtimestamp(eta)

            print()
            print(f"⚡ Current Speed: {actual_speed:.1f} PDFs/minute (LIVE)")
            print(f"⏱️  Estimated Time Remaining: {hours_remaining:.1f} hours ({minutes_remaining:.0f} minutes)")
            print(f"🎯 Estimated Completion: {eta_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   (Elapsed: {elapsed_minutes:.1f} minutes)")
        else:
            print()
            print(f"⏱️  Calculating speed... (just started)")
    elif extracted_count > 0 and remaining > 0:
        # Fallback if can't determine start time
        avg_speed = 80  # default estimate
        minutes_remaining = remaining / avg_speed
        hours_remaining = minutes_remaining / 60
        print()
        print(f"⏱️  Estimated Time Remaining: {hours_remaining:.1f} hours ({minutes_remaining:.0f} minutes)")
        print(f"   (Based on estimated speed: ~{avg_speed} PDFs/minute)")

    # Check if extraction is complete
    print()
    print("-" * 80)
    if extracted_count >= total_pdfs and extracted_count > 0:
        print("✅ STATUS: EXTRACTION COMPLETE! 🎉")
        print("   All PDFs have been successfully extracted")
        print()
        print("🚀 Ready for Step 3: Semantic Embeddings & Vector Search")
    elif extracted_count > 0:
        print("🔄 STATUS: EXTRACTION IN PROGRESS")
        print("   The extraction process is actively running")
        print()
        print("💡 TIP: Leave your laptop on and check back later")
    else:
        print("⚠️  STATUS: NOT STARTED or JUST STARTING")
        print("   No extracted files found yet")
        print()
        print("💡 TIP: Run 'python step2/step2_extract_hybrid.py' to start extraction")

    print("-" * 80)

    # Check latest log
    print()
    print("📋 RECENT ACTIVITY")
    print("-" * 80)

    if LOGS_DIR.exists():
        log_files = sorted(LOGS_DIR.glob("text_extraction*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            latest_log = log_files[0]
            log_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
            print(f"Latest log: {latest_log.name}")
            print(f"Last updated: {log_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Show last few lines
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_lines = [line.strip() for line in lines[-5:] if line.strip() and not line.strip().startswith('2025-')]
                    if recent_lines:
                        print("\nRecent log entries:")
                        for line in recent_lines[-3:]:
                            if len(line) > 70:
                                line = line[:67] + "..."
                            print(f"  • {line}")
            except Exception as e:
                print(f"  (Could not read log: {e})")
        else:
            print("No extraction logs found")
    else:
        print("No logs directory found")

    print()
    print("-" * 80)

    # Sample extracted files
    if extracted_count > 0 and EXTRACTED_DIR.exists():
        print()
        print("📁 SAMPLE EXTRACTED FILES")
        print("-" * 80)
        sample_files = list(EXTRACTED_DIR.glob("*.txt"))[:5]
        for txt_file in sample_files:
            size_kb = txt_file.stat().st_size / 1024
            print(f"  • {txt_file.name} ({size_kb:.1f} KB)")

    # Storage info
    if extracted_count > 0 and EXTRACTED_DIR.exists():
        total_size = sum(f.stat().st_size for f in EXTRACTED_DIR.glob("*"))
        total_mb = total_size / (1024 * 1024)
        print()
        print(f"💾 Total Storage Used: {total_mb:.1f} MB")

    print()
    print("=" * 80)

    # Quick stats
    if extracted_count > 0:
        print()
        print("📊 QUICK STATISTICS")
        print("-" * 80)

        # Calculate average file sizes
        txt_sizes = [f.stat().st_size for f in EXTRACTED_DIR.glob("*.txt")]
        if txt_sizes:
            avg_size = sum(txt_sizes) / len(txt_sizes) / 1024
            print(f"Average text file size: {avg_size:.1f} KB")

        # Check metadata
        if metadata_count > 0:
            # Load a sample metadata
            sample_meta = list(EXTRACTED_DIR.glob("*.meta.json"))[0]
            try:
                with open(sample_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    print(f"Extraction method: {meta.get('extraction_method', 'Unknown')}")
                    if 'num_words' in meta:
                        print(f"Sample word count: {meta['num_words']:,} words")
                    if 'num_pages' in meta:
                        print(f"Sample page count: {meta['num_pages']} pages")
            except:
                pass

        print("-" * 80)

    print()
    print("✨ To check progress again, run: python step2/check_progress.py")
    print()

if __name__ == "__main__":
    try:
        check_progress()
    except KeyboardInterrupt:
        print("\n\n⚠️  Progress check interrupted")
    except Exception as e:
        print(f"\n❌ Error checking progress: {e}")
        import traceback
        traceback.print_exc()
