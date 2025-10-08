#!/usr/bin/env python3
"""
Check preprocessing progress and statistics - FULLY DYNAMIC
Real-time monitoring with automatic refresh
"""

import json
from pathlib import Path
from collections import defaultdict
import time
import sys
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = BASE_DIR / "extracted_text"
PREPROCESSED_DIR = BASE_DIR / "preprocessed_text"
ENTITIES_DIR = BASE_DIR / "extracted_entities"
PREPROCESSING_LOG = BASE_DIR / "preprocessing_results.csv"
LOGS_DIR = BASE_DIR / "logs"

def clear_screen():
    """Clear console screen"""
    print("\033[2J\033[H", end="")

def get_latest_log_file():
    """Get the most recent preprocessing log file"""
    log_files = list(LOGS_DIR.glob("preprocessing_*.log"))
    if log_files:
        return max(log_files, key=lambda f: f.stat().st_mtime)
    return None

def get_processing_speed():
    """Calculate current processing speed from log file"""
    log_file = get_latest_log_file()
    if not log_file:
        return None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Look for timestamp patterns to calculate speed
        if len(lines) > 10:
            # Get first and last timestamps
            first_time = None
            last_time = None

            for line in lines[:20]:
                if " - INFO - " in line and "Found" in line and "text files" in line:
                    timestamp_str = line.split(" - ")[0]
                    try:
                        first_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    except:
                        pass

            # Get most recent timestamp
            for line in reversed(lines[-50:]):
                if " - " in line:
                    timestamp_str = line.split(" - ")[0]
                    try:
                        last_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                        break
                    except:
                        pass

            if first_time and last_time:
                elapsed = (last_time - first_time).total_seconds()
                if elapsed > 0:
                    return elapsed
    except:
        pass

    return None

def read_csv_results():
    """Read preprocessing results CSV for real-time stats"""
    if not PREPROCESSING_LOG.exists():
        return None

    results = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_words': 0,
        'total_chars': 0,
        'total_unique': 0,
        'total_entities': 0
    }

    try:
        with open(PREPROCESSING_LOG, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    status = parts[1]
                    if status == 'success':
                        results['success'] += 1
                        if len(parts) >= 6:
                            try:
                                results['total_words'] += int(parts[2]) if parts[2] else 0
                                results['total_chars'] += int(parts[3]) if parts[3] else 0
                                results['total_unique'] += int(parts[4]) if parts[4] else 0
                                results['total_entities'] += int(parts[5]) if parts[5] else 0
                            except:
                                pass
                    elif status == 'failed':
                        results['failed'] += 1
                    elif status == 'skipped':
                        results['skipped'] += 1

        return results
    except:
        return None

def format_time(minutes):
    """Format time in human-readable format"""
    if minutes < 1:
        return f"{minutes * 60:.0f} seconds"
    elif minutes < 60:
        return f"{minutes:.1f} minutes"
    else:
        hours = minutes / 60
        return f"{hours:.1f} hours"

def display_progress(watch_mode=False):
    """Display current preprocessing progress - FULLY DYNAMIC"""

    while True:
        if watch_mode:
            clear_screen()

        # Count files DYNAMICALLY
        extracted_files = list(EXTRACTED_DIR.glob("*.txt"))
        preprocessed_files = list(PREPROCESSED_DIR.glob("*.txt"))
        metadata_files = list(PREPROCESSED_DIR.glob("*.meta.json"))
        entity_files = list(ENTITIES_DIR.glob("*.json"))

        total_extracted = len(extracted_files)
        total_preprocessed = len(preprocessed_files)
        total_metadata = len(metadata_files)
        total_entities = len(entity_files)

        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("=" * 80)
        print(f"STEP 3: PREPROCESSING PROGRESS CHECK - {current_time}")
        print("=" * 80)

        # Input files
        print(f"\n📁 Input Files (extracted_text/):")
        print(f"   Total extracted text files: {total_extracted:,}")

        # Output files
        print(f"\n📊 Output Files (preprocessed_text/):")
        print(f"   Preprocessed text files: {total_preprocessed:,}")
        print(f"   Metadata files: {total_metadata:,}")

        # Progress bar
        if total_extracted > 0:
            progress_pct = (total_preprocessed / total_extracted) * 100
            bar_length = 50
            filled_length = int(bar_length * total_preprocessed / total_extracted)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"   Progress: [{bar}] {progress_pct:.1f}%")
            print(f"   Processed: {total_preprocessed:,} / {total_extracted:,}")

        # Entity extraction
        print(f"\n🏷️  Entity Extraction (extracted_entities/):")
        print(f"   Papers with entities: {total_entities:,}")

        # Read CSV results for real-time stats
        csv_results = read_csv_results()
        if csv_results:
            print(f"\n📈 Processing Results (from CSV log):")
            print(f"   ✅ Success: {csv_results['success']:,}")
            print(f"   ⏭️  Skipped: {csv_results['skipped']:,}")
            print(f"   ❌ Failed: {csv_results['failed']:,}")

            total_processed = csv_results['success']
            if total_processed > 0:
                print(f"\n📊 Average Statistics per Paper:")
                print(f"   Words: {csv_results['total_words'] / total_processed:,.0f}")
                print(f"   Characters: {csv_results['total_chars'] / total_processed:,.0f}")
                print(f"   Unique words: {csv_results['total_unique'] / total_processed:,.0f}")
                print(f"   Entities found: {csv_results['total_entities'] / total_processed:,.1f}")

        # Detailed statistics from ALL metadata files DYNAMICALLY
        if metadata_files:
            # Sample or process all based on count
            sample_size = min(len(metadata_files), 1000)  # Process up to 1000 for speed

            print(f"\n🔬 Detailed Analysis (from {sample_size:,} metadata files):")

            total_words = 0
            total_chars = 0
            total_unique = 0
            total_sentences = 0
            total_entities_count = 0
            entity_counts = defaultdict(int)
            processing_modes = defaultdict(int)
            failed_reads = 0

            # Process metadata files dynamically
            import random
            sample_files = random.sample(metadata_files, sample_size) if len(metadata_files) > sample_size else metadata_files

            for meta_file in sample_files:
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        stats = meta.get('statistics', {})
                        total_words += stats.get('word_count', 0)
                        total_chars += stats.get('char_count', 0)
                        total_unique += stats.get('unique_words', 0)
                        total_sentences += stats.get('sentence_count', 0)

                        # Processing mode
                        mode = meta.get('preprocessing_mode', 'unknown')
                        processing_modes[mode] += 1

                        # Entity counts
                        for ent_type, count in meta.get('entity_counts', {}).items():
                            entity_counts[ent_type] += count
                            total_entities_count += count
                except Exception as e:
                    failed_reads += 1

            if sample_size > 0:
                actual_reads = sample_size - failed_reads
                if actual_reads > 0:
                    print(f"\n   📝 Text Quality Metrics:")
                    print(f"      Words per paper: {total_words / actual_reads:,.0f}")
                    print(f"      Characters per paper: {total_chars / actual_reads:,.0f}")
                    print(f"      Unique words per paper: {total_unique / actual_reads:,.0f}")
                    print(f"      Sentences per paper: {total_sentences / actual_reads:,.0f}")
                    print(f"      Vocabulary richness: {total_unique / total_words * 100 if total_words > 0 else 0:.1f}%")

                    if entity_counts:
                        print(f"\n   🏷️  Named Entities Extracted:")
                        print(f"      Total entities: {total_entities_count:,}")
                        print(f"      Avg per paper: {total_entities_count / actual_reads:.1f}")
                        print(f"\n      By type:")
                        for ent_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
                            if count > 0:
                                avg_per_paper = count / actual_reads
                                print(f"         {ent_type:15s}: {count:6,} total  ({avg_per_paper:.1f} avg/paper)")

                    if processing_modes:
                        print(f"\n   ⚙️  Processing Modes:")
                        for mode, count in processing_modes.items():
                            print(f"      {mode}: {count:,} files")

        # Calculate remaining and ETA DYNAMICALLY
        remaining = total_extracted - total_preprocessed
        if remaining > 0:
            print(f"\n⏳ Processing Status:")
            print(f"   Remaining files: {remaining:,}")
            print(f"   Completion: {progress_pct:.2f}%")

            # Dynamic speed calculation
            elapsed = get_processing_speed()
            if elapsed and total_preprocessed > 0:
                files_per_second = total_preprocessed / elapsed
                if files_per_second > 0:
                    eta_seconds = remaining / files_per_second
                    eta_minutes = eta_seconds / 60
                    print(f"   Processing speed: {files_per_second * 60:.1f} files/minute")
                    print(f"   Estimated time remaining: {format_time(eta_minutes)}")
            else:
                # Fallback estimation
                eta_minutes_low = remaining / 50
                eta_minutes_high = remaining / 20
                print(f"   Estimated time: {format_time(eta_minutes_low)} - {format_time(eta_minutes_high)}")

            # Latest log file info
            latest_log = get_latest_log_file()
            if latest_log:
                log_modified = datetime.fromtimestamp(latest_log.stat().st_mtime)
                time_since = (datetime.now() - log_modified).total_seconds()
                print(f"\n   💡 Latest log: {latest_log.name}")
                print(f"      Last updated: {time_since:.0f} seconds ago")

                if time_since > 300:  # 5 minutes
                    print(f"      ⚠️  Warning: Log hasn't updated recently - process may have stopped")
        else:
            print(f"\n✅ ALL FILES PREPROCESSED!")
            print(f"\n   📊 Final Statistics:")
            print(f"      Total papers processed: {total_preprocessed:,}")
            print(f"      Papers with entities: {total_entities:,}")
            print(f"\n   🎯 Next Steps:")
            print(f"      1. Review preprocessing_results.csv for quality check")
            print(f"      2. Verify extracted_entities/ folder")
            print(f"      3. Proceed to Step 4: Generate Embeddings")

        print("=" * 80)

        if watch_mode and remaining > 0:
            print("\n🔄 Auto-refreshing every 10 seconds... (Press Ctrl+C to stop)")
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                print("\n\n✋ Monitoring stopped by user.")
                break
        else:
            break

def main():
    """Main function with watch mode option"""
    import argparse
    parser = argparse.ArgumentParser(description='Check preprocessing progress')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Watch mode: auto-refresh every 10 seconds')

    args = parser.parse_args()

    try:
        display_progress(watch_mode=args.watch)
    except KeyboardInterrupt:
        print("\n\n✋ Stopped by user.")

if __name__ == "__main__":
    main()
