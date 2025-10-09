#!/usr/bin/env python3
"""
Check Step 4 embedding generation progress - FULLY DYNAMIC
Real-time monitoring with statistics
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = BASE_DIR / "preprocessed_text"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CHUNKS_DIR = BASE_DIR / "chunks"
METADATA_DIR = BASE_DIR / "metadata"
LOGS_DIR = BASE_DIR / "logs"

def format_size(bytes_size):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def get_directory_size(directory):
    """Calculate total size of directory"""
    total = 0
    try:
        for file in directory.glob("**/*"):
            if file.is_file():
                total += file.stat().st_size
    except:
        pass
    return total

def main():
    """Display embedding generation progress"""

    # Count files dynamically
    preprocessed_files = list(PREPROCESSED_DIR.glob("*.txt"))
    doc_embeddings = list(EMBEDDINGS_DIR.glob("*_document.npy"))
    chunk_embeddings = list(EMBEDDINGS_DIR.glob("*_chunks.npy"))
    chunk_metadata = list(CHUNKS_DIR.glob("*_chunks.json"))
    embed_metadata = list(METADATA_DIR.glob("*_embed_meta.json"))

    total_papers = len(preprocessed_files)
    papers_embedded = len(doc_embeddings)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print(f"STEP 4: EMBEDDING GENERATION PROGRESS - {current_time}")
    print("=" * 80)

    # Input status
    print(f"\n📁 Input Files (preprocessed_text/):")
    print(f"   Total preprocessed papers: {total_papers:,}")

    # Output status
    print(f"\n📊 Output Files:")
    print(f"   Document embeddings: {papers_embedded:,}")
    print(f"   Chunk embeddings: {len(chunk_embeddings):,}")
    print(f"   Chunk metadata files: {len(chunk_metadata):,}")
    print(f"   Embedding metadata: {len(embed_metadata):,}")

    # Progress bar
    if total_papers > 0:
        progress_pct = (papers_embedded / total_papers) * 100
        bar_length = 50
        filled_length = int(bar_length * papers_embedded / total_papers)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\n   Progress: [{bar}] {progress_pct:.1f}%")
        print(f"   Processed: {papers_embedded:,} / {total_papers:,}")

    # Analyze metadata for statistics
    if embed_metadata:
        print(f"\n🔬 Detailed Statistics (from {len(embed_metadata):,} metadata files):")

        total_chunks = 0
        total_doc_dim = 0
        total_chunk_dim = 0
        models_used = defaultdict(int)
        gpu_count = 0
        fp16_count = 0

        sample_size = min(len(embed_metadata), 1000)
        import random
        sample_files = random.sample(embed_metadata, sample_size)

        for meta_file in sample_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    total_chunks += meta.get('num_chunks', 0)
                    total_doc_dim = meta.get('document_embedding_dim', 768)
                    total_chunk_dim = meta.get('chunk_embedding_dim', 768)

                    doc_model = meta.get('document_model', 'unknown')
                    models_used[doc_model] += 1

                    if meta.get('gpu_used', False):
                        gpu_count += 1
                    if meta.get('fp16_used', False):
                        fp16_count += 1
            except:
                pass

        if sample_size > 0:
            avg_chunks = total_chunks / sample_size
            print(f"\n   📝 Embedding Metrics:")
            print(f"      Average chunks per paper: {avg_chunks:.1f}")
            print(f"      Total chunks created: {total_chunks:,} (from sample)")
            print(f"      Estimated total chunks: {int(avg_chunks * papers_embedded):,}")
            print(f"      Document embedding dimension: {total_doc_dim}")
            print(f"      Chunk embedding dimension: {total_chunk_dim}")

            print(f"\n   ⚙️  Processing Configuration:")
            for model, count in models_used.items():
                print(f"      Model: {model} ({count} papers)")
            print(f"      GPU acceleration: {gpu_count}/{sample_size} papers ({gpu_count/sample_size*100:.1f}%)")
            print(f"      FP16 mixed precision: {fp16_count}/{sample_size} papers ({fp16_count/sample_size*100:.1f}%)")

            est_speed = 200 if gpu_count > sample_size * 0.8 else 10
        else:
            est_speed = 100  # Default estimate
    else:
        est_speed = 100  # Default estimate

    # Storage analysis
    print(f"\n💾 Storage Usage:")

    embeddings_size = get_directory_size(EMBEDDINGS_DIR)
    chunks_size = get_directory_size(CHUNKS_DIR)
    metadata_size = get_directory_size(METADATA_DIR)
    total_size = embeddings_size + chunks_size + metadata_size

    print(f"   Embeddings: {format_size(embeddings_size)}")
    print(f"   Chunks metadata: {format_size(chunks_size)}")
    print(f"   Processing metadata: {format_size(metadata_size)}")
    print(f"   Total: {format_size(total_size)}")

    # Estimate for completion
    if papers_embedded > 0 and total_papers > papers_embedded:
        avg_size_per_paper = total_size / papers_embedded
        remaining_papers = total_papers - papers_embedded
        estimated_final_size = total_size + (avg_size_per_paper * remaining_papers)
        print(f"   Estimated final size: {format_size(estimated_final_size)}")

    # Check for latest log
    log_files = list(LOGS_DIR.glob("embeddings_*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        log_modified = datetime.fromtimestamp(latest_log.stat().st_mtime)
        time_since = (datetime.now() - log_modified).total_seconds()

        print(f"\n💡 Latest Log:")
        print(f"   File: {latest_log.name}")
        print(f"   Last updated: {time_since:.0f} seconds ago")

        if time_since > 300 and papers_embedded < total_papers:
            print(f"   ⚠️  Warning: Log hasn't updated recently - process may have stopped")

    # Status
    remaining = total_papers - papers_embedded
    if remaining > 0:
        print(f"\n⏳ Processing Status:")
        print(f"   Remaining papers: {remaining:,}")
        print(f"   Completion: {progress_pct:.2f}%")

        # Estimate time
        eta_minutes = remaining / est_speed
        if eta_minutes < 60:
            print(f"   Estimated time remaining: {eta_minutes:.1f} minutes")
        else:
            print(f"   Estimated time remaining: {eta_minutes/60:.1f} hours")

        if papers_embedded == 0:
            print(f"\n   🔄 Status: Models downloading and initializing...")
            print(f"   Please wait - first-time model download can take 5-10 minutes")
    else:
        print(f"\n✅ ALL PAPERS EMBEDDED!")
        print(f"\n   📊 Final Statistics:")
        print(f"      Total papers: {papers_embedded:,}")
        print(f"      Document embeddings: {len(doc_embeddings):,}")
        print(f"      Chunk embeddings: {len(chunk_embeddings):,}")
        print(f"      Total storage: {format_size(total_size)}")

        print(f"\n   🎯 Next Steps:")
        print(f"      1. Review embedding quality")
        print(f"      2. Build FAISS vector index (Step 5)")
        print(f"      3. Implement semantic search (Step 6)")

    print("=" * 80)

if __name__ == "__main__":
    main()
