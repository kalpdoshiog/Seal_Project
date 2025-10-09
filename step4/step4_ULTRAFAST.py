#!/usr/bin/env python3
"""
Step 4: ULTRA-FAST Embeddings (OPTIMIZED for Resume Mode)

KEY OPTIMIZATION: Pre-filter already processed papers to avoid wasting time
- Fast pre-check of all files (1 second for 12k papers)
- Only process new papers in large batches
- Zero overhead on already-processed papers

Expected: Process remaining 4,000 papers in 8-12 minutes

Author: AI Document Understanding System
Date: October 8, 2025
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = BASE_DIR / "preprocessed_text"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CHUNKS_DIR = BASE_DIR / "chunks"
METADATA_DIR = BASE_DIR / "metadata"

# Configuration
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
USE_FP16 = USE_GPU

# ULTRA-FAST SETTINGS - OPTIMIZED for 6GB VRAM
BATCH_SIZE_CHUNK = 128 if USE_GPU else 32  # Reduced from 512 to 128 (safe for 6GB)
PAPERS_PER_BATCH = 15 if USE_GPU else 10   # Reduced from 50 to 15 (safer)

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_PAPER = 100

# Models
DOCUMENT_MODEL = "allenai/specter2_base"
CHUNK_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

document_model = None
chunk_model = None


def load_models():
    """Load models once"""
    global document_model, chunk_model

    print("Loading models...")
    document_model = SentenceTransformer(DOCUMENT_MODEL, device=DEVICE)
    chunk_model = SentenceTransformer(CHUNK_MODEL, device=DEVICE)

    if USE_FP16 and USE_GPU:
        document_model.half()
        chunk_model.half()

    print(f"✅ Models loaded on {DEVICE}")


def is_already_processed(paper_id: str) -> bool:
    """Quick check if paper is already processed"""
    return all([
        (EMBEDDINGS_DIR / f"{paper_id}_document.npy").exists(),
        (CHUNKS_DIR / f"{paper_id}_chunks.json").exists(),
        (EMBEDDINGS_DIR / f"{paper_id}_chunks.npy").exists(),
        (METADATA_DIR / f"{paper_id}_embed_meta.json").exists()
    ])


def chunk_text_fast(text: str, paper_id: str) -> List[Dict]:
    """Fast chunking with minimal allocations"""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    idx = 0

    while start < len(words) and idx < MAX_CHUNKS_PER_PAPER:
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append({
            'chunk_id': f"{paper_id}_chunk_{idx}",
            'paper_id': paper_id,
            'text': " ".join(words[start:end]),
            'chunk_index': idx,
            'word_count': end - start,
            'start_pos': 0,
            'end_pos': 0
        })
        start = end - CHUNK_OVERLAP
        idx += 1
        if start >= len(words):
            break

    return chunks


def process_batch_ultrafast(paper_ids: List[str], texts: List[str]) -> int:
    """Process a batch of papers at maximum speed"""

    # Clear GPU cache before processing
    if USE_GPU:
        torch.cuda.empty_cache()

    # Step 1: Generate all document embeddings in ONE batch
    doc_texts = [t[:5000] for t in texts]
    doc_embeddings = document_model.encode(
        doc_texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    # Step 2: Chunk ALL papers and collect all chunk texts
    all_chunk_texts = []
    paper_info = []  # (paper_id, doc_embedding, chunks, start_idx, end_idx)

    for i, (paper_id, text) in enumerate(zip(paper_ids, texts)):
        chunks = chunk_text_fast(text, paper_id)
        if not chunks:
            continue

        start_idx = len(all_chunk_texts)
        all_chunk_texts.extend([c['text'] for c in chunks])
        end_idx = len(all_chunk_texts)

        doc_emb = doc_embeddings[i] if len(doc_embeddings.shape) > 1 else doc_embeddings
        paper_info.append((paper_id, doc_emb, chunks, start_idx, end_idx))

    # Step 3: Generate ALL chunk embeddings in batches
    if all_chunk_texts:
        all_chunk_embeddings = chunk_model.encode(
            all_chunk_texts,
            batch_size=BATCH_SIZE_CHUNK,  # 128 chunks at once (safe for 6GB)
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    else:
        return 0

    # Step 4: Save all results (fast I/O)
    for paper_id, doc_emb, chunks, start_idx, end_idx in paper_info:
        chunk_embs = all_chunk_embeddings[start_idx:end_idx]

        # Save document embedding
        np.save(EMBEDDINGS_DIR / f"{paper_id}_document.npy", doc_emb)

        # Save chunks metadata
        chunks_meta = [{
            'chunk_id': c['chunk_id'],
            'paper_id': c['paper_id'],
            'chunk_index': c['chunk_index'],
            'word_count': c['word_count'],
            'start_pos': 0,
            'end_pos': 0
        } for c in chunks]

        with open(CHUNKS_DIR / f"{paper_id}_chunks.json", 'w') as f:
            json.dump(chunks_meta, f)

        # Save chunk embeddings
        np.save(EMBEDDINGS_DIR / f"{paper_id}_chunks.npy", chunk_embs)

        # Save metadata
        meta = {
            'paper_id': paper_id,
            'processing_date': datetime.now().isoformat(),
            'document_model': DOCUMENT_MODEL,
            'chunk_model': CHUNK_MODEL,
            'document_embedding_dim': len(doc_emb),
            'chunk_embedding_dim': len(chunk_embs[0]),
            'num_chunks': len(chunks),
            'total_words': sum(c['word_count'] for c in chunks),
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'gpu_used': USE_GPU,
            'fp16_used': USE_FP16
        }

        with open(METADATA_DIR / f"{paper_id}_embed_meta.json", 'w') as f:
            json.dump(meta, f)

    # Clear GPU cache after processing
    if USE_GPU:
        torch.cuda.empty_cache()

    return len(paper_info)


def main():
    """ULTRA-FAST main pipeline"""
    print("=" * 80)
    print("STEP 4: ULTRA-FAST EMBEDDINGS (Resume-Optimized)")
    print("=" * 80)

    if USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Chunk batch size: {BATCH_SIZE_CHUNK}")
        print(f"Papers per batch: {PAPERS_PER_BATCH}")

    load_models()

    # Get all text files
    all_txt_files = sorted(PREPROCESSED_DIR.glob("*.txt"))
    print(f"\nTotal papers: {len(all_txt_files):,}")

    # PRE-FILTER: Find only papers that need processing
    print("Scanning for unprocessed papers...")
    papers_to_process = []

    for txt_file in tqdm(all_txt_files, desc="Scanning", unit="file"):
        paper_id = txt_file.stem
        if not is_already_processed(paper_id):
            papers_to_process.append(txt_file)

    total_to_process = len(papers_to_process)
    already_done = len(all_txt_files) - total_to_process

    print(f"\n✅ Already processed: {already_done:,}")
    print(f"🔄 Need to process: {total_to_process:,}")
    print("=" * 80)

    if total_to_process == 0:
        print("\n✅ All papers already processed!")
        return

    # Process ONLY the new papers in large batches
    processed = 0

    with tqdm(total=total_to_process, desc="Processing", unit="paper") as pbar:
        for i in range(0, total_to_process, PAPERS_PER_BATCH):
            batch_files = papers_to_process[i:i + PAPERS_PER_BATCH]

            # Read all texts in batch
            paper_ids = []
            texts = []

            for txt_file in batch_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if text.strip():
                        paper_ids.append(txt_file.stem)
                        texts.append(text)
                except:
                    pass

            # Process batch
            if paper_ids:
                count = process_batch_ultrafast(paper_ids, texts)
                processed += count

            pbar.update(len(batch_files))
            pbar.set_postfix({'Processed': processed})

    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"Newly processed: {processed:,}")
    print(f"Total complete: {already_done + processed:,} / {len(all_txt_files):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
