#!/usr/bin/env python3
"""
Step 5: Advanced Retrieval System with Hybrid Search & Reranking

Multi-stage retrieval pipeline:
- Stage 1: FAISS semantic search + BM25 keyword search
- Stage 2: Hybrid fusion (Reciprocal Rank Fusion)
- Stage 3: Cross-encoder reranking for quality
- Stage 4: Post-processing and context extraction

Author: AI Document Understanding System
Date: October 8, 2025
"""

import os
import sys
import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ----------------------------
# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = BASE_DIR / "faiss_indices"
PREPROCESSED_DIR = BASE_DIR / "preprocessed_text"
CHUNKS_DIR = BASE_DIR / "chunks"

# ----------------------------
# Configuration
EMBEDDING_DIM = 768
DEFAULT_TOP_K = 10
RERANK_TOP_K = 50  # Retrieve more, then rerank to top 10


@dataclass
class SearchResult:
    """Search result with metadata"""
    paper_id: str
    chunk_id: Optional[str]
    score: float
    text: str
    chunk_index: Optional[int] = None
    rank: int = 0
    rerank_score: Optional[float] = None


class HybridRetriever:
    """Advanced retrieval system with semantic + keyword search"""

    def __init__(self, faiss_dir: Path, use_gpu: bool = True):
        self.faiss_dir = faiss_dir

        # Check if CUDA is actually available
        import torch
        cuda_available = torch.cuda.is_available()
        self.use_gpu = use_gpu and cuda_available

        # Load FAISS indices
        self._load_faiss_indices()

        # Load embedding model for query encoding
        print("Loading query encoder...")
        self.query_encoder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

        # Only move to GPU if CUDA is truly available
        if self.use_gpu:
            try:
                self.query_encoder = self.query_encoder.cuda()
                print("✅ Using GPU for query encoding")
            except Exception as e:
                print(f"⚠️  GPU not available, using CPU: {e}")
                self.use_gpu = False

        # Build BM25 index
        self._build_bm25_index()

        # Load reranker (optional, for quality boost)
        self.reranker = None

    def _load_faiss_indices(self):
        """Load pre-built FAISS indices"""
        print("Loading FAISS indices...")

        # Load document index
        doc_index_path = self.faiss_dir / "document_index.faiss"
        self.doc_index = faiss.read_index(str(doc_index_path))

        with open(self.faiss_dir / "document_paper_ids.pkl", 'rb') as f:
            self.doc_paper_ids = pickle.load(f)

        # Load chunk index
        chunk_index_path = self.faiss_dir / "chunk_index.faiss"
        self.chunk_index = faiss.read_index(str(chunk_index_path))

        with open(self.faiss_dir / "chunk_metadata.pkl", 'rb') as f:
            self.chunk_metadata = pickle.load(f)

        print(f"✅ Loaded document index: {self.doc_index.ntotal:,} vectors")
        print(f"✅ Loaded chunk index: {self.chunk_index.ntotal:,} vectors")

    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        print("Building BM25 index...")
        print(f"⏰ This will load and tokenize {len(self.chunk_metadata):,} chunks (takes 2-3 minutes)...")

        # Import tqdm for progress bar
        from tqdm import tqdm

        # Tokenize all chunk texts
        chunk_texts = []

        print("\n📦 Loading chunk texts from preprocessed files...")
        for meta in tqdm(self.chunk_metadata, desc="Loading chunks", unit="chunk"):
            # Load actual text from chunks file
            paper_id = meta['paper_id']
            chunk_file = CHUNKS_DIR / f"{paper_id}_chunks.json"

            if chunk_file.exists():
                with open(chunk_file, 'r') as f:
                    chunks = json.load(f)
                    if meta['chunk_index'] < len(chunks):
                        # For BM25, we need the actual text (not stored in metadata)
                        # Load from preprocessed text
                        text_file = PREPROCESSED_DIR / f"{paper_id}.txt"
                        if text_file.exists():
                            with open(text_file, 'r', encoding='utf-8') as tf:
                                full_text = tf.read()
                                # Extract chunk text using positions
                                chunk_text = full_text[meta['start_pos']:meta['end_pos']]
                                chunk_texts.append(chunk_text)
                        else:
                            chunk_texts.append("")
                    else:
                        chunk_texts.append("")
            else:
                chunk_texts.append("")

        # Tokenize for BM25
        print("\n🔤 Tokenizing chunks for BM25...")
        tokenized_corpus = []
        for text in tqdm(chunk_texts, desc="Tokenizing", unit="chunk"):
            tokenized_corpus.append(text.lower().split())

        print("\n🔨 Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunk_texts = chunk_texts

        print(f"✅ BM25 index built: {len(chunk_texts):,} chunks")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query into embedding vector"""
        embedding = self.query_encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding

    def semantic_search_documents(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search documents using FAISS semantic similarity"""
        # Encode query
        query_embedding = self.encode_query(query).reshape(1, -1).astype('float32')

        # Search
        scores, indices = self.doc_index.search(query_embedding, top_k)

        # Convert to results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid result
                paper_id = self.doc_paper_ids[idx]

                # Load document text
                text_file = PREPROCESSED_DIR / f"{paper_id}.txt"
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read()[:500]  # First 500 chars as preview
                else:
                    text = ""

                results.append(SearchResult(
                    paper_id=paper_id,
                    chunk_id=None,
                    score=float(score),
                    text=text,
                    rank=rank + 1
                ))

        return results

    def semantic_search_chunks(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """Search chunks using FAISS semantic similarity"""
        # Encode query
        query_embedding = self.encode_query(query).reshape(1, -1).astype('float32')

        # Search
        scores, indices = self.chunk_index.search(query_embedding, top_k)

        # Convert to results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.chunk_metadata):
                meta = self.chunk_metadata[idx]
                text = self.chunk_texts[idx] if idx < len(self.chunk_texts) else ""

                results.append(SearchResult(
                    paper_id=meta['paper_id'],
                    chunk_id=meta['chunk_id'],
                    score=float(score),
                    text=text,
                    chunk_index=meta['chunk_index'],
                    rank=rank + 1
                ))

        return results

    def bm25_search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """Search chunks using BM25 keyword matching"""
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Convert to results
        results = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self.chunk_metadata):
                meta = self.chunk_metadata[idx]
                text = self.chunk_texts[idx] if idx < len(self.chunk_texts) else ""

                results.append(SearchResult(
                    paper_id=meta['paper_id'],
                    chunk_id=meta['chunk_id'],
                    score=float(scores[idx]),
                    text=text,
                    chunk_index=meta['chunk_index'],
                    rank=rank + 1
                ))

        return results

    def hybrid_search(self, query: str, top_k: int = 10,
                     semantic_weight: float = 0.7) -> List[SearchResult]:
        """
        Hybrid search combining semantic (FAISS) and keyword (BM25) search
        Uses Reciprocal Rank Fusion (RRF) to merge results
        """
        # Get results from both methods
        semantic_results = self.semantic_search_chunks(query, top_k=RERANK_TOP_K)
        bm25_results = self.bm25_search(query, top_k=RERANK_TOP_K)

        # Reciprocal Rank Fusion
        rrf_scores = {}
        k_rrf = 60  # RRF parameter

        # Add semantic scores
        for result in semantic_results:
            key = (result.paper_id, result.chunk_id)
            rrf_scores[key] = rrf_scores.get(key, 0) + semantic_weight / (k_rrf + result.rank)

        # Add BM25 scores
        for result in bm25_results:
            key = (result.paper_id, result.chunk_id)
            rrf_scores[key] = rrf_scores.get(key, 0) + (1 - semantic_weight) / (k_rrf + result.rank)

        # Sort by combined score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Create result objects
        results = []
        seen_chunks = {}

        for (paper_id, chunk_id), score in sorted_results[:top_k]:
            # Find original result
            for r in semantic_results + bm25_results:
                if r.paper_id == paper_id and r.chunk_id == chunk_id:
                    if chunk_id not in seen_chunks:
                        r.score = score
                        r.rank = len(results) + 1
                        results.append(r)
                        seen_chunks[chunk_id] = True
                        break

            if len(results) >= top_k:
                break

        return results

    def rerank(self, query: str, results: List[SearchResult],
               top_k: int = 10) -> List[SearchResult]:
        """
        Rerank results using cross-encoder for better quality
        Cross-encoders are slower but more accurate than bi-encoders
        """
        if self.reranker is None:
            print("Loading cross-encoder reranker...")
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Prepare pairs for reranking
        pairs = [[query, result.text] for result in results]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Update results with rerank scores
        for result, score in zip(results, rerank_scores):
            result.rerank_score = float(score)

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.rerank_score, reverse=True)

        # Update ranks
        for rank, result in enumerate(reranked[:top_k]):
            result.rank = rank + 1

        return reranked[:top_k]

    def search(self, query: str, top_k: int = 10,
               use_hybrid: bool = True, use_reranking: bool = True) -> List[SearchResult]:
        """
        Complete search pipeline with all stages

        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Use hybrid search (semantic + BM25)
            use_reranking: Apply cross-encoder reranking

        Returns:
            List of SearchResult objects
        """
        start_time = time.time()

        # Stage 1: Initial retrieval
        if use_hybrid:
            results = self.hybrid_search(query, top_k=RERANK_TOP_K if use_reranking else top_k)
        else:
            results = self.semantic_search_chunks(query, top_k=RERANK_TOP_K if use_reranking else top_k)

        # Stage 2: Reranking (optional but recommended)
        if use_reranking and len(results) > 0:
            results = self.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        print(f"\n🔍 Search completed in {elapsed:.1f}ms")
        print(f"📊 Hybrid: {use_hybrid} | Reranking: {use_reranking}")
        print(f"✅ Found {len(results)} results")

        return results


def format_results(results: List[SearchResult], show_text: bool = True):
    """Pretty print search results"""
    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n{result.rank}. Paper: {result.paper_id}")
        if result.chunk_id:
            print(f"   Chunk: {result.chunk_id}")
        print(f"   Score: {result.score:.4f}", end="")
        if result.rerank_score is not None:
            print(f" | Rerank: {result.rerank_score:.4f}", end="")
        print()

        if show_text and result.text:
            preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
            print(f"   Preview: {preview}")

    print("=" * 80)


def main():
    """Demo of the retrieval system"""
    print("\n" + "=" * 80)
    print("STEP 5: ADVANCED RETRIEVAL SYSTEM DEMO")
    print("=" * 80)

    # Initialize retriever
    print("\nInitializing retrieval system...")
    retriever = HybridRetriever(FAISS_DIR, use_gpu=True)

    # Example queries
    queries = [
        "transformer attention mechanism",
        "BERT language model",
        "computer vision deep learning"
    ]

    print("\n" + "=" * 80)
    print("RUNNING EXAMPLE SEARCHES")
    print("=" * 80)

    for query in queries:
        print(f"\n\n🔍 Query: '{query}'")
        print("-" * 80)

        # Hybrid search with reranking (best quality)
        results = retriever.search(query, top_k=5, use_hybrid=True, use_reranking=True)
        format_results(results, show_text=True)


if __name__ == "__main__":
    main()
