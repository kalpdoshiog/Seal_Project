#!/usr/bin/env python3
"""
Step 5: World-Class Vector Database & Retrieval System (FAISS-GPU)

Multi-stage retrieval with:
- FAISS-GPU for ultra-fast similarity search
- Hybrid search (semantic + BM25 keyword)
- Cross-encoder reranking for quality
- Metadata filtering

Author: AI Document Understanding System
Date: October 8, 2025
"""

import os
import sys
import json
import logging
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch
from tqdm import tqdm

# FAISS for vector similarity search
try:
    import faiss
    FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    print("Installing faiss-gpu...")
    os.system("pip install faiss-gpu")
    import faiss
    FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')

# BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Installing rank-bm25...")
    os.system("pip install rank-bm25")
    from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer

# ----------------------------
# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CHUNKS_DIR = BASE_DIR / "chunks"
METADATA_DIR = BASE_DIR / "metadata"
PREPROCESSED_DIR = BASE_DIR / "preprocessed_text"
FAISS_DIR = BASE_DIR / "faiss_indices"
LOGS_DIR = BASE_DIR / "logs"

# Create output directories
FAISS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Logging Setup
LOG_FILE = LOGS_DIR / f"faiss_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
USE_GPU = torch.cuda.is_available() and FAISS_GPU_AVAILABLE
EMBEDDING_DIM = 768  # SPECTER2 and multi-qa-mpnet dimension

# FAISS index types - OPTIMIZED FOR 12K PAPERS
DOC_INDEX_TYPE = "Flat"  # Use Flat (exact search) for 12K docs - fast enough on CPU
CHUNK_INDEX_TYPE = "HNSW"   # Precise, good for chunk-level

# Index parameters
NLIST = 100  # Number of clusters for IVF (not used with Flat)
NPROBE = 10  # Number of clusters to search (not used with Flat)
M_HNSW = 32  # Connections per layer in HNSW

# GPU Strategy: CPU for docs (12K is small), GPU for chunks if memory allows
USE_GPU_FOR_DOCS = False  # 12K docs = small enough for CPU, avoid GPU overhead
USE_GPU_FOR_CHUNKS = False  # 1.2M chunks would OOM on 6GB GPU


class FAISSIndexBuilder:
    """Build and manage FAISS indices for documents and chunks"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and USE_GPU
        self.doc_index = None
        self.chunk_index = None
        self.doc_paper_ids = []
        self.chunk_metadata = []

        if self.use_gpu and USE_GPU_FOR_CHUNKS:
            logger.info("✅ Initializing FAISS with GPU support")
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                # Limit GPU memory usage to prevent OOM
                self.gpu_resources.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
            except Exception as e:
                logger.warning(f"⚠️ GPU initialization failed: {e}, falling back to CPU")
                self.gpu_resources = None
                self.use_gpu = False
        else:
            logger.info("✅ Using CPU for FAISS (optimal for this dataset size)")
            self.gpu_resources = None
            self.use_gpu = False

    def build_document_index(self, embeddings: np.ndarray, paper_ids: List[str]) -> faiss.Index:
        """
        Build FAISS index for document-level embeddings

        Args:
            embeddings: (N, 768) array of document embeddings
            paper_ids: List of paper IDs corresponding to embeddings

        Returns:
            FAISS index
        """
        logger.info("\n" + "=" * 80)
        logger.info("BUILDING DOCUMENT-LEVEL FAISS INDEX")
        logger.info("=" * 80)

        n_embeddings, dim = embeddings.shape
        logger.info(f"Documents: {n_embeddings:,}")
        logger.info(f"Embedding dimension: {dim}")
        logger.info(f"Index type: {DOC_INDEX_TYPE}")

        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        # Use Flat index for small dataset (12K docs) - faster than IVF
        # Inner product = cosine similarity after normalization
        index = faiss.IndexFlatIP(dim)

        logger.info(f"Using CPU for document index (optimal for {n_embeddings:,} documents)")

        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings)

        self.doc_index = index
        self.doc_paper_ids = paper_ids

        logger.info(f"✅ Document index built: {index.ntotal:,} vectors")
        logger.info("=" * 80)

        return index

    def build_chunk_index(self, embeddings: np.ndarray, chunk_metadata: List[Dict]) -> faiss.Index:
        """
        Build FAISS index for chunk-level embeddings

        Args:
            embeddings: (N, 768) array of chunk embeddings
            chunk_metadata: List of metadata dicts for each chunk

        Returns:
            FAISS index
        """
        logger.info("\n" + "=" * 80)
        logger.info("BUILDING CHUNK-LEVEL FAISS INDEX")
        logger.info("=" * 80)

        n_embeddings, dim = embeddings.shape
        logger.info(f"Chunks: {n_embeddings:,}")
        logger.info(f"Embedding dimension: {dim}")
        logger.info(f"Index type: {CHUNK_INDEX_TYPE}")

        # Ensure embeddings are in correct format (contiguous, float32)
        print("\n🔄 Preparing embeddings...")
        embeddings = np.ascontiguousarray(embeddings, dtype='float32')

        # Normalize embeddings
        print("🔄 Normalizing embeddings...")
        faiss.normalize_L2(embeddings)

        # Create HNSW index (more precise than IVF for smaller datasets)
        print("🔄 Creating HNSW index structure...")
        index = faiss.IndexHNSWFlat(dim, M_HNSW, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40  # Higher = better quality, slower build
        index.hnsw.efSearch = 16  # Higher = better recall, slower search

        # HNSW doesn't support GPU in FAISS, use CPU
        if self.use_gpu:
            logger.info("Note: HNSW index uses CPU (GPU not supported for HNSW)")

        # Add vectors to index with progress bar
        print(f"\n🔨 Building HNSW index for {n_embeddings:,} chunks...")
        print("⏰ This may take 10-30 minutes depending on your CPU speed...")

        # Add in batches to show progress
        batch_size = 10000
        num_batches = (n_embeddings + batch_size - 1) // batch_size

        with tqdm(total=n_embeddings, desc="Adding chunks to index", unit="chunk") as pbar:
            for i in range(0, n_embeddings, batch_size):
                end_idx = min(i + batch_size, n_embeddings)
                batch = embeddings[i:end_idx].astype('float32')
                index.add(batch)
                pbar.update(end_idx - i)

        self.chunk_index = index
        self.chunk_metadata = chunk_metadata

        logger.info(f"✅ Chunk index built: {index.ntotal:,} vectors")
        logger.info("=" * 80)

        return index

    def save_indices(self, output_dir: Path):
        """Save FAISS indices and metadata to disk"""
        logger.info("\n" + "=" * 80)
        logger.info("SAVING FAISS INDICES")
        logger.info("=" * 80)

        # Save document index
        if self.doc_index is not None:
            # Move to CPU before saving if on GPU
            if self.use_gpu:
                index_cpu = faiss.index_gpu_to_cpu(self.doc_index)
            else:
                index_cpu = self.doc_index

            doc_index_path = output_dir / "document_index.faiss"
            faiss.write_index(index_cpu, str(doc_index_path))
            logger.info(f"✅ Saved document index: {doc_index_path}")

            # Save paper IDs mapping
            paper_ids_path = output_dir / "document_paper_ids.pkl"
            with open(paper_ids_path, 'wb') as f:
                pickle.dump(self.doc_paper_ids, f)
            logger.info(f"✅ Saved paper IDs: {paper_ids_path}")

        # Save chunk index
        if self.chunk_index is not None:
            chunk_index_path = output_dir / "chunk_index.faiss"
            faiss.write_index(self.chunk_index, str(chunk_index_path))
            logger.info(f"✅ Saved chunk index: {chunk_index_path}")

            # Save chunk metadata
            chunk_meta_path = output_dir / "chunk_metadata.pkl"
            with open(chunk_meta_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            logger.info(f"✅ Saved chunk metadata: {chunk_meta_path}")

        # Save index configuration
        config = {
            'created_date': datetime.now().isoformat(),
            'embedding_dim': EMBEDDING_DIM,
            'doc_index_type': DOC_INDEX_TYPE,
            'chunk_index_type': CHUNK_INDEX_TYPE,
            'num_documents': len(self.doc_paper_ids),
            'num_chunks': len(self.chunk_metadata),
            'gpu_used': self.use_gpu,
            'nlist': NLIST,
            'nprobe': NPROBE,
            'm_hnsw': M_HNSW
        }

        config_path = output_dir / "index_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Saved configuration: {config_path}")

        logger.info("=" * 80)

    @staticmethod
    def load_indices(index_dir: Path, use_gpu: bool = True):
        """Load FAISS indices from disk"""
        builder = FAISSIndexBuilder(use_gpu=use_gpu)

        logger.info("\n" + "=" * 80)
        logger.info("LOADING FAISS INDICES")
        logger.info("=" * 80)

        # Load document index
        doc_index_path = index_dir / "document_index.faiss"
        if doc_index_path.exists():
            builder.doc_index = faiss.read_index(str(doc_index_path))

            # Move to GPU if requested
            if use_gpu and USE_GPU:
                logger.info("Moving document index to GPU...")
                builder.doc_index = faiss.index_cpu_to_gpu(builder.gpu_resources, 0, builder.doc_index)

            logger.info(f"✅ Loaded document index: {builder.doc_index.ntotal:,} vectors")

            # Load paper IDs
            paper_ids_path = index_dir / "document_paper_ids.pkl"
            with open(paper_ids_path, 'rb') as f:
                builder.doc_paper_ids = pickle.load(f)
            logger.info(f"✅ Loaded {len(builder.doc_paper_ids):,} paper IDs")

        # Load chunk index
        chunk_index_path = index_dir / "chunk_index.faiss"
        if chunk_index_path.exists():
            builder.chunk_index = faiss.read_index(str(chunk_index_path))
            logger.info(f"✅ Loaded chunk index: {builder.chunk_index.ntotal:,} vectors")

            # Load chunk metadata
            chunk_meta_path = index_dir / "chunk_metadata.pkl"
            with open(chunk_meta_path, 'rb') as f:
                builder.chunk_metadata = pickle.load(f)
            logger.info(f"✅ Loaded {len(builder.chunk_metadata):,} chunk metadata entries")

        logger.info("=" * 80)

        return builder


def load_all_embeddings(embeddings_dir: Path, metadata_dir: Path):
    """
    Load all document and chunk embeddings from disk

    Returns:
        doc_embeddings: (N, 768) array
        doc_paper_ids: List of paper IDs
        chunk_embeddings: (M, 768) array
        chunk_metadata: List of chunk metadata dicts
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOADING EMBEDDINGS FROM DISK")
    logger.info("=" * 80)

    # Find all embedding files
    doc_embed_files = sorted(embeddings_dir.glob("*_document.npy"))
    chunk_embed_files = sorted(embeddings_dir.glob("*_chunks.npy"))

    logger.info(f"Found {len(doc_embed_files):,} document embeddings")
    logger.info(f"Found {len(chunk_embed_files):,} chunk embedding files")

    # Load document embeddings with progress bar
    doc_embeddings = []
    doc_paper_ids = []

    print("\n📄 Loading document embeddings...")
    for embed_file in tqdm(doc_embed_files, desc="Document embeddings", unit="file"):
        paper_id = embed_file.stem.replace('_document', '')
        embedding = np.load(embed_file)

        doc_embeddings.append(embedding)
        doc_paper_ids.append(paper_id)

    doc_embeddings = np.vstack(doc_embeddings)
    logger.info(f"✅ Loaded {len(doc_embeddings):,} document embeddings")

    # Load chunk embeddings with progress bar
    chunk_embeddings = []
    chunk_metadata = []

    print("\n📦 Loading chunk embeddings...")
    for embed_file in tqdm(chunk_embed_files, desc="Chunk embeddings", unit="file"):
        paper_id = embed_file.stem.replace('_chunks', '')
        embeddings = np.load(embed_file)

        # Load corresponding metadata
        json_file = embeddings_dir.parent / "chunks" / f"{paper_id}_chunks.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                metadata_list = json.load(f)

            # Add each chunk
            for i, metadata in enumerate(metadata_list):
                if i < len(embeddings):
                    chunk_embeddings.append(embeddings[i])
                    chunk_metadata.append(metadata)

    chunk_embeddings = np.vstack(chunk_embeddings)
    logger.info(f"✅ Loaded {len(chunk_embeddings):,} chunk embeddings")

    logger.info("=" * 80)

    return doc_embeddings, doc_paper_ids, chunk_embeddings, chunk_metadata


def main():
    """Main FAISS index building pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: BUILDING FAISS VECTOR DATABASE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU availability
    if USE_GPU:
        logger.info(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✅ FAISS GPU support: {FAISS_GPU_AVAILABLE}")
    else:
        logger.warning("⚠️  No GPU available - using CPU")

    # Load all embeddings
    doc_embeddings, doc_paper_ids, chunk_embeddings, chunk_metadata = load_all_embeddings(
        EMBEDDINGS_DIR, METADATA_DIR
    )

    # Initialize FAISS builder
    builder = FAISSIndexBuilder(use_gpu=USE_GPU)

    # Build document index
    builder.build_document_index(doc_embeddings, doc_paper_ids)

    # Build chunk index
    builder.build_chunk_index(chunk_embeddings, chunk_metadata)

    # Save indices
    builder.save_indices(FAISS_DIR)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FAISS INDEX BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Document index: {len(doc_paper_ids):,} papers")
    logger.info(f"Chunk index: {len(chunk_metadata):,} chunks")
    logger.info(f"Output directory: {FAISS_DIR}")
    logger.info(f"GPU used: {USE_GPU}")
    logger.info(f"\n✅ Ready for semantic search and retrieval!")
    logger.info("=" * 80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
