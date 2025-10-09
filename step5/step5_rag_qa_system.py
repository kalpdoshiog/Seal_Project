#!/usr/bin/env python3
"""
Step 5: RAG-Based QA and Summarization System

Complete pipeline:
- Retrieval: FAISS + BM25 hybrid search
- Generation: GPU-optimized BART/FLAN-T5 for summarization and QA
- Logging: Comprehensive query/response logging

Author: AI Document Understanding System
Date: October 9, 2025
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# Import our retrieval system
from step5_advanced_retrieval import HybridRetriever, SearchResult

# ----------------------------
# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = BASE_DIR / "faiss_indices"
LOGS_DIR = BASE_DIR / "logs"
QA_LOGS_DIR = LOGS_DIR / "qa_logs"

# Create directories
QA_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Logging Setup
LOG_FILE = LOGS_DIR / f"rag_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # GPU-optimized for summarization
QA_MODEL = "google/flan-t5-large"  # Good balance of quality and speed
MAX_CONTEXT_LENGTH = 2048  # Max tokens for context
MAX_SUMMARY_LENGTH = 512  # Max summary length
MAX_ANSWER_LENGTH = 256  # Max answer length


@dataclass
class QAResponse:
    """Question answering response with metadata"""
    question: str
    answer: str
    context_chunks: List[SearchResult]
    model_used: str
    confidence: float
    generation_time_ms: float
    timestamp: str
    retrieval_method: str

    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            'question': self.question,
            'answer': self.answer,
            'num_chunks': len(self.context_chunks),
            'top_papers': [c.paper_id for c in self.context_chunks[:3]],
            'model_used': self.model_used,
            'confidence': self.confidence,
            'generation_time_ms': self.generation_time_ms,
            'timestamp': self.timestamp,
            'retrieval_method': self.retrieval_method
        }


@dataclass
class SummaryResponse:
    """Summarization response with metadata"""
    query: str
    summary: str
    source_chunks: List[SearchResult]
    model_used: str
    compression_ratio: float
    generation_time_ms: float
    timestamp: str

    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            'query': self.query,
            'summary': self.summary,
            'num_chunks': len(self.source_chunks),
            'top_papers': [c.paper_id for c in self.source_chunks[:3]],
            'model_used': self.model_used,
            'compression_ratio': self.compression_ratio,
            'generation_time_ms': self.generation_time_ms,
            'timestamp': self.timestamp
        }


class RAGQASystem:
    """Complete RAG-based QA and Summarization System"""

    def __init__(self, faiss_dir: Path, use_gpu: bool = True):
        self.faiss_dir = faiss_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"

        logger.info("=" * 80)
        logger.info("INITIALIZING RAG QA SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")

        # Initialize retrieval system
        logger.info("\n📚 Loading retrieval system...")
        self.retriever = HybridRetriever(faiss_dir, use_gpu=self.use_gpu)

        # Load generative models
        self._load_models()

        logger.info("\n✅ RAG QA System ready!")
        logger.info("=" * 80)

    def _load_models(self):
        """Load generative models for summarization and QA"""

        # Load summarization model (BART)
        logger.info("\n📝 Loading summarization model (BART)...")
        try:
            self.summarizer = pipeline(
                "summarization",
                model=SUMMARIZATION_MODEL,
                device=0 if self.use_gpu else -1,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32
            )
            logger.info(f"✅ Loaded: {SUMMARIZATION_MODEL}")
        except Exception as e:
            logger.warning(f"⚠️  Could not load BART, falling back to lighter model: {e}")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-base",
                device=0 if self.use_gpu else -1
            )

        # Load QA model (FLAN-T5)
        logger.info("\n❓ Loading QA model (FLAN-T5)...")
        try:
            self.qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
            self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(
                QA_MODEL,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32
            )
            if self.use_gpu:
                self.qa_model = self.qa_model.to(self.device)
            self.qa_model.eval()
            logger.info(f"✅ Loaded: {QA_MODEL}")
        except Exception as e:
            logger.warning(f"⚠️  Could not load FLAN-T5, using lighter model: {e}")
            self.qa_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            if self.use_gpu:
                self.qa_model = self.qa_model.to(self.device)
            self.qa_model.eval()

    def answer_question(self, question: str, top_k: int = 5,
                       use_reranking: bool = True) -> QAResponse:
        """
        Answer a question using RAG (Retrieval-Augmented Generation)

        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve for context
            use_reranking: Use cross-encoder reranking

        Returns:
            QAResponse with answer and metadata
        """
        start_time = time.time()

        logger.info("\n" + "=" * 80)
        logger.info(f"QUESTION: {question}")
        logger.info("=" * 80)

        # Step 1: Retrieve relevant context
        logger.info("\n🔍 Retrieving relevant context...")
        chunks = self.retriever.search(
            question,
            top_k=top_k,
            use_hybrid=True,
            use_reranking=use_reranking
        )

        # Step 2: Prepare context for generation
        logger.info("\n📝 Preparing context for generation...")
        context = self._prepare_context(chunks, max_length=MAX_CONTEXT_LENGTH)

        # Step 3: Generate answer
        logger.info("\n🤖 Generating answer...")
        answer, confidence = self._generate_answer(question, context)

        generation_time = (time.time() - start_time) * 1000

        # Create response
        response = QAResponse(
            question=question,
            answer=answer,
            context_chunks=chunks,
            model_used=QA_MODEL,
            confidence=confidence,
            generation_time_ms=generation_time,
            timestamp=datetime.now().isoformat(),
            retrieval_method="hybrid+reranking" if use_reranking else "hybrid"
        )

        # Log the QA interaction
        self._log_qa_response(response)

        logger.info(f"\n✅ Answer generated in {generation_time:.1f}ms")

        return response

    def summarize(self, query: str, top_k: int = 10,
                  max_length: int = 512) -> SummaryResponse:
        """
        Generate a summary of relevant papers/chunks

        Args:
            query: Topic or query to summarize
            top_k: Number of chunks to retrieve
            max_length: Maximum summary length

        Returns:
            SummaryResponse with summary and metadata
        """
        start_time = time.time()

        logger.info("\n" + "=" * 80)
        logger.info(f"SUMMARIZATION QUERY: {query}")
        logger.info("=" * 80)

        # Step 1: Retrieve relevant chunks
        logger.info("\n🔍 Retrieving relevant content...")
        chunks = self.retriever.search(
            query,
            top_k=top_k,
            use_hybrid=True,
            use_reranking=True
        )

        # Step 2: Prepare text for summarization
        logger.info("\n📝 Preparing text for summarization...")
        combined_text = self._prepare_context(chunks, max_length=MAX_CONTEXT_LENGTH * 2)

        # Step 3: Generate summary
        logger.info("\n🤖 Generating summary...")
        summary = self._generate_summary(combined_text, max_length=max_length)

        generation_time = (time.time() - start_time) * 1000

        # Calculate compression ratio
        compression_ratio = len(summary) / len(combined_text) if combined_text else 0

        # Create response
        response = SummaryResponse(
            query=query,
            summary=summary,
            source_chunks=chunks,
            model_used=SUMMARIZATION_MODEL,
            compression_ratio=compression_ratio,
            generation_time_ms=generation_time,
            timestamp=datetime.now().isoformat()
        )

        # Log the summary
        self._log_summary_response(response)

        logger.info(f"\n✅ Summary generated in {generation_time:.1f}ms")
        logger.info(f"📊 Compression ratio: {compression_ratio:.2%}")

        return response

    def _prepare_context(self, chunks: List[SearchResult], max_length: int) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        current_length = 0

        for chunk in chunks:
            if current_length >= max_length:
                break

            # Add chunk text with source information
            chunk_text = f"[Source: {chunk.paper_id}]\n{chunk.text}\n"
            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)[:max_length]

    def _generate_answer(self, question: str, context: str) -> tuple[str, float]:
        """Generate answer using FLAN-T5"""

        # Prepare prompt
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

        # Tokenize
        inputs = self.qa_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_CONTEXT_LENGTH,
            truncation=True
        )

        if self.use_gpu:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate answer
        with torch.no_grad():
            outputs = self.qa_model.generate(
                **inputs,
                max_length=MAX_ANSWER_LENGTH,
                num_beams=4,
                temperature=0.7,
                do_sample=False,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode answer
        answer = self.qa_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Calculate confidence (average of token scores)
        if hasattr(outputs, 'sequences_scores'):
            confidence = float(torch.exp(outputs.sequences_scores[0]))
        else:
            confidence = 0.8  # Default confidence

        return answer, confidence

    def _generate_summary(self, text: str, max_length: int) -> str:
        """Generate summary using BART"""

        if not text.strip():
            return "No content available to summarize."

        try:
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=max_length // 4,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Error generating summary: {str(e)}"

    def _log_qa_response(self, response: QAResponse):
        """Log QA response to file for analytics"""
        log_file = QA_LOGS_DIR / f"qa_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(response.to_dict()) + '\n')

    def _log_summary_response(self, response: SummaryResponse):
        """Log summary response to file for analytics"""
        log_file = QA_LOGS_DIR / f"summaries_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(response.to_dict()) + '\n')


def format_qa_response(response: QAResponse):
    """Pretty print QA response"""
    print("\n" + "=" * 80)
    print("QUESTION & ANSWER")
    print("=" * 80)
    print(f"\n❓ Question: {response.question}")
    print(f"\n💡 Answer: {response.answer}")
    print(f"\n📊 Confidence: {response.confidence:.2%}")
    print(f"⏱️  Generation time: {response.generation_time_ms:.1f}ms")
    print(f"\n📚 Source Papers ({len(response.context_chunks)} chunks):")

    seen_papers = set()
    for chunk in response.context_chunks[:5]:
        if chunk.paper_id not in seen_papers:
            print(f"  - {chunk.paper_id}")
            seen_papers.add(chunk.paper_id)

    print("=" * 80)


def format_summary_response(response: SummaryResponse):
    """Pretty print summary response"""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n🔍 Query: {response.query}")
    print(f"\n📝 Summary:\n{response.summary}")
    print(f"\n📊 Compression: {response.compression_ratio:.2%}")
    print(f"⏱️  Generation time: {response.generation_time_ms:.1f}ms")
    print(f"\n📚 Based on {len(response.source_chunks)} relevant chunks")
    print("=" * 80)


def main():
    """Demo of RAG QA system"""
    print("\n" + "=" * 80)
    print("STEP 5: RAG-BASED QA & SUMMARIZATION DEMO")
    print("=" * 80)

    # Initialize system
    qa_system = RAGQASystem(FAISS_DIR, use_gpu=True)

    # Example 1: Question Answering
    print("\n\n" + "=" * 80)
    print("DEMO 1: QUESTION ANSWERING")
    print("=" * 80)

    question = "What are transformer attention mechanisms and how do they work?"
    response = qa_system.answer_question(question, top_k=5)
    format_qa_response(response)

    # Example 2: Summarization
    print("\n\n" + "=" * 80)
    print("DEMO 2: SUMMARIZATION")
    print("=" * 80)

    query = "recent advances in computer vision"
    summary_response = qa_system.summarize(query, top_k=10)
    format_summary_response(summary_response)


if __name__ == "__main__":
    main()
