#!/usr/bin/env python3
"""
Step 3: Text Preprocessing & Cleaning (CPU-OPTIMIZED with NER)
Minimal preprocessing optimized for transformer-based embedding models
Quality-focused with Named Entity Recognition enabled
Author: AI Document Understanding System
Date: October 8, 2025
"""

import os
import sys
import json
import logging
import re
import spacy
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from tqdm import tqdm

# ----------------------------
# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
EXTRACTED_DIR = BASE_DIR / "extracted_text"
PREPROCESSED_DIR = BASE_DIR / "preprocessed_text"
ENTITIES_DIR = BASE_DIR / "extracted_entities"
PREPROCESSING_LOG = BASE_DIR / "preprocessing_results.csv"
LOGS_DIR = BASE_DIR / "logs"

PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Logging Setup
LOG_FILE = LOGS_DIR / f"preprocessing_cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
# CPU Configuration
NUM_WORKERS = 4  # Optimal for most CPUs
ENABLE_NER = True  # Named Entity Recognition - QUALITY OVER SPEED
MIN_WORDS_THRESHOLD = 100  # Minimum words to keep document
MAX_TEXT_LENGTH = 500000  # Max characters per document (for memory)

# Preprocessing mode
PREPROCESSING_MODE = "embedding_optimized_cpu_ner"

# Common footer/header patterns in academic papers
NOISE_PATTERNS = [
    r'^\s*\d+\s*$',  # Page numbers alone on a line
    r'^[\*\-=_]{5,}$',  # Long separator lines
    r'^\s*Page\s+\d+\s*(of\s+\d+)?\s*$',  # "Page X" or "Page X of Y"
    r'^\s*arXiv:\d+\.\d+v\d+\s+\[.*?\]\s+\d+\s+\w+\s+\d{4}\s*$',  # arXiv headers
]

# Global spaCy model (loaded once per process)
nlp = None

def load_spacy_model():
    """Load spaCy model for CPU processing"""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")

            # Optimize pipeline - keep only necessary components for quality NER
            if "ner" in nlp.pipe_names:
                # Disable unnecessary components but keep tok2vec for quality
                pipes_to_disable = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
                # Only disable if they exist
                pipes_to_disable = [pipe for pipe in pipes_to_disable if pipe in nlp.pipe_names]
                if pipes_to_disable:
                    nlp.disable_pipes(*pipes_to_disable)
                    logger.info(f"✓ Disabled pipes for speed: {pipes_to_disable}")
                logger.info(f"✓ Active pipes: {nlp.pipe_names}")

        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            nlp = None
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            nlp = None

    return nlp


def remove_noise_patterns(text):
    """
    Remove ONLY clear noise (page numbers, separator lines)
    Keep most text intact for embedding models
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip truly empty lines (but keep lines with just spaces initially)
        if len(line.strip()) == 0:
            cleaned_lines.append('')  # Preserve paragraph breaks
            continue

        # Check against noise patterns
        is_noise = False
        for pattern in NOISE_PATTERNS:
            if re.match(pattern, line.strip()):
                is_noise = True
                break

        if not is_noise:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_whitespace(text):
    """
    Gentle whitespace normalization - preserves structure
    Does NOT lowercase or remove punctuation (critical for embeddings!)
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)

    # Replace more than 3 newlines with 2 (preserve paragraph breaks)
    text = re.sub(r'\n{4,}', '\n\n', text)

    # Remove spaces at start/end of lines
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def remove_references_section(text):
    """
    Optional: Remove references section (often very long and not useful for embeddings)
    This is debatable - references can provide context
    """
    # Look for common reference section headers
    ref_patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
    ]

    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Keep everything before references
            return text[:match.start()]

    return text


def clean_for_embeddings(text):
    """
    MINIMAL cleaning optimized for transformer embedding models

    What we DO:
    - Remove page numbers and obvious noise
    - Normalize excessive whitespace
    - Fix encoding issues
    - Remove references section (optional)

    What we DON'T do (unlike traditional preprocessing):
    - Lowercase (case matters: "Apple" vs "apple")
    - Remove punctuation (grammatical context matters)
    - Stemming/lemmatization (not needed for transformers)
    - Aggressive tokenization (transformers have their own)
    """
    # Step 1: Fix encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    # Step 2: Remove clear noise patterns
    text = remove_noise_patterns(text)

    # Step 3: Gentle whitespace normalization
    text = normalize_whitespace(text)

    # Step 4: Optional - remove references section
    # Uncomment if you want to exclude references
    # text = remove_references_section(text)

    return text


def extract_named_entities(text, paper_id):
    """Extract named entities using spaCy NER - QUALITY FOCUSED"""
    nlp_model = load_spacy_model()

    if nlp_model is None or not ENABLE_NER:
        return None

    try:
        # Process first 100k chars for NER (balance quality vs performance)
        text_sample = text[:100000]
        doc = nlp_model(text_sample)

        # Extract entities by type
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'PRODUCT': [],
            'WORK_OF_ART': [],
            'DATE': [],
            'LAW': [],
            'LANGUAGE': [],
            'NORP': [],
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        # Count frequencies and get unique entities
        entity_counts = {}
        unique_entities = {}

        for label, ents in entities.items():
            unique_texts = list(set([e['text'] for e in ents]))
            entity_counts[label] = len(ents)
            unique_entities[label] = unique_texts[:30]  # Top 30 unique for quality

        return {
            'paper_id': paper_id,
            'entity_counts': entity_counts,
            'unique_entities': unique_entities,
            'total_entities': sum(entity_counts.values())
        }

    except Exception as e:
        logger.warning(f"NER failed for {paper_id}: {e}")
        return None


def get_text_statistics(text):
    """Calculate text statistics"""
    # Word count (simple split)
    words = text.split()
    word_count = len(words)

    # Character count
    char_count = len(text)

    # Unique words (case-insensitive for stats only)
    unique_words = len(set(word.lower() for word in words))

    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if len(s.strip()) > 10])

    return {
        'char_count': char_count,
        'word_count': word_count,
        'unique_words': unique_words,
        'sentence_count': sentence_count,
        'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
        'vocabulary_richness': unique_words / word_count if word_count > 0 else 0
    }


def process_single_file(txt_file):
    """Process a single text file with NER"""
    try:
        paper_id = txt_file.stem

        # Check if already processed
        output_file = PREPROCESSED_DIR / f"{paper_id}.txt"
        meta_file = PREPROCESSED_DIR / f"{paper_id}.meta.json"

        if output_file.exists() and meta_file.exists():
            return {
                'paper_id': paper_id,
                'status': 'skipped',
                'reason': 'already_processed',
                'word_count': 0,
                'char_count': 0,
                'unique_words': 0,
                'entities_found': 0
            }

        # Read original text
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        original_length = len(text)

        # Clean for embeddings (minimal preprocessing)
        cleaned_text = clean_for_embeddings(text)

        # Get statistics
        stats = get_text_statistics(cleaned_text)

        # Check minimum threshold
        if stats['word_count'] < MIN_WORDS_THRESHOLD:
            return {
                'paper_id': paper_id,
                'status': 'failed',
                'reason': f'too_short_{stats["word_count"]}_words',
                'word_count': stats['word_count'],
                'char_count': stats['char_count'],
                'unique_words': 0,
                'entities_found': 0
            }

        # Extract named entities (QUALITY FOCUSED)
        entities_data = None
        if ENABLE_NER:
            entities_data = extract_named_entities(cleaned_text, paper_id)

        # Save preprocessed text (PRESERVE ORIGINAL CASE AND PUNCTUATION!)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        # Save metadata
        metadata = {
            'paper_id': paper_id,
            'processing_date': datetime.now().isoformat(),
            'preprocessing_mode': PREPROCESSING_MODE,
            'original_length': original_length,
            'statistics': stats,
            'ner_enabled': ENABLE_NER,
            'entity_counts': entities_data['entity_counts'] if entities_data else {}
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Save entities separately if NER was performed
        if entities_data and entities_data['total_entities'] > 0:
            entities_file = ENTITIES_DIR / f"{paper_id}_entities.json"
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2)

        return {
            'paper_id': paper_id,
            'status': 'success',
            'word_count': stats['word_count'],
            'char_count': stats['char_count'],
            'unique_words': stats['unique_words'],
            'entities_found': entities_data['total_entities'] if entities_data else 0
        }

    except Exception as e:
        logger.error(f"Error processing {txt_file.name}: {e}")
        return {
            'paper_id': txt_file.stem,
            'status': 'failed',
            'reason': str(e),
            'word_count': 0,
            'char_count': 0,
            'unique_words': 0,
            'entities_found': 0
        }


def main():
    """Main preprocessing pipeline - CPU optimized with NER for quality"""
    logger.info("=" * 80)
    logger.info("Step 3: Text Preprocessing & Cleaning (CPU with NER - QUALITY FOCUSED)")
    logger.info("=" * 80)

    logger.info(f"Mode: {PREPROCESSING_MODE}")
    logger.info(f"CPU Workers: {NUM_WORKERS}")
    logger.info(f"NER Enabled: {ENABLE_NER} ✓ QUALITY OVER SPEED")
    logger.info("Optimized for transformer-based embedding models")
    logger.info("Preserving case, punctuation, and natural text structure")
    logger.info("=" * 80)

    # Find all extracted text files
    txt_files = list(EXTRACTED_DIR.glob("*.txt"))
    total_files = len(txt_files)

    logger.info(f"Found {total_files} text files to preprocess")

    if total_files == 0:
        logger.error("No text files found in extracted_text/")
        return

    # Load spaCy model once (will be loaded per process in multiprocessing)
    if ENABLE_NER:
        logger.info("Loading spaCy model for NER...")
        load_spacy_model()
        logger.info(f"✓ spaCy model loaded: en_core_web_sm")

    # Initialize CSV log
    csv_file = open(PREPROCESSING_LOG, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'paper_id', 'status', 'word_count', 'char_count',
        'unique_words', 'entities_found', 'reason'
    ])
    csv_writer.writeheader()

    # Process files
    results = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    logger.info(f"Starting CPU preprocessing with {NUM_WORKERS} workers...")
    logger.info(f"Minimum words threshold: {MIN_WORDS_THRESHOLD}")

    # Process with multiprocessing for CPU efficiency
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, txt_file): txt_file for txt_file in txt_files}

        # Process results with progress bar
        with tqdm(total=total_files, desc="Processing", unit="file") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

                # Write to CSV
                csv_writer.writerow(result)
                csv_file.flush()

                # Update counts
                if result['status'] == 'success':
                    success_count += 1
                elif result['status'] == 'skipped':
                    skipped_count += 1
                else:
                    failed_count += 1

                pbar.update(1)
                pbar.set_postfix({
                    'Success': success_count,
                    'Skipped': skipped_count,
                    'Failed': failed_count
                })

    csv_file.close()

    # Calculate statistics
    successful_results = [r for r in results if r['status'] == 'success']

    if successful_results:
        avg_words = sum(r['word_count'] for r in successful_results) / len(successful_results)
        avg_chars = sum(r['char_count'] for r in successful_results) / len(successful_results)
        avg_unique = sum(r['unique_words'] for r in successful_results) / len(successful_results)
        total_entities = sum(r['entities_found'] for r in successful_results)
    else:
        avg_words = avg_chars = avg_unique = total_entities = 0

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Files: {total_files}")
    logger.info(f"Successfully Processed: {success_count}")
    logger.info(f"Skipped (already processed): {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success Rate: {(success_count / (total_files - skipped_count) * 100 if total_files > skipped_count else 0):.2f}%")
    logger.info("\nStatistics (Processed Files):")
    logger.info(f"  Average Words: {avg_words:.0f}")
    logger.info(f"  Average Characters: {avg_chars:.0f}")
    logger.info(f"  Average Unique Words: {avg_unique:.0f}")
    if ENABLE_NER:
        logger.info(f"  Total Entities Extracted: {total_entities}")
        logger.info(f"  Avg Entities per Paper: {total_entities / len(successful_results) if successful_results else 0:.1f}")
    logger.info(f"\nOutput saved to: {PREPROCESSED_DIR}")
    logger.info(f"Results log: {PREPROCESSING_LOG}")
    logger.info(f"Processing log: {LOG_FILE}")
    logger.info("\n" + "=" * 80)
    logger.info("Text is optimized for embedding models:")
    logger.info("  ✓ Original case preserved")
    logger.info("  ✓ Punctuation preserved")
    logger.info("  ✓ Natural text structure maintained")
    logger.info("  ✓ Named Entity Recognition completed")
    logger.info("  ✓ Ready for sentence-transformers")
    logger.info("  ✓ CPU-optimized quality processing")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

