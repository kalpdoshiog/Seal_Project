#!/usr/bin/env python3
"""
Step 5: Complete Interactive CLI - Search, QA, and Summarization

Features:
- Semantic search across 12,108 papers
- Question answering with RAG
- Automatic summarization
- Full logging and analytics

Usage:
    python step5_complete_cli.py

Author: AI Document Understanding System
Date: October 9, 2025
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step5.step5_rag_qa_system import RAGQASystem, format_qa_response, format_summary_response
from step5.step5_advanced_retrieval import format_results

BASE_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = BASE_DIR / "faiss_indices"


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 80)
    print("🤖 AI RESEARCH ASSISTANT - Interactive Mode")
    print("=" * 80)
    print("\n📚 System: 12,108 papers | 1,210,800 chunks indexed")
    print("🔍 Features: Search | Question Answering | Summarization")
    print("=" * 80)


def print_help():
    """Print help message"""
    print("\n📖 COMMANDS:")
    print("  search <query>     - Search for papers (e.g., 'search transformers')")
    print("  ask <question>     - Ask a question (e.g., 'ask what are GANs?')")
    print("  summarize <topic>  - Get summary (e.g., 'summarize BERT models')")
    print("  help               - Show this help")
    print("  quit / exit        - Exit the program")
    print("\n💡 TIPS:")
    print("  - Be specific in questions for better answers")
    print("  - Summaries work best with broader topics")
    print("  - All interactions are logged for analytics")
    print("=" * 80)


def interactive_mode(qa_system: RAGQASystem):
    """Interactive mode with search, QA, and summarization"""

    print_header()
    print_help()

    while True:
        try:
            # Get user input
            print("\n" + "-" * 80)
            user_input = input("🤖 > ").strip()

            if not user_input:
                continue

            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            query = parts[1] if len(parts) > 1 else ""

            # Handle commands
            if command in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using AI Research Assistant!")
                print("📊 Check logs/ folder for query analytics")
                break

            elif command == 'help':
                print_help()

            elif command == 'search':
                if not query:
                    print("❌ Usage: search <query>")
                    continue

                print(f"\n🔍 Searching for: '{query}'")
                results = qa_system.retriever.search(
                    query,
                    top_k=10,
                    use_hybrid=True,
                    use_reranking=True
                )
                format_results(results, show_text=True)

            elif command == 'ask':
                if not query:
                    print("❌ Usage: ask <question>")
                    continue

                print(f"\n❓ Answering question: '{query}'")
                response = qa_system.answer_question(query, top_k=5)
                format_qa_response(response)

            elif command == 'summarize':
                if not query:
                    print("❌ Usage: summarize <topic>")
                    continue

                print(f"\n📝 Summarizing: '{query}'")
                response = qa_system.summarize(query, top_k=10)
                format_summary_response(response)

            else:
                # Default to search if no command recognized
                print(f"\n🔍 Searching for: '{user_input}'")
                results = qa_system.retriever.search(
                    user_input,
                    top_k=10,
                    use_hybrid=True,
                    use_reranking=True
                )
                format_results(results, show_text=True)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue


def quick_test():
    """Quick test of all features"""
    print("\n" + "=" * 80)
    print("🚀 QUICK TEST - All Features")
    print("=" * 80)

    print("\n⏰ Initializing AI Research Assistant...")
    print("   (First run: ~3-5 minutes to build BM25 index)")
    print("   (Future runs: ~10 seconds)")

    qa_system = RAGQASystem(FAISS_DIR, use_gpu=False)

    print("\n" + "=" * 80)
    print("✅ SYSTEM READY - Running Test Queries")
    print("=" * 80)

    # Test 1: Search
    print("\n\n📌 TEST 1: SEMANTIC SEARCH")
    print("-" * 80)
    results = qa_system.retriever.search(
        "transformer attention mechanisms",
        top_k=3,
        use_hybrid=True,
        use_reranking=True
    )
    format_results(results, show_text=True)

    # Test 2: Question Answering
    print("\n\n📌 TEST 2: QUESTION ANSWERING")
    print("-" * 80)
    response = qa_system.answer_question(
        "What is BERT and how does it work?",
        top_k=3
    )
    format_qa_response(response)

    # Test 3: Summarization
    print("\n\n📌 TEST 3: SUMMARIZATION")
    print("-" * 80)
    summary = qa_system.summarize(
        "recent advances in neural networks",
        top_k=5,
        max_length=200
    )
    format_summary_response(summary)

    print("\n\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 80)
    print("\n📊 Logs saved to: logs/qa_logs/")
    print("\n🎯 To use interactively, run:")
    print("   python step5/step5_complete_cli.py")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='AI Research Assistant - Search, QA, and Summarization'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick test of all features'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )

    args = parser.parse_args()

    if args.test:
        # Quick test mode
        quick_test()
    else:
        # Interactive mode
        print("\n⏰ Initializing AI Research Assistant...")
        print("   (First run takes 3-5 minutes to build BM25 index)")
        print("   (This is a one-time initialization)")

        qa_system = RAGQASystem(FAISS_DIR, use_gpu=not args.no_gpu)
        interactive_mode(qa_system)


if __name__ == "__main__":
    main()

