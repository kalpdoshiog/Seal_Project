#!/usr/bin/env python3
"""
Step 5: Ultimate CLI - All Features in One Place

Combines:
- Basic search (FAISS + BM25)
- Question answering (RAG)
- Summarization
- World-class features (citations, multi-hop, verification)

Usage:
    python step5_ultimate_cli.py

Commands:
    search <query>      - Search for papers
    ask <question>      - Ask a question with AI-generated answer
    summarize <topic>   - Get a summary on a topic
    help                - Show help
    quit                - Exit

Author: AI Document Understanding System
Date: October 9, 2025
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step5.step5_advanced_retrieval import format_results

# Try to import world-class system, fall back to basic if models not available
try:
    from step5.step5_world_class_rag import WorldClassRAGSystem, format_world_class_response
    WORLD_CLASS_AVAILABLE = True
except ImportError:
    WORLD_CLASS_AVAILABLE = False
    print("⚠️  World-class system not available, using basic RAG")

try:
    from step5.step5_rag_qa_system import RAGQASystem, format_qa_response, format_summary_response
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG QA system not available, using search only")

BASE_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = BASE_DIR / "faiss_indices"


class UltimateCLI:
    """Ultimate CLI combining all Step 5 features"""

    def __init__(self, use_world_class: bool = True, use_gpu: bool = False):
        self.use_gpu = use_gpu

        print("\n" + "=" * 80)
        print("🚀 ULTIMATE AI RESEARCH ASSISTANT")
        print("=" * 80)
        print("\n⏰ Initializing system...")
        print("   (First run: ~3-5 minutes to build BM25 index)")
        print("   (Future runs: ~10 seconds)")
        print("=" * 80)

        # Initialize the best available system
        if use_world_class and WORLD_CLASS_AVAILABLE:
            print("\n🌟 Using WORLD-CLASS RAG system")
            self.system = WorldClassRAGSystem(FAISS_DIR, use_gpu=use_gpu)
            self.mode = "world_class"
        elif RAG_AVAILABLE:
            print("\n🤖 Using BASIC RAG system")
            self.system = RAGQASystem(FAISS_DIR, use_gpu=use_gpu)
            self.mode = "basic_rag"
        else:
            print("\n🔍 Using SEARCH-ONLY system")
            from step5.step5_advanced_retrieval import HybridRetriever
            self.system = HybridRetriever(FAISS_DIR, use_gpu=use_gpu)
            self.mode = "search_only"

        print("\n✅ System ready!")

    def search(self, query: str, top_k: int = 10):
        """Search for papers"""
        print(f"\n🔍 Searching for: '{query}'")

        results = self.system.retriever.search(
            query,
            top_k=top_k,
            use_hybrid=True,
            use_reranking=True
        ) if hasattr(self.system, 'retriever') else self.system.search(
            query,
            top_k=top_k,
            use_hybrid=True,
            use_reranking=True
        )

        format_results(results, show_text=True)

    def ask(self, question: str):
        """Ask a question and get AI-generated answer"""
        if self.mode == "search_only":
            print("❌ QA not available in search-only mode")
            print("💡 Try: search <query> instead")
            return

        print(f"\n❓ Answering: '{question}'")

        if self.mode == "world_class":
            response = self.system.answer_question(
                question,
                use_multi_hop=True,
                use_self_reflection=True,
                top_k=5
            )
            format_world_class_response(response)
        else:
            response = self.system.answer_question(question, top_k=5)
            format_qa_response(response)

    def summarize(self, topic: str):
        """Generate a summary on a topic"""
        if self.mode == "search_only":
            print("❌ Summarization not available in search-only mode")
            print("💡 Try: search <query> instead")
            return

        print(f"\n📝 Summarizing: '{topic}'")

        response = self.system.summarize(topic, top_k=10)
        format_summary_response(response)

    def interactive(self):
        """Interactive mode"""
        print("\n" + "=" * 80)
        print("📚 INTERACTIVE MODE")
        print("=" * 80)

        self.print_help()

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
                    break

                elif command == 'help':
                    self.print_help()

                elif command == 'search':
                    if not query:
                        print("❌ Usage: search <query>")
                        continue
                    self.search(query)

                elif command == 'ask':
                    if not query:
                        print("❌ Usage: ask <question>")
                        continue
                    self.ask(query)

                elif command == 'summarize':
                    if not query:
                        print("❌ Usage: summarize <topic>")
                        continue
                    self.summarize(query)

                else:
                    # Default to search
                    self.search(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def print_help(self):
        """Print help message"""
        print("\n📖 COMMANDS:")
        print("  search <query>      - Search for papers (e.g., 'search transformers')")

        if self.mode != "search_only":
            print("  ask <question>      - Ask a question (e.g., 'ask what are GANs?')")
            print("  summarize <topic>   - Get summary (e.g., 'summarize BERT models')")

        print("  help                - Show this help")
        print("  quit / exit         - Exit the program")

        print("\n💡 FEATURES:")
        if self.mode == "world_class":
            print("  ✅ Semantic + Keyword hybrid search")
            print("  ✅ Question answering with inline citations [1][2]")
            print("  ✅ Multi-hop reasoning for complex questions")
            print("  ✅ Summarization")
            print("  ✅ Hallucination detection & verification")
            print("  ✅ Query optimization (automatic rewrites)")
            print("  ✅ Smart caching (300x faster for repeated queries)")
        elif self.mode == "basic_rag":
            print("  ✅ Semantic + Keyword hybrid search")
            print("  ✅ Question answering")
            print("  ✅ Summarization")
            print("  ✅ Cross-encoder reranking")
        else:
            print("  ✅ Semantic + Keyword hybrid search")
            print("  ✅ Cross-encoder reranking")

        print("\n🎯 QUICK START:")
        print("  1. Try: search deep learning")
        if self.mode != "search_only":
            print("  2. Try: ask What is BERT?")
            print("  3. Try: summarize transformer models")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Ultimate AI Research Assistant - All Features'
    )
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Use basic RAG instead of world-class'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    parser.add_argument(
        'command',
        nargs='*',
        help='Command to run (e.g., "search transformers")'
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = UltimateCLI(
        use_world_class=not args.basic,
        use_gpu=not args.no_gpu
    )

    # If command provided, run it
    if args.command:
        command = args.command[0].lower()
        query = ' '.join(args.command[1:]) if len(args.command) > 1 else ' '.join(args.command)

        if command == 'search':
            cli.search(query)
        elif command == 'ask':
            cli.ask(query)
        elif command == 'summarize':
            cli.summarize(query)
        else:
            # Treat whole thing as search query
            cli.search(' '.join(args.command))
    else:
        # Interactive mode
        cli.interactive()


if __name__ == "__main__":
    main()

