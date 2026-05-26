"""
Finance Policy Assistant - CLI Entry Point
Run interactive Q&A or batch demo queries against finance policy documents.

Usage:
    python main.py                    # Interactive mode
    python main.py --demo             # Run predefined demo queries
    python main.py --query "..."      # Single query mode
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure OpenAI API key is present
if not os.getenv("OPENAI_API_KEY"):
    print("[ERROR] OPENAI_API_KEY is not set. Please add it to your .env file.")
    sys.exit(1)

from src.rag_pipeline import DocumentIngestionPipeline, FinancePolicyRAG

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         AI-Powered Finance Policy Assistant                  ║
║         Retrieval-Augmented Generation (RAG) System          ║
╚══════════════════════════════════════════════════════════════╝
"""

DEMO_QUERIES = [
    "What is the maximum reimbursement limit for domestic business travel?",
    "What approvals are required for vendor payments above $10,000?",
    "Can employees claim international travel expenses? What are the conditions?",
    "What documents are mandatory for a procurement request?",
    "What is the policy for meal expense reimbursement during client meetings?",
    "How long does it take to process a reimbursement request?",
    "Are personal travel expenses combined with business trips reimbursable?",
    "What is the process for emergency cash advances for travel?",
    "What are the tax compliance requirements for vendor payments?",
    "What receipts are required for expense claims under $25?",
]


def build_rag_system(docs_dir: str = "data/documents") -> FinancePolicyRAG:
    """Initialize ingestion pipeline and RAG system."""
    print(BANNER)
    print("Initializing Finance Policy Assistant...")

    pipeline = DocumentIngestionPipeline(
        docs_dir=docs_dir,
        chunk_size=800,
        chunk_overlap=150,
    )
    vector_store = pipeline.ingest()

    rag = FinancePolicyRAG(
        vector_store=vector_store,
        llm_model="gpt-4o-mini",
        top_k=4,
        temperature=0.0,
    )
    print("\nSystem ready. Finance Policy Assistant is online.\n")
    return rag


def run_interactive(rag: FinancePolicyRAG):
    """Interactive CLI loop for employee queries."""
    print("Type your finance policy question below.")
    print("Commands: 'quit' or 'exit' to stop | 'demo' to run sample queries\n")

    while True:
        try:
            user_input = input("Your Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Finance Policy Assistant. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting Finance Policy Assistant. Goodbye!")
            break
        if user_input.lower() == "demo":
            run_demo(rag)
            continue

        try:
            result = rag.query(user_input)
            print(rag.format_response(result))
        except Exception as e:
            print(f"\n[ERROR] Failed to process query: {e}\n")


def run_demo(rag: FinancePolicyRAG):
    """Run all predefined demo queries and print results."""
    print("\n" + "=" * 70)
    print("RUNNING DEMO QUERIES")
    print("=" * 70)
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\nDemo Query {i}/{len(DEMO_QUERIES)}")
        try:
            result = rag.query(query)
            print(rag.format_response(result))
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")


def run_single_query(rag: FinancePolicyRAG, question: str):
    """Run a single query and print the result."""
    result = rag.query(question)
    print(rag.format_response(result))


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Finance Policy Assistant (RAG)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run predefined demo queries"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single finance policy query"
    )
    parser.add_argument(
        "--docs-dir", type=str, default="data/documents",
        help="Path to the finance documents directory (default: data/documents)"
    )
    args = parser.parse_args()

    rag = build_rag_system(docs_dir=args.docs_dir)

    if args.demo:
        run_demo(rag)
    elif args.query:
        run_single_query(rag, args.query)
    else:
        run_interactive(rag)


if __name__ == "__main__":
    main()
