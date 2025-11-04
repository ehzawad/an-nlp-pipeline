#!/usr/bin/env python3
"""
Unified CLI for all NLP components.

Usage:
    python cli.py ner --text "আমার নাম জন ডো"
    python cli.py classification --text "আমার NID স্ট্যাটাস কি?"
    python cli.py search --text "NID কিভাবে আবেদন করবো?"
    python cli.py llm --text "Summarize this text"
    python cli.py dialogue --interactive
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


async def run_ner(args):
    """Run NER component"""
    from src.components.ner.component import main
    await main()


async def run_classification(args):
    """Run Classification component"""
    from src.components.classification.component import main
    await main()


async def run_search(args):
    """Run Semantic Search component"""
    from src.components.semantic_search.component import main
    await main()


async def run_llm(args):
    """Run LLM component"""
    from src.components.llm.component import main
    await main()


async def run_dialogue(args):
    """Run Dialogue Manager"""
    from src.components.dialogue.manager import main
    await main()


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified CLI for NLP Components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="component", help="Component to run")

    # NER
    ner_parser = subparsers.add_parser("ner", help="Named Entity Recognition")
    ner_parser.add_argument("--text", type=str, help="Text to process")
    ner_parser.add_argument("--input-file", type=str, help="Input JSON file")
    ner_parser.add_argument("--output-file", type=str, help="Output JSON file")
    ner_parser.add_argument("--device", type=str, default="cpu")
    ner_parser.add_argument("--regex-only", action="store_true")

    # Classification
    cls_parser = subparsers.add_parser("classification", help="Intent Classification")
    cls_parser.add_argument("--text", type=str, help="Text to classify")
    cls_parser.add_argument("--input-file", type=str, help="Input JSON file")
    cls_parser.add_argument("--output-file", type=str, help="Output JSON file")
    cls_parser.add_argument("--device", type=str, default="cpu")
    cls_parser.add_argument("--top-k", type=int, default=3)

    # Semantic Search
    search_parser = subparsers.add_parser("search", help="Semantic Search")
    search_parser.add_argument("--text", type=str, help="Text to search")
    search_parser.add_argument("--input-file", type=str, help="Input JSON file")
    search_parser.add_argument("--output-file", type=str, help="Output JSON file")
    search_parser.add_argument("--device", type=str, default="cpu")
    search_parser.add_argument("--cluster", type=str, help="Cluster to search in")
    search_parser.add_argument("--top-k", type=int, default=5)

    # LLM
    llm_parser = subparsers.add_parser("llm", help="Language Model Generation")
    llm_parser.add_argument("--text", type=str, help="Text/prompt")
    llm_parser.add_argument("--input-file", type=str, help="Input JSON file")
    llm_parser.add_argument("--output-file", type=str, help="Output JSON file")
    llm_parser.add_argument("--backend", type=str, default="local", choices=["local", "openai", "anthropic"])
    llm_parser.add_argument("--model-name", type=str, default="google/flan-t5-base")
    llm_parser.add_argument("--device", type=str, default="cpu")
    llm_parser.add_argument("--max-tokens", type=int, default=256)

    # Dialogue
    dialogue_parser = subparsers.add_parser("dialogue", help="Dialogue Manager")
    dialogue_parser.add_argument("--text", type=str, help="User input")
    dialogue_parser.add_argument("--session-id", type=str, help="Session ID")
    dialogue_parser.add_argument("--input-file", type=str, help="Input JSON file")
    dialogue_parser.add_argument("--output-file", type=str, help="Output JSON file")
    dialogue_parser.add_argument("--device", type=str, default="cpu")
    dialogue_parser.add_argument("--enable-llm", action="store_true")
    dialogue_parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    if not args.component:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate component
    if args.component == "ner":
        asyncio.run(run_ner(args))
    elif args.component == "classification":
        asyncio.run(run_classification(args))
    elif args.component == "search":
        asyncio.run(run_search(args))
    elif args.component == "llm":
        asyncio.run(run_llm(args))
    elif args.component == "dialogue":
        asyncio.run(run_dialogue(args))
    else:
        print(f"Unknown component: {args.component}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
