#!/usr/bin/env python3
"""Simulate conversation from input.txt and record responses."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Project root - go up from examples/ to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.tenant_context import TenantContext, TenantConfig


def read_input_file(filepath: str) -> list[str]:
    """Read queries from input file, skipping comments and blank lines."""
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and blank lines
            if line and not line.startswith('#'):
                queries.append(line)
    return queries


async def process_conversation(queries: list[str], output_file: str):
    """Process all queries and record responses."""

    # Initialize tenant context
    tenant_config = TenantConfig(
        tenant_id="test",
        tenant_name="Test Tenant",
        enable_forms=True,
        enable_context_augmentation=True,
        enable_ner=True,
        confidence_threshold=0.7
    )
    tenant_context = TenantContext(
        tenant_id="test",
        config=tenant_config
    )

    # Get NLP service (handles both classification and search)
    nlp_service = await tenant_context.get_nlp_service()

    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CONVERSATION SIMULATION RESULTS\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total Queries: {len(queries)}\n")
        f.write("="*80 + "\n\n")

        # Process each query
        for i, query in enumerate(queries, 1):
            print(f"\nProcessing query {i}/{len(queries)}: {query}")

            f.write(f"\n{'='*80}\n")
            f.write(f"QUERY #{i}\n")
            f.write(f"{'='*80}\n")
            f.write(f"User Input: {query}\n\n")

            try:
                # Process query through NLP service
                nlp_result = await nlp_service.process(query)

                # Classification results
                f.write("--- CLASSIFICATION ---\n")
                f.write(f"Predicted Intent: {nlp_result.classification.predicted_label}\n")
                f.write(f"Confidence: {nlp_result.classification.confidence:.4f}\n")

                # Show top 3 predictions
                if nlp_result.classification.top_k_predictions:
                    f.write("\nTop 3 Predictions:\n")
                    for pred in nlp_result.classification.top_k_predictions[:3]:
                        f.write(f"  {pred[0]}: {pred[1]:.4f}\n")
                f.write("\n")

                # Semantic search results
                f.write("--- SEMANTIC SEARCH ---\n")
                f.write(f"Cluster: {nlp_result.classification.predicted_label}\n")
                f.write(f"Found {len(nlp_result.search_results)} results:\n\n")

                for j, result in enumerate(nlp_result.search_results, 1):
                    f.write(f"Result {j}:\n")
                    f.write(f"  Score: {result.score:.4f}\n")
                    f.write(f"  Question: {result.question[:100]}...\n")
                    f.write(f"  Answer: {result.answer[:150]}...\n\n")

                # Determine bot response (simple logic)
                if nlp_result.search_results:
                    best_result = nlp_result.search_results[0]
                    bot_response = best_result.answer
                    response_confidence = best_result.score

                    f.write("--- BOT RESPONSE ---\n")
                    if response_confidence > 0.8:
                        f.write("Confidence Level: HIGH\n")
                    elif response_confidence > 0.6:
                        f.write("Confidence Level: MEDIUM\n")
                    else:
                        f.write("Confidence Level: LOW\n")

                    f.write(f"\nBot: {bot_response}\n")
                else:
                    f.write("--- BOT RESPONSE ---\n")
                    f.write("Bot: দুঃখিত, আমি এই প্রশ্নের উত্তর খুঁজে পাইনি। অনুগ্রহ করে আরও বিস্তারিত বলুন।\n")

                f.write(f"\nStatus: ✅ SUCCESS\n")
                print(f"  ✅ Success - Intent: {nlp_result.classification.predicted_label}")

            except Exception as e:
                f.write(f"--- ERROR ---\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"\nStatus: ❌ FAILED\n")
                print(f"  ❌ Failed - {str(e)}")
                import traceback
                f.write(f"\nTraceback:\n{traceback.format_exc()}\n")

        # Summary
        f.write(f"\n{'='*80}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total Queries Processed: {len(queries)}\n")
        f.write(f"Completed at: {datetime.now().isoformat()}\n")

    print(f"\n✅ All queries processed! Results saved to {output_file}")


async def main():
    """Main entry point."""
    # Use paths relative to script location
    script_dir = Path(__file__).parent
    input_file = script_dir / "input.txt"
    output_file = script_dir / "output.txt"

    print(f"\n{'='*80}")
    print("CONVERSATION SIMULATION")
    print(f"{'='*80}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Check if input file exists
    if not input_file.exists():
        print(f"\n❌ Error: {input_file} not found!")
        return 1

    # Read queries
    queries = read_input_file(input_file)
    print(f"\nFound {len(queries)} queries to process")

    # Process conversation
    await process_conversation(queries, output_file)

    print(f"\n{'='*80}")
    print("✅ SIMULATION COMPLETED!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")
    print("You can review the detailed responses in the output file.")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
