"""
Demo: Composeable NLP Components

This example demonstrates:
1. Running components independently
2. Composing components together
3. Using the Dialogue Manager
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.ner import NERComponent, NERInput, NERConfig
from src.components.classification import ClassificationComponent, ClassificationInput, ClassificationConfig
from src.components.semantic_search import SemanticSearchComponent, SemanticSearchInput, SemanticSearchConfig
from src.components.dialogue import DialogueManager, DialogueInput, DialogueConfig


async def demo_independent_components():
    """Demo: Running components independently"""
    print("=" * 60)
    print("Demo 1: Running Components Independently")
    print("=" * 60)

    # Test text
    text = "আমার নাম জন ডো এবং আমার NID 12345678901234567"

    # 1. NER Component
    print("\n1. NER Component:")
    ner_config = NERConfig(device="cpu", enable_transformer=False)  # Regex-only for demo
    ner = NERComponent(ner_config)
    await ner.initialize()

    ner_result = await ner.process(NERInput(text=text))
    print(f"   Entities: {ner_result.entities}")
    print(f"   Method: {ner_result.extraction_method}")
    print(f"   Time: {ner_result.processing_time_ms:.2f}ms")

    # 2. Classification Component
    print("\n2. Classification Component:")
    cls_config = ClassificationConfig(device="cpu")
    classifier = ClassificationComponent(cls_config)
    await classifier.initialize()

    cls_result = await classifier.process(ClassificationInput(text="আমার NID স্ট্যাটাস কি?"))
    print(f"   Intent: {cls_result.top_cluster}")
    print(f"   Confidence: {cls_result.top_confidence:.2f}")
    print(f"   Time: {cls_result.processing_time_ms:.2f}ms")

    # 3. Semantic Search Component
    print("\n3. Semantic Search Component:")
    search_config = SemanticSearchConfig(device="cpu")
    searcher = SemanticSearchComponent(search_config)
    await searcher.initialize()

    search_result = await searcher.process(SemanticSearchInput(text="NID কিভাবে আবেদন করবো?"))
    print(f"   Results: {len(search_result.results)}")
    if search_result.results:
        print(f"   Top Result: {search_result.results[0].question[:50]}...")
        print(f"   Score: {search_result.results[0].score:.2f}")
    print(f"   Time: {search_result.processing_time_ms:.2f}ms")


async def demo_component_pipeline():
    """Demo: Composing components in a pipeline"""
    print("\n" + "=" * 60)
    print("Demo 2: Component Pipeline")
    print("=" * 60)

    from src.components.base import ComponentPipeline

    # Create components
    ner = NERComponent(NERConfig(device="cpu", enable_transformer=False))
    classifier = ClassificationComponent(ClassificationConfig(device="cpu"))
    searcher = SemanticSearchComponent(SemanticSearchConfig(device="cpu"))

    # Create pipeline
    pipeline = ComponentPipeline([ner, classifier, searcher])
    await pipeline.initialize_all()

    # Process through pipeline
    text = "আমার NID নাম্বার 12345678901234567, আমার স্ট্যাটাস কি?"

    from src.components.base import ComponentInput
    input_data = ComponentInput(text=text)

    print(f"\nProcessing: {text}")
    outputs = await pipeline.process(input_data)

    print(f"\nPipeline Results:")
    for output in outputs:
        print(f"  - {output.component_name}: {output.success}, {output.processing_time_ms:.2f}ms")


async def demo_dialogue_manager():
    """Demo: Using Dialogue Manager"""
    print("\n" + "=" * 60)
    print("Demo 3: Dialogue Manager")
    print("=" * 60)

    # Create dialogue manager
    config = DialogueConfig(
        device="cpu",
        enable_ner=True,
        enable_classification=True,
        enable_search=True,
        enable_llm=False,  # Disable LLM for demo
    )

    # Override component configs to use lightweight settings
    config.ner_config = NERConfig(device="cpu", enable_transformer=False)

    dm = DialogueManager(config)
    await dm.initialize()

    # Multi-turn conversation
    queries = [
        "আমার NID নাম্বার 12345678901234567",
        "আমার স্ট্যাটাস কি?",  # Fractional query
        "কিভাবে NID সংশোধন করবো?",
    ]

    session_id = "demo_session"

    for i, query in enumerate(queries, 1):
        print(f"\nTurn {i}:")
        print(f"  User: {query}")

        result = await dm.process(DialogueInput(text=query, session_id=session_id))

        if result.success:
            print(f"  Bot: {result.response}")
            print(f"  Entities: {result.entities}")
            print(f"  Intent: {result.intent} ({result.confidence:.2f})")
            print(f"  Time: {result.processing_time_ms:.2f}ms")
        else:
            print(f"  Error: {result.error}")

    # Check session state
    session = dm.get_session(session_id)
    print(f"\nSession Info:")
    print(f"  Messages: {len(session.messages)}")
    print(f"  Accumulated Entities: {session.entities}")
    print(f"  Current Intent: {session.current_intent}")


async def main():
    """Run all demos"""
    try:
        await demo_independent_components()
        await demo_component_pipeline()
        await demo_dialogue_manager()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running demos: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
