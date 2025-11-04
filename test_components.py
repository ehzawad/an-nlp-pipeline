#!/usr/bin/env python3
"""
Quick test script for composeable components.
Tests what works without requiring all dependencies.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_ner_component():
    """Test NER component (regex-only, no dependencies)"""
    print("\n" + "="*60)
    print("TEST 1: NER Component (Regex-only)")
    print("="*60)

    try:
        from src.components.ner import NERComponent, NERInput, NERConfig

        config = NERConfig(device="cpu", enable_transformer=False)
        ner = NERComponent(config)
        await ner.initialize()

        # Test 1: NID extraction
        result = await ner.process(NERInput(
            text="আমার NID 12345678901234567"
        ))

        print(f"✓ NER initialized: {ner.get_name()}")
        print(f"✓ Extracted entities: {result.entities}")
        print(f"✓ Method: {result.extraction_method}")
        print(f"✓ Time: {result.processing_time_ms:.2f}ms")

        # Test 2: Multiple entities
        result2 = await ner.process(NERInput(
            text="Contact: john@example.com, phone 01712345678"
        ))

        print(f"✓ Multiple entities: {result2.entities}")

        # Test 3: Health check
        health = await ner.health_check()
        print(f"✓ Health check: {health['status']}")

        return True

    except Exception as e:
        print(f"✗ NER test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_base_interfaces():
    """Test base component interfaces"""
    print("\n" + "="*60)
    print("TEST 2: Base Component Interfaces")
    print("="*60)

    try:
        from src.components.base import (
            Component,
            ComponentInput,
            ComponentOutput,
            ComponentConfig,
            ComponentPipeline
        )

        # Test ComponentInput serialization
        input_data = ComponentInput(
            text="test text",
            language="bn",
            session_id="test123"
        )

        json_str = input_data.to_json()
        restored = ComponentInput.from_json(json_str)

        print(f"✓ ComponentInput serialization works")
        print(f"✓ Text: {restored.text}")
        print(f"✓ Session ID: {restored.session_id}")

        # Test ComponentOutput serialization
        output_data = ComponentOutput(
            success=True,
            data={"test": "data"},
            component_name="test"
        )

        json_str = output_data.to_json()
        print(f"✓ ComponentOutput serialization works")

        return True

    except Exception as e:
        print(f"✗ Base interfaces test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_imports():
    """Test all component imports"""
    print("\n" + "="*60)
    print("TEST 3: Component Imports")
    print("="*60)

    results = {}

    # Test NER
    try:
        from src.components.ner import NERComponent
        results['NER'] = "✓"
    except Exception as e:
        results['NER'] = f"✗ ({str(e)[:30]})"

    # Test Classification
    try:
        from src.components.classification import ClassificationComponent
        results['Classification'] = "✓"
    except Exception as e:
        results['Classification'] = f"✗ ({str(e)[:30]})"

    # Test Search
    try:
        from src.components.semantic_search import SemanticSearchComponent
        results['Semantic Search'] = "✓"
    except Exception as e:
        results['Semantic Search'] = f"✗ ({str(e)[:30]})"

    # Test LLM
    try:
        from src.components.llm import LLMComponent
        results['LLM'] = "✓"
    except Exception as e:
        results['LLM'] = f"✗ ({str(e)[:30]})"

    # Test Dialogue
    try:
        from src.components.dialogue import DialogueManager
        results['Dialogue Manager'] = "✓"
    except Exception as e:
        results['Dialogue Manager'] = f"✗ ({str(e)[:30]})"

    for component, status in results.items():
        print(f"{component:20s} : {status}")

    all_ok = all('✓' in status for status in results.values())
    return all_ok


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("COMPOSEABLE COMPONENTS - SMOKE TESTS")
    print("="*60)

    results = []

    # Run tests
    results.append(("Base Interfaces", await test_base_interfaces()))
    results.append(("Component Imports", await test_imports()))
    results.append(("NER Component", await test_ner_component()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} : {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nNote: Some components require dependencies to be installed.")
        print("Run: pip install -r requirements.txt")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
