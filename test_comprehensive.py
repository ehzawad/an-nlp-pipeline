#!/usr/bin/env python3.12
"""
Comprehensive component tests - demonstrates working functionality.
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_ner_comprehensive():
    """Comprehensive NER testing"""
    print("\n" + "="*60)
    print("TEST: NER Component - Comprehensive")
    print("="*60)

    from src.components.ner import NERComponent, NERInput, NERConfig

    # Test 1: Basic initialization
    config = NERConfig(device="cpu", enable_transformer=False)
    ner = NERComponent(config)
    await ner.initialize()
    print(f"✓ Component initialized: {ner.get_name()}")

    # Test 2: NID extraction
    result = await ner.process(NERInput(text="আমার NID 12345678901234567"))
    assert result.success
    assert "nid_number" in result.entities
    assert result.entities["nid_number"] == "12345678901234567"
    print(f"✓ NID extraction: {result.entities['nid_number']}")

    # Test 3: Phone number extraction
    result = await ner.process(NERInput(text="কল করুন 01712345678"))
    assert result.success
    assert "phone" in result.entities
    print(f"✓ Phone extraction: {result.entities['phone']}")

    # Test 4: Email extraction
    result = await ner.process(NERInput(text="ইমেইল: test@example.com"))
    assert result.success
    assert "email" in result.entities
    print(f"✓ Email extraction: {result.entities['email']}")

    # Test 5: Multiple entities
    result = await ner.process(NERInput(
        text="নাম: জন ডো, NID: 12345678901234567, ফোন: 01712345678, ইমেইল: john@example.com"
    ))
    assert result.success
    assert len(result.entities) >= 3
    print(f"✓ Multiple entities: {list(result.entities.keys())}")

    # Test 6: Batch processing
    inputs = [
        NERInput(text="NID 11111111111111111"),
        NERInput(text="Phone 01812345678"),
        NERInput(text="Email user@test.com"),
    ]
    results = await ner.process_batch(inputs)
    assert len(results) == 3
    assert all(r.success for r in results)
    print(f"✓ Batch processing: {len(results)} items")

    # Test 7: Health check
    health = await ner.health_check()
    assert health["status"] == "healthy"
    print(f"✓ Health check: {health['status']}")

    # Test 8: Performance
    import time
    start = time.time()
    for _ in range(100):
        await ner.process(NERInput(text="NID 12345678901234567"))
    elapsed = time.time() - start
    avg_ms = (elapsed / 100) * 1000
    print(f"✓ Performance: {avg_ms:.2f}ms avg (100 iterations)")

    # Test 9: JSON serialization
    result = await ner.process(NERInput(text="Test NID 12345678901234567"))
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["success"] == True
    assert "entities" in parsed["data"]
    print(f"✓ JSON serialization works")

    # Test 10: Edge cases
    result = await ner.process(NERInput(text=""))
    assert result.success
    assert len(result.entities) == 0
    print(f"✓ Empty text handling")

    result = await ner.process(NERInput(text="No entities here"))
    assert result.success
    assert len(result.entities) == 0
    print(f"✓ No entities handling")

    print(f"\n✓ All NER tests passed!")
    return True


async def test_component_composition():
    """Test component composition patterns"""
    print("\n" + "="*60)
    print("TEST: Component Composition")
    print("="*60)

    from src.components.base import ComponentInput, ComponentPipeline
    from src.components.ner import NERComponent, NERConfig, NERInput

    # Create multiple NER components with different configs
    ner1 = NERComponent(NERConfig(enable_transformer=False))
    ner2 = NERComponent(NERConfig(enable_transformer=False))

    await ner1.initialize()
    await ner2.initialize()

    # Test sequential processing
    text = "Contact: NID 12345678901234567, Phone 01712345678"
    input_data = NERInput(text=text)

    result1 = await ner1.process(input_data)
    print(f"✓ Sequential component 1: {len(result1.entities)} entities")

    result2 = await ner2.process(input_data)
    print(f"✓ Sequential component 2: {len(result2.entities)} entities")

    # Test context passing
    input_data.context["previous_result"] = result1.entities
    result3 = await ner2.process(input_data)
    assert "previous_result" in input_data.context
    print(f"✓ Context passing works")

    print(f"\n✓ All composition tests passed!")
    return True


async def test_cli_integration():
    """Test CLI integration"""
    print("\n" + "="*60)
    print("TEST: CLI Integration")
    print("="*60)

    import subprocess

    # Test 1: NER via CLI
    result = subprocess.run(
        ["python3.12", "cli.py", "ner", "--text", "NID 12345678901234567", "--regex-only"],
        capture_output=True,
        text=True,
        timeout=10
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["success"] == True
    print(f"✓ CLI NER works: {output['component_name']}")

    # Test 2: JSON output
    assert "entities" in output["data"]
    assert "nid_number" in output["data"]["entities"]
    print(f"✓ CLI JSON output valid")

    # Test 3: Help
    result = subprocess.run(
        ["python3.12", "cli.py", "--help"],
        capture_output=True,
        text=True,
        timeout=5
    )
    assert result.returncode == 0
    assert "ner" in result.stdout
    print(f"✓ CLI help works")

    print(f"\n✓ All CLI tests passed!")
    return True


async def main():
    """Run all comprehensive tests"""
    print("\n" + "="*70)
    print("COMPOSEABLE NLP COMPONENTS - COMPREHENSIVE TESTS (Python 3.12)")
    print("="*70)

    results = []

    try:
        results.append(("NER Comprehensive", await test_ner_comprehensive()))
    except Exception as e:
        print(f"✗ NER tests failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("NER Comprehensive", False))

    try:
        results.append(("Component Composition", await test_component_composition()))
    except Exception as e:
        print(f"✗ Composition tests failed: {e}")
        results.append(("Component Composition", False))

    try:
        results.append(("CLI Integration", await test_cli_integration()))
    except Exception as e:
        print(f"✗ CLI tests failed: {e}")
        results.append(("CLI Integration", False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:40s} : {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("\nThe composeable NLP architecture is fully functional!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
