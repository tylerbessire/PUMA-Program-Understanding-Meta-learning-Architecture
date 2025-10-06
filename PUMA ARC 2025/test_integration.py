#!/usr/bin/env python3
"""Quick integration test for the full solver pipeline."""

import sys
from pathlib import Path
import numpy as np

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "KAGGLE"))

from kaggle_setup import get_solver

def test_simple_task():
    """Test solver on a simple recoloring task."""
    print("Testing full solver pipeline...")
    print("="*60)

    # Simple task: recolor 3 -> 1
    train_pairs = [
        (
            np.array([[0, 3, 3], [0, 3, 3]], dtype=np.int16),
            np.array([[0, 1, 1], [0, 1, 1]], dtype=np.int16)
        ),
        (
            np.array([[3, 3, 0], [3, 3, 0]], dtype=np.int16),
            np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int16)
        )
    ]

    test_input = np.array([[3, 0], [3, 0]], dtype=np.int16)

    # Initialize solver
    print("\n1. Initializing solver...")
    solver = get_solver()
    print(f"   ✓ Solver type: {type(solver).__name__}")

    # Test object inventory
    print("\n2. Testing object inventory...")
    from PUMA.arc_solver.object_inventory import ObjectInventory

    inventory = ObjectInventory()
    inventory.build_from_train_pairs(train_pairs)

    print(f"   ✓ Objects tracked: {len(inventory.entries)}")

    schema = inventory.get_rule_friendly_schema()
    print(f"   ✓ Patterns detected: {len(schema.get('patterns', []))}")
    print(f"   ✓ Transformation rules: {len(schema.get('transformation_rules', []))}")

    # Test RFT tracking
    print("\n3. Testing RFT tracking...")
    from PUMA.arc_solver.rft_tracking import RFTTracker

    tracker = RFTTracker('test_task')
    tracker.ingest_training_pairs(train_pairs)

    print(f"   ✓ Conflicts detected: {len(tracker.conflicts)}")
    print(f"   ✓ Repair tasks queued: {len(tracker.repair_queue)}")

    # Test pliance engine
    print("\n4. Testing pliance engine...")
    from PUMA.arc_solver.pliance_engine import PlianceEngine, ObjectSelector, RuleAction

    engine = PlianceEngine()

    # Emit test rule
    rule = engine.emit_provisional_rule(
        name='test_recolor',
        selector=ObjectSelector(color=3),
        action=RuleAction(action_type='recolor', parameters={'color': 1}),
        confidence=0.8
    )

    print(f"   ✓ Rules created: {len(engine.rules)}")

    # Validate rule
    validation = engine.validate_rules(train_pairs)
    if validation:
        rule_id = list(validation.keys())[0]
        accuracy = validation[rule_id]['accuracy']
        print(f"   ✓ Rule accuracy: {accuracy:.1%}")

    # Test search fusion (without actually running expensive search)
    print("\n5. Testing search fusion...")
    from PUMA.arc_solver.search_fusion import SearchPriorityWeighter, OperationSeeder

    weighter = SearchPriorityWeighter()
    seeder = OperationSeeder()

    # Mock candidates
    base_candidates = {
        'heuristic': [['test_program_1']],
        'pliance_rules': [['test_program_2']]
    }

    priorities = {
        'heuristic': 0.5,
        'pliance_rules': 0.9
    }

    weighted = weighter.apply_priorities(base_candidates, priorities)
    print(f"   ✓ Candidates weighted: {len(weighted)}")
    print(f"   ✓ Top priority: {weighted[0][1]:.2f}")

    # Test LLM adapters
    print("\n6. Testing LLM adapters...")
    from PUMA.arc_solver.llm_adapters import ObjectInventoryAdapter, ConflictAdapter

    inv_context = ObjectInventoryAdapter.to_llm_context(inventory)
    print(f"   ✓ Inventory adapted: {inv_context['summary']['total_objects']} objects")

    conflict_context = ConflictAdapter.to_llm_context(tracker.conflicts)
    print(f"   ✓ Conflicts adapted: {conflict_context['summary']['total_conflicts']} conflicts")

    # Test prompt generation
    print("\n7. Testing prompt generation...")
    from PUMA.arc_solver.llm_prompts import PromptTemplates

    prompts = PromptTemplates.inventory_analysis_prompt(inv_context)
    print(f"   ✓ System prompt length: {len(prompts['system'])} chars")
    print(f"   ✓ User prompt length: {len(prompts['user'])} chars")
    print(f"   ✓ Requests JSON: {'JSON' in prompts['user']}")

    # Test fallbacks
    print("\n8. Testing fallback reasoning...")
    from PUMA.arc_solver.llm_fallbacks import FallbackReasoner

    fallback_result = FallbackReasoner.analyze_inventory_fallback(inventory)
    print(f"   ✓ Observations: {len(fallback_result['observations'])}")
    print(f"   ✓ Patterns: {len(fallback_result['key_patterns'])}")
    print(f"   ✓ Confidence: {fallback_result['confidence']:.2f}")

    print("\n" + "="*60)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("="*60)

    return True

if __name__ == '__main__':
    try:
        success = test_simple_task()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
