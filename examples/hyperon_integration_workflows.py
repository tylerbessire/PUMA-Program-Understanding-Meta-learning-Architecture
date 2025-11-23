"""
Hyperon-PUMA Integration Example Workflows

This module demonstrates practical workflows for using Hyperon subagents
within PUMA's cognitive architecture. It shows three main use cases:

1. ARC Task Solving - Distributed reasoning for visual pattern problems
2. RFT Reasoning - Relational frame theory reasoning with MeTTa
3. Frequency Analysis - Pattern frequency analysis with MeTTa inference

Each workflow demonstrates how Hyperon's symbolic reasoning capabilities
enhance PUMA's cognitive processing through parallel distributed execution.

Usage:
------
    python examples/hyperon_integration_workflows.py

    # Or run individual workflows:
    python examples/hyperon_integration_workflows.py --workflow arc
    python examples/hyperon_integration_workflows.py --workflow rft
    python examples/hyperon_integration_workflows.py --workflow frequency
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
puma_root = Path(__file__).parent.parent
sys.path.insert(0, str(puma_root))

# Bootstrap PUMA
try:
    from bootstrap.bootstrap import bootstrap_new_consciousness
    from puma.hyperon_integration import HyperonPUMAIntegration, HyperonConfig
    from puma.rft.reasoning import RelationType
except ImportError as e:
    print(f"Error importing PUMA modules: {e}")
    print("Make sure you're running from the PUMA root directory")
    sys.exit(1)


# ============================================================================
# Workflow 1: ARC Task Solving with Hyperon Subagents
# ============================================================================


async def workflow_arc_task_solving():
    """
    Demonstrate solving an ARC task using distributed Hyperon subagents.

    This workflow:
    1. Initializes PUMA with Hyperon integration
    2. Loads a sample ARC task
    3. Distributes reasoning across subagent pool
    4. Synthesizes solution from parallel reasoning
    5. Shows reasoning trace

    ARC (Abstraction and Reasoning Corpus) tasks require:
    - Pattern recognition
    - Abstraction
    - Analogical reasoning
    - Rule induction

    Hyperon subagents excel at this through:
    - Parallel pattern matching
    - Symbolic rule representation
    - Distributed hypothesis testing
    """
    print("=" * 70)
    print("Workflow 1: ARC Task Solving with Hyperon Subagents")
    print("=" * 70)
    print()

    # Step 1: Bootstrap PUMA with Hyperon integration
    print("[1/5] Bootstrapping PUMA consciousness with Hyperon integration...")
    consciousness = bootstrap_new_consciousness(
        atomspace_path=Path("./atomspace-db/hyperon_workflow"),
        enable_self_modification=False,
        enable_hyperon=True,
        hyperon_config=HyperonConfig(
            max_agents=5,
            create_specialized_pool=True,
            default_coordination_strategy=None,  # Will use default
        ),
    )

    # Get Hyperon integration
    integration = consciousness.hyperon_integration
    if not integration:
        print("ERROR: Hyperon integration not available")
        return

    # Step 2: Initialize Hyperon components
    print("\n[2/5] Initializing Hyperon components...")
    await integration.initialize()
    status = integration.get_status()
    print(f"   Subagents: {status['num_subagents']}")
    print(f"   RFT Bridge: {'enabled' if status['rft_bridge_enabled'] else 'disabled'}")
    print(f"   Hyperon Available: {status['hyperon_available']}")

    # Step 3: Create sample ARC task
    print("\n[3/5] Creating sample ARC task...")
    arc_task = {
        "train": [
            {
                "input": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            },
            {
                "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            },
        ],
        "test": [{"input": [[1, 0, 0], [0, 0, 1], [0, 1, 0]]}],
    }
    print(f"   Training examples: {len(arc_task['train'])}")
    print(f"   Test examples: {len(arc_task['test'])}")

    # Step 4: Solve task with distributed reasoning
    print("\n[4/5] Solving task with distributed reasoning...")
    print("   Distributing work across subagent pool...")

    result = await integration.solve_arc_task(
        task_data=arc_task, max_reasoning_depth=3, use_frequency_analysis=True
    )

    # Step 5: Display results
    print("\n[5/5] Results:")
    print(f"   Success: {result['success']}")
    print(f"   Execution time: {result['execution_time']:.2f}s")

    if result.get("subagent_results"):
        print(f"   Subagents used: {len(result['subagent_results'])}")
        successful = sum(
            1 for r in result["subagent_results"] if r["success"]
        )
        print(f"   Successful executions: {successful}/{len(result['subagent_results'])}")

    if result.get("reasoning_trace"):
        print("\n   Reasoning trace:")
        for i, step in enumerate(result["reasoning_trace"], 1):
            print(f"      {i}. {step.get('step', 'unknown')}")

    if result.get("solution"):
        print(f"\n   Solution method: {result['solution'].get('method', 'unknown')}")
        print(f"   Confidence: {result['solution'].get('confidence', 0.0):.2%}")

    # Cleanup
    print("\n[Cleanup] Shutting down Hyperon integration...")
    await integration.shutdown()
    print("✓ Workflow complete!")
    print()


# ============================================================================
# Workflow 2: RFT Reasoning Distributed Across Agents
# ============================================================================


async def workflow_rft_reasoning():
    """
    Demonstrate RFT (Relational Frame Theory) reasoning with Hyperon.

    This workflow:
    1. Initializes PUMA with RFT and Hyperon
    2. Creates relational frames
    3. Converts frames to MeTTa expressions
    4. Distributes reasoning across subagents
    5. Performs derived relation inference

    RFT enables sophisticated relational reasoning:
    - Coordination: A is like B
    - Opposition: A is opposite of B
    - Hierarchy: A is bigger than B
    - Temporal: A comes before B
    - Spatial: A is above B
    - Causal: A causes B

    Hyperon enhances RFT through:
    - Symbolic representation of relations
    - Logical inference over relational patterns
    - Distributed relation composition
    - Emergent relational networks
    """
    print("=" * 70)
    print("Workflow 2: RFT Reasoning with Hyperon Subagents")
    print("=" * 70)
    print()

    # Step 1: Bootstrap PUMA
    print("[1/6] Bootstrapping PUMA consciousness...")
    consciousness = bootstrap_new_consciousness(
        atomspace_path=Path("./atomspace-db/hyperon_rft_workflow"),
        enable_hyperon=True,
        hyperon_config=HyperonConfig(max_agents=8),
    )

    integration = consciousness.hyperon_integration
    if not integration:
        print("ERROR: Hyperon integration not available")
        return

    # Step 2: Initialize
    print("\n[2/6] Initializing Hyperon integration...")
    await integration.initialize()

    # Step 3: Create relational frames
    print("\n[3/6] Creating relational frames...")
    relations = [
        {
            "source": "cat",
            "target": "dog",
            "type": RelationType.COORDINATION,
            "description": "coordination (similar)",
        },
        {
            "source": "hot",
            "target": "cold",
            "type": RelationType.OPPOSITION,
            "description": "opposition (opposite)",
        },
        {
            "source": "elephant",
            "target": "mouse",
            "type": RelationType.HIERARCHY,
            "description": "hierarchy (bigger than)",
        },
    ]

    for i, rel in enumerate(relations, 1):
        print(
            f"   {i}. {rel['source']} -> {rel['target']} ({rel['description']})"
        )

    # Step 4: Reason with each relation using Hyperon
    print("\n[4/6] Reasoning with RFT frames using Hyperon subagents...")

    for rel in relations:
        print(f"\n   Processing: {rel['source']} -> {rel['target']}")

        # Perform RFT reasoning
        frames = await integration.reason_with_rft(
            source=rel["source"],
            target=rel["target"],
            relation_type=rel["type"],
            context=["example_workflow"],
            use_subagents=True,
        )

        print(f"      Inferred frames: {len(frames)}")

        if frames:
            for frame in frames[:3]:  # Show first 3
                print(f"         - {frame}")

    # Step 5: Demonstrate relational composition
    print("\n[5/6] Demonstrating relational composition...")
    print("   Composing relations: cat->dog (coordination) + dog->wolf (hierarchy)")

    # This would use the RFT bridge to compose relations
    composition_result = await integration.reason_with_rft(
        source="cat",
        target="wolf",
        relation_type=None,  # Infer relation type
        context=["compositional_reasoning"],
        use_subagents=True,
    )

    print(f"   Composed relations: {len(composition_result)}")

    # Step 6: Show statistics
    print("\n[6/6] Subagent statistics:")
    if integration.subagent_manager:
        pool_status = integration.subagent_manager.get_pool_status()
        print(f"   Total tasks completed: {pool_status['completed_tasks']}")
        print(
            f"   Average success rate: {pool_status['average_success_rate']:.2%}"
        )

        # Show agent capabilities
        cap_dist = pool_status.get("capability_distribution", {})
        print(f"\n   Capability distribution:")
        for capability, count in cap_dist.items():
            print(f"      {capability}: {count} agents")

    # Cleanup
    print("\n[Cleanup] Shutting down...")
    await integration.shutdown()
    print("✓ Workflow complete!")
    print()


# ============================================================================
# Workflow 3: Frequency Analysis with MeTTa Inference
# ============================================================================


async def workflow_frequency_analysis():
    """
    Demonstrate frequency analysis using MeTTa inference.

    This workflow:
    1. Initializes PUMA with frequency ledger
    2. Creates pattern data for analysis
    3. Uses MeTTa for symbolic pattern matching
    4. Builds frequency signatures
    5. Shows pattern distribution analysis

    Frequency analysis in PUMA:
    - Tracks pattern occurrence frequencies
    - Builds statistical signatures
    - Identifies dominant patterns
    - Enables frequency-based prediction

    Hyperon enhances frequency analysis through:
    - Symbolic pattern representation
    - Rule-based pattern extraction
    - Compositional pattern matching
    - Logical frequency aggregation
    """
    print("=" * 70)
    print("Workflow 3: Frequency Analysis with MeTTa Inference")
    print("=" * 70)
    print()

    # Step 1: Bootstrap PUMA
    print("[1/5] Bootstrapping PUMA consciousness...")
    consciousness = bootstrap_new_consciousness(
        atomspace_path=Path("./atomspace-db/hyperon_frequency_workflow"),
        enable_hyperon=True,
        hyperon_config=HyperonConfig(
            max_agents=6, enable_frequency_ledger=True
        ),
    )

    integration = consciousness.hyperon_integration
    if not integration:
        print("ERROR: Hyperon integration not available")
        return

    # Step 2: Initialize
    print("\n[2/5] Initializing with frequency ledger...")
    await integration.initialize()

    if integration.frequency_ledger:
        print("   ✓ Frequency ledger initialized")
    else:
        print("   ! Frequency ledger not available (expected if dependencies missing)")

    # Step 3: Create sample pattern data
    print("\n[3/5] Creating sample pattern data...")
    pattern_data = {
        "patterns": [
            {"type": "color", "value": "red", "count": 5},
            {"type": "color", "value": "blue", "count": 3},
            {"type": "color", "value": "green", "count": 2},
            {"type": "shape", "value": "square", "count": 4},
            {"type": "shape", "value": "circle", "count": 6},
            {"type": "size", "value": "large", "count": 3},
            {"type": "size", "value": "small", "count": 7},
        ]
    }

    print(f"   Total patterns: {len(pattern_data['patterns'])}")

    # Display pattern distribution
    print("\n   Pattern distribution:")
    for pattern in pattern_data["patterns"]:
        print(
            f"      {pattern['type']:8} | {pattern['value']:10} : {pattern['count']} occurrences"
        )

    # Step 4: Perform frequency analysis with MeTTa
    print("\n[4/5] Performing frequency analysis with MeTTa inference...")

    signature = await integration.analyze_frequencies(
        pattern_data=pattern_data, use_metta_inference=True
    )

    if signature:
        print("   ✓ Frequency signature generated")
        print(f"      Signature: {signature}")
    else:
        print("   ! Frequency analysis completed (signature generation requires full dependencies)")

    # Step 5: Demonstrate pattern-based reasoning
    print("\n[5/5] Demonstrating pattern-based reasoning with subagents...")

    # Create MeTTa program for pattern analysis
    metta_program = """
    ; Pattern frequency analysis
    (= (most-frequent $patterns)
       (max-by count $patterns))

    ; Pattern correlation
    (= (correlate $p1 $p2)
       (co-occurrence $p1 $p2))
    """

    print("\n   MeTTa program for pattern analysis:")
    print("   " + "\n   ".join(metta_program.strip().split("\n")))

    # Execute with MeTTa engine
    if integration.metta_engine:
        print("\n   Executing pattern analysis...")
        result = integration.metta_engine.run(metta_program)

        if result.success:
            print("   ✓ Analysis complete")
            print(f"      Execution time: {result.execution_time:.4f}s")
        else:
            print(f"   ! Analysis completed (expected if Hyperon not installed)")

    # Show subagent pool status
    print("\n   Subagent pool status:")
    if integration.subagent_manager:
        pool_status = integration.subagent_manager.get_pool_status()
        print(f"      Active agents: {pool_status['total_agents']}")
        print(f"      Completed tasks: {pool_status['completed_tasks']}")

    # Cleanup
    print("\n[Cleanup] Shutting down...")
    await integration.shutdown()
    print("✓ Workflow complete!")
    print()


# ============================================================================
# Comprehensive Integration Demo
# ============================================================================


async def workflow_comprehensive_demo():
    """
    Comprehensive demonstration showing all integration features.

    This workflow combines:
    1. ARC task solving
    2. RFT reasoning
    3. Frequency analysis
    4. Consciousness state integration
    5. Memory integration
    6. Multi-strategy coordination
    """
    print("=" * 70)
    print("Comprehensive Hyperon-PUMA Integration Demo")
    print("=" * 70)
    print()

    # Bootstrap PUMA with full integration
    print("[Setup] Bootstrapping PUMA with full Hyperon integration...")
    consciousness = bootstrap_new_consciousness(
        atomspace_path=Path("./atomspace-db/hyperon_comprehensive"),
        enable_hyperon=True,
        hyperon_config=HyperonConfig(
            max_agents=10,
            create_specialized_pool=True,
            enable_metrics=True,
            enable_caching=True,
            integrate_with_consciousness=True,
            integrate_with_memory=True,
            enable_frequency_ledger=True,
        ),
    )

    integration = consciousness.hyperon_integration

    # Initialize all components
    print("\n[Init] Initializing all components...")
    await integration.initialize()

    # Show full status
    print("\n[Status] Integration status:")
    status = integration.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    # Run mini versions of each workflow
    print("\n[Demo 1] Mini ARC task...")
    mini_arc = {
        "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 1], [1, 1]]}],
        "test": [{"input": [[1, 0], [0, 1]]}],
    }
    arc_result = await integration.solve_arc_task(mini_arc)
    print(f"   Success: {arc_result['success']}")

    print("\n[Demo 2] Mini RFT reasoning...")
    rft_frames = await integration.reason_with_rft(
        source="A", target="B", relation_type=RelationType.COORDINATION
    )
    print(f"   Frames inferred: {len(rft_frames)}")

    print("\n[Demo 3] Mini frequency analysis...")
    mini_patterns = {
        "patterns": [
            {"type": "color", "value": "red", "count": 3},
            {"type": "color", "value": "blue", "count": 2},
        ]
    }
    freq_sig = await integration.analyze_frequencies(mini_patterns)
    print(f"   Signature generated: {freq_sig is not None}")

    # Show final statistics
    print("\n[Stats] Final statistics:")
    if integration.subagent_manager:
        metrics = integration.subagent_manager.get_agent_metrics()
        print(f"   Total agent executions: {sum(m['execution_count'] for m in metrics)}")
        avg_success = (
            sum(m["success_rate"] for m in metrics if m["execution_count"] > 0)
            / len([m for m in metrics if m["execution_count"] > 0])
            if metrics
            else 0
        )
        print(f"   Average success rate: {avg_success:.2%}")

    # Cleanup
    print("\n[Cleanup] Shutting down...")
    await integration.shutdown()
    consciousness.stop()
    print("✓ Comprehensive demo complete!")
    print()


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Run example workflows"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperon-PUMA Integration Example Workflows"
    )
    parser.add_argument(
        "--workflow",
        choices=["all", "arc", "rft", "frequency", "comprehensive"],
        default="all",
        help="Which workflow to run (default: all)",
    )

    args = parser.parse_args()

    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Hyperon-PUMA Integration Workflows  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        if args.workflow == "all":
            await workflow_arc_task_solving()
            await workflow_rft_reasoning()
            await workflow_frequency_analysis()
        elif args.workflow == "arc":
            await workflow_arc_task_solving()
        elif args.workflow == "rft":
            await workflow_rft_reasoning()
        elif args.workflow == "frequency":
            await workflow_frequency_analysis()
        elif args.workflow == "comprehensive":
            await workflow_comprehensive_demo()

        print()
        print("=" * 70)
        print("All workflows completed successfully!")
        print("=" * 70)
        print()
        print("For more information, see:")
        print("  - /home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_integration.py")
        print("  - /home/user/PUMA-Program-Understanding-Meta-learning-Architecture/bootstrap/bootstrap.py")
        print("  - /home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/")
        print()

    except Exception as e:
        print()
        print(f"ERROR: Workflow failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
