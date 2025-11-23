"""
Hyperon Atomspace Integration Example

Demonstrates how to use the HyperonAtomspaceAdapter for PUMA cognitive architecture.
Shows type mappings, pattern matching, and dual persistence.
"""

from pathlib import Path
from datetime import datetime, timezone
from core import (
    HyperonAtomspaceAdapter,
    Atomspace,
    Atom,
    Link,
    AtomType,
    bootstrap_atomspace,
    get_atomspace_info,
    demonstrate_hyperon_queries,
    HYPERON_AVAILABLE
)


def example_basic_usage():
    """Example 1: Basic atomspace creation and usage"""
    print("=== Example 1: Basic Atomspace Usage ===\n")

    # Create atomspace with Hyperon integration
    atomspace = bootstrap_atomspace(
        persistence_path=Path('./data/atomspace'),
        use_hyperon=True
    )

    print(f"Atomspace type: {type(atomspace).__name__}")
    if isinstance(atomspace, HyperonAtomspaceAdapter):
        print(f"Hyperon enabled: {atomspace.use_hyperon}")

    # Add some atoms
    concept1 = Atom(
        id="concept_learning",
        type=AtomType.CONCEPT,
        content={'name': 'learning', 'domain': 'cognitive'},
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(concept1)

    episodic = Atom(
        id="episode_001",
        type=AtomType.EPISODIC_MEMORY,
        content={
            'event': 'learned new concept',
            'context': 'study session',
            'outcome': 'success'
        },
        timestamp=datetime.now(timezone.utc),
        truth_value=0.9,
        confidence=0.85
    )
    atomspace.add_atom(episodic)

    # Add link
    link = Link(
        source_id="episode_001",
        target_id="concept_learning",
        link_type="relates_to",
        strength=0.9
    )
    atomspace.add_link(link)

    print(f"\nAtoms created: {atomspace.count_atoms()}")
    print(f"Concepts: {atomspace.count_concepts()}")

    # Save to both JSON and Hyperon formats
    atomspace.save()
    print("\nAtomspace saved to disk (JSON + Hyperon)")


def example_hyperon_type_mapping():
    """Example 2: PUMA type to Hyperon atom mapping"""
    print("\n\n=== Example 2: Type Mapping ===\n")

    atomspace = HyperonAtomspaceAdapter(
        persistence_path=Path('./data/type_demo'),
        use_hyperon=True
    )

    # 1. EpisodicMemoryNode → Hyperon EpisodicMemory
    episodic = Atom(
        id="mem_001",
        type=AtomType.EPISODIC_MEMORY,
        content={'event': 'first interaction', 'timestamp': datetime.now(timezone.utc).isoformat()},
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(episodic)
    print("✓ EpisodicMemoryNode → (EpisodicMemory id content timestamp tv conf)")

    # 2. ConceptNode → Hyperon Concept
    concept = Atom(
        id="concept_002",
        type=AtomType.CONCEPT,
        content={'name': 'consciousness', 'properties': {'abstract': True}},
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(concept)
    print("✓ ConceptNode → (Concept id name properties tv conf)")

    # 3. RelationalFrameNode → Hyperon RelationalFrame
    frame = Atom(
        id="frame_001",
        type=AtomType.RELATIONAL_FRAME,
        content={'type': 'cause-effect', 'relations': [{'cause': 'A', 'effect': 'B'}]},
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(frame)
    print("✓ RelationalFrameNode → (RelationalFrame id type relations tv conf)")

    # 4. CodeNode → Hyperon Code (MeTTa executable)
    code = Atom(
        id="code_001",
        type=AtomType.CODE,
        content="(= (factorial 0) 1)\n(= (factorial $n) (* $n (factorial (- $n 1))))",
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(code)
    print("✓ CodeNode → (Code id metta-expr timestamp)")

    # 5. SelfModelNode → Hyperon SelfModel
    self_model = Atom(
        id="self_001",
        type=AtomType.SELF_MODEL,
        content={
            'identity': 'PUMA Agent',
            'capabilities': ['reason', 'learn', 'communicate'],
            'state': 'active'
        },
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(self_model)
    print("✓ SelfModelNode → (SelfModel id properties timestamp tv conf)")

    # 6. GoalNode → Hyperon Goal
    goal = Atom(
        id="goal_001",
        type=AtomType.GOAL,
        content={'objective': 'learn pattern matching', 'priority': 'high', 'status': 'active'},
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(goal)
    print("✓ GoalNode → (Goal id content timestamp tv conf)")

    atomspace.save()
    print(f"\n{atomspace.count_atoms()} atoms saved with type mappings")


def example_pattern_matching():
    """Example 3: Hyperon pattern matching queries"""
    print("\n\n=== Example 3: Pattern Matching Queries ===\n")

    if not HYPERON_AVAILABLE:
        print("Hyperon not available - skipping pattern matching demo")
        return

    atomspace = HyperonAtomspaceAdapter(
        persistence_path=Path('./data/queries'),
        use_hyperon=True
    )

    # Add test data
    for i in range(5):
        concept = Atom(
            id=f"concept_{i}",
            type=AtomType.CONCEPT,
            content={'name': f'concept_{i}', 'category': 'test'},
            timestamp=datetime.now(timezone.utc),
            truth_value=0.8 + i * 0.04,
            confidence=0.9
        )
        atomspace.add_atom(concept)

    for i in range(3):
        episode = Atom(
            id=f"episode_{i}",
            type=AtomType.EPISODIC_MEMORY,
            content={'event': f'event_{i}', 'importance': i},
            timestamp=datetime.now(timezone.utc)
        )
        atomspace.add_atom(episode)

    # Query using Hyperon pattern matching
    print("Query 1: All concepts")
    results = atomspace.query_hyperon("(Concept $id $name $props $tv $conf)")
    print(f"  Found: {len(results)} matches\n")

    print("Query 2: All episodic memories")
    results = atomspace.query_hyperon("(EpisodicMemory $id $content $ts $tv $conf)")
    print(f"  Found: {len(results)} matches\n")

    # Run comprehensive demo
    demonstrate_hyperon_queries(atomspace)


def example_dual_persistence():
    """Example 4: Dual persistence (JSON + Hyperon)"""
    print("\n\n=== Example 4: Dual Persistence ===\n")

    persistence_path = Path('./data/dual_persist')

    # Create and populate atomspace
    atomspace = HyperonAtomspaceAdapter(
        persistence_path=persistence_path,
        use_hyperon=True
    )

    # Add diverse atom types
    atoms_to_add = [
        Atom(id="c1", type=AtomType.CONCEPT, content={'name': 'test'}, timestamp=datetime.now(timezone.utc)),
        Atom(id="e1", type=AtomType.EPISODIC_MEMORY, content={'event': 'test'}, timestamp=datetime.now(timezone.utc)),
        Atom(id="g1", type=AtomType.GOAL, content={'objective': 'test'}, timestamp=datetime.now(timezone.utc)),
    ]

    for atom in atoms_to_add:
        atomspace.add_atom(atom)

    # Save to both formats
    atomspace.save()

    print("Files created:")
    if persistence_path.exists():
        for file in persistence_path.iterdir():
            if file.is_file():
                print(f"  - {file.name} ({file.stat().st_size} bytes)")

    # Load from persistence
    print("\nLoading from persistence...")
    atomspace2 = HyperonAtomspaceAdapter(
        persistence_path=persistence_path,
        use_hyperon=True
    )

    print(f"Loaded {atomspace2.count_atoms()} atoms")
    print(f"Hyperon enabled: {atomspace2.use_hyperon}")


def example_snapshot_restore():
    """Example 5: Snapshot and restore"""
    print("\n\n=== Example 5: Snapshot and Restore ===\n")

    atomspace = HyperonAtomspaceAdapter(
        persistence_path=Path('./data/snapshots'),
        use_hyperon=True
    )

    # Add initial state
    atom1 = Atom(
        id="state_1",
        type=AtomType.CONCEPT,
        content={'version': 1},
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(atom1)
    print(f"Initial state: {atomspace.count_atoms()} atoms")

    # Create snapshot
    snapshot_id = atomspace.create_snapshot()
    print(f"Snapshot created: {snapshot_id}")

    # Modify state
    atom2 = Atom(
        id="state_2",
        type=AtomType.CONCEPT,
        content={'version': 2},
        timestamp=datetime.now(timezone.utc)
    )
    atomspace.add_atom(atom2)
    print(f"Modified state: {atomspace.count_atoms()} atoms")

    # Restore snapshot
    atomspace.restore_snapshot(snapshot_id)
    print(f"Restored state: {atomspace.count_atoms()} atoms")
    print("✓ Snapshot restore successful")


def example_backward_compatibility():
    """Example 6: Backward compatibility with JSON-based Atomspace"""
    print("\n\n=== Example 6: Backward Compatibility ===\n")

    # Create old-style JSON atomspace
    json_atomspace = Atomspace(Path('./data/json_compat'))
    json_atomspace.add_atom(Atom(
        id="compat_test",
        type=AtomType.CONCEPT,
        content={'test': 'compatibility'},
        timestamp=datetime.now(timezone.utc)
    ))
    json_atomspace.save()
    print(f"JSON Atomspace: {json_atomspace.count_atoms()} atoms")

    # Create new Hyperon-enabled atomspace
    hyperon_atomspace = HyperonAtomspaceAdapter(
        Path('./data/hyperon_compat'),
        use_hyperon=True
    )
    hyperon_atomspace.add_atom(Atom(
        id="hyperon_test",
        type=AtomType.CONCEPT,
        content={'test': 'hyperon'},
        timestamp=datetime.now(timezone.utc)
    ))
    hyperon_atomspace.save()
    print(f"Hyperon Atomspace: {hyperon_atomspace.count_atoms()} atoms")

    # Both use same API
    print("\n✓ Both implementations support identical API:")
    print("  - add_atom()")
    print("  - get_atom()")
    print("  - query_by_type()")
    print("  - save() / load()")
    print("  - create_snapshot() / restore_snapshot()")


def main():
    """Run all examples"""
    print("=" * 60)
    print("PUMA Atomspace - Hyperon Integration Examples")
    print("=" * 60)

    # Show system info
    info = get_atomspace_info()
    print(f"\nHyperon Available: {info['hyperon_available']}")
    print(f"Default Backend: {info['default_backend']}")
    print()

    # Run examples
    try:
        example_basic_usage()
        example_hyperon_type_mapping()
        example_pattern_matching()
        example_dual_persistence()
        example_snapshot_restore()
        example_backward_compatibility()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
