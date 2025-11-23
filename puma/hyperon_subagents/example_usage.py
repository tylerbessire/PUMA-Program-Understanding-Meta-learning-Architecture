"""
Example Usage of MeTTa Execution Engine

Demonstrates integration between PUMA's RFT system and MeTTa symbolic reasoning.
Shows common operations: pattern matching, relational reasoning, frequency analysis,
and DSL-to-MeTTa compilation.
"""

from pathlib import Path
from puma.hyperon_subagents import (
    MeTTaExecutionEngine,
    ExecutionMode,
    ExecutionResult,
)
from puma.rft import (
    RelationalFrame,
    RelationType,
    Context,
    Entity,
    Limits,
)


def example_basic_execution():
    """Example 1: Basic MeTTa program execution"""
    print("=" * 70)
    print("EXAMPLE 1: Basic MeTTa Program Execution")
    print("=" * 70)

    # Initialize engine
    engine = MeTTaExecutionEngine(execution_mode=ExecutionMode.BATCH)

    # Execute simple arithmetic
    result = engine.execute_program("(+ 2 3)")
    print(f"Result: {result}")
    print(f"Values: {result.results}")
    print()

    # Execute pattern definition and matching
    program = """
    (color-cell 0 0 blue)
    (color-cell 1 0 red)
    (color-cell 2 0 blue)
    !(match &self (color-cell ?x ?y blue) (color-cell ?x ?y blue))
    """

    result = engine.execute_program(program)
    print(f"Pattern matching result: {result}")
    print()


def example_rft_integration():
    """Example 2: RFT to MeTTa conversion"""
    print("=" * 70)
    print("EXAMPLE 2: RFT to MeTTa Integration")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Create relational frames
    frames = [
        RelationalFrame(
            relation_type=RelationType.COORDINATION,
            source="square",
            target="rectangle",
            strength=0.8,
            context=["shape_similarity"]
        ),
        RelationalFrame(
            relation_type=RelationType.HIERARCHY,
            source="blue",
            target="category:color",
            strength=1.0,
            context=["color_taxonomy"]
        ),
        RelationalFrame(
            relation_type=RelationType.CAUSAL,
            source="action_rotate",
            target="outcome_transformed",
            strength=0.9,
            context=["transformation"]
        ),
    ]

    # Convert frames to MeTTa and execute
    print("Converting RFT frames to MeTTa:\n")
    metta_program_lines = []

    for frame in frames:
        metta_expr = engine.rft_to_metta(frame)
        metta_program_lines.append(metta_expr)
        print(f"  {metta_expr}")

    # Execute the MeTTa program
    metta_program = "\n".join(metta_program_lines)
    result = engine.execute_program(metta_program)
    print(f"\nExecution result: {result}")
    print()

    # Query the frames
    print("Querying for coordination frames:")
    query_result = engine.query_atomspace(
        "(RelFrame coordination ?source ?target ?strength)"
    )
    print(f"  Found {len(query_result)} coordination frames")
    print()


def example_dsl_compilation():
    """Example 3: DSL to MeTTa compilation"""
    print("=" * 70)
    print("EXAMPLE 3: PUMA DSL to MeTTa Compilation")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Example DSL operations
    dsl_operations = [
        {
            "operation": "pattern_match",
            "params": {
                "pattern": "(cell ?x ?y blue)",
                "target": "(cell 0 0 blue)"
            }
        },
        {
            "operation": "transform",
            "params": {
                "input_pattern": "(cell ?x ?y blue)",
                "output_pattern": "(cell ?x ?y red)",
                "target": "$grid"
            }
        },
        {
            "operation": "frequency_analysis",
            "params": {
                "items": ["blue", "blue", "red", "blue", "green"]
            }
        },
        {
            "operation": "relational_query",
            "params": {
                "relation_type": "coordination",
                "source": "square",
                "target": "?similar"
            }
        },
    ]

    print("Compiling DSL operations to MeTTa:\n")
    for i, dsl_op in enumerate(dsl_operations, 1):
        print(f"{i}. DSL Operation: {dsl_op['operation']}")
        try:
            metta_code = engine.compile_dsl_to_metta(dsl_op)
            print(f"   MeTTa Code: {metta_code}")
        except Exception as e:
            print(f"   Error: {e}")
        print()


def example_context_conversion():
    """Example 4: Convert RFT Context to MeTTa"""
    print("=" * 70)
    print("EXAMPLE 4: RFT Context to MeTTa Knowledge Base")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Create a sample RFT context
    context = Context(
        state={
            "grid_size": (3, 3),
            "current_cell": (1, 1),
            "colors_found": ["blue", "red", "green"],
            "pattern_count": 5,
        },
        history=[],
        constraints={
            "max_steps": 100,
            "allowed_colors": ["blue", "red", "green", "yellow"],
            "grid_bounds": (0, 0, 3, 3),
        },
        goal_test=lambda state: state.get("pattern_count", 0) >= 5,
        limits=Limits(
            pliance_steps=50,
            tracking_budget=20,
            thresh=0.7,
            outer_budget=10
        ),
        metrics={
            "steps_taken": 12,
            "patterns_found": 5,
            "transformations_applied": 3,
        }
    )

    # Convert context to MeTTa
    metta_kb = engine.context_to_metta(context)
    print("Generated MeTTa Knowledge Base:\n")
    print(metta_kb)
    print()

    # Execute the knowledge base
    result = engine.execute_program(metta_kb)
    print(f"Execution result: {result}")
    print()


def example_entity_conversion():
    """Example 5: Convert PUMA Entities to MeTTa"""
    print("=" * 70)
    print("EXAMPLE 5: Entity to MeTTa Conversion")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Create entities
    entities = [
        Entity(
            id="obj_1",
            type="grid_object",
            features={
                "color": "blue",
                "size": 3,
                "position": (1, 1),
                "frequency": 5
            }
        ),
        Entity(
            id="obj_2",
            type="grid_object",
            features={
                "color": "red",
                "size": 1,
                "position": (2, 2),
                "frequency": 2
            }
        ),
    ]

    print("Converting PUMA Entities to MeTTa:\n")
    metta_lines = []

    for entity in entities:
        metta_expr = engine.entity_to_metta(entity)
        metta_lines.append(metta_expr)
        print(f"  {metta_expr}")

    # Execute entity definitions
    metta_program = "\n".join(metta_lines)
    result = engine.execute_program(metta_program)
    print(f"\nExecution result: {result}")
    print()


def example_sample_programs():
    """Example 6: Demonstrate sample programs"""
    print("=" * 70)
    print("EXAMPLE 6: Sample MeTTa Programs for PUMA Operations")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Get sample programs
    samples = engine.get_sample_programs()

    print(f"Available {len(samples)} sample programs:\n")
    for name, code in samples.items():
        print(f"--- {name.upper().replace('_', ' ')} ---")
        print(code)
        print()


def example_file_loading():
    """Example 7: Load and execute MeTTa file"""
    print("=" * 70)
    print("EXAMPLE 7: Load MeTTa File")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Path to sample programs file
    sample_file = Path(__file__).parent / "sample_programs.metta"

    if sample_file.exists():
        print(f"Loading MeTTa file: {sample_file}\n")

        try:
            result = engine.load_metta_file(sample_file)
            print(f"File execution result: {result}")
            print(f"Number of results: {len(result.results)}")
            print(f"Execution time: {result.execution_time:.4f}s")
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        print(f"Sample file not found: {sample_file}")

    print()


def example_execution_modes():
    """Example 8: Different execution modes"""
    print("=" * 70)
    print("EXAMPLE 8: Execution Modes (Interactive, Batch, Async)")
    print("=" * 70)

    program = """
    (define-atom concept_a)
    (define-atom concept_b)
    (relate concept_a concept_b)
    """

    # Batch mode
    print("1. BATCH MODE:")
    engine_batch = MeTTaExecutionEngine(execution_mode=ExecutionMode.BATCH)
    result = engine_batch.execute_program(program)
    print(f"   Result: {result}\n")

    # Interactive mode
    print("2. INTERACTIVE MODE:")
    engine_interactive = MeTTaExecutionEngine(execution_mode=ExecutionMode.INTERACTIVE)
    result = engine_interactive.execute_program(program)
    print(f"   Result: {result}\n")

    # Async mode
    print("3. ASYNC MODE:")
    engine_async = MeTTaExecutionEngine(execution_mode=ExecutionMode.ASYNC)
    result = engine_async.execute_program(program)
    print(f"   Result: {result}\n")


def example_statistics():
    """Example 9: Engine statistics"""
    print("=" * 70)
    print("EXAMPLE 9: Execution Statistics")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Execute several programs
    programs = [
        "(+ 1 2)",
        "(* 3 4)",
        "(- 10 5)",
        "(/ 20 4)",
    ]

    for prog in programs:
        engine.execute_program(prog)

    # Get statistics
    stats = engine.get_statistics()

    print("Engine Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def example_atom_registration():
    """Example 10: Register custom atoms"""
    print("=" * 70)
    print("EXAMPLE 10: Custom Atom Registration")
    print("=" * 70)

    engine = MeTTaExecutionEngine()

    # Register different types of atoms
    print("Registering custom atoms:\n")

    # String atom
    engine.register_atom("my_concept", "learning_algorithm")
    print("  Registered: my_concept = 'learning_algorithm'")

    # Numeric atom
    engine.register_atom("learning_rate", 0.001)
    print("  Registered: learning_rate = 0.001")

    # Dict atom (converted to MeTTa structure)
    engine.register_atom("model_config", {
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12
    })
    print("  Registered: model_config = {dict with 3 keys}")

    print(f"\nTotal registered atoms: {len(engine._registered_atoms)}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("PUMA MeTTa Execution Engine - Comprehensive Examples")
    print("=" * 70 + "\n")

    try:
        # Run examples
        example_basic_execution()
        example_rft_integration()
        example_dsl_compilation()
        example_context_conversion()
        example_entity_conversion()
        example_sample_programs()
        example_file_loading()
        example_execution_modes()
        example_atom_registration()
        example_statistics()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
