"""
Example: Using LLM Meta-Reasoning in PUMA ARC Solver

This script demonstrates how to use the LLM meta-reasoner to coordinate
search strategies in the PUMA ARC solver.
"""

import numpy as np
from arc_solver.enhanced_search import EnhancedSearch

def example_simple_transformation():
    """Example 1: Simple transformation task."""
    print("=" * 60)
    print("EXAMPLE 1: Simple Transformation Task")
    print("=" * 60)
    
    # Create training pairs (simple +1 transformation)
    train_pairs = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]])),
        (np.array([[0, 1], [2, 3]]), np.array([[1, 2], [3, 4]])),
    ]
    
    # Create search with LLM enabled
    search = EnhancedSearch(
        llm_config={
            'use_llm_reasoning': True,
            'llm_model_name': 'microsoft/Phi-3-mini-4k-instruct',
            'llm_temperature': 0.3,
            'llm_max_tokens': 512
        }
    )
    
    print("\nSynthesizing programs with LLM guidance...")
    programs = search.synthesize_enhanced(train_pairs, max_programs=10)
    
    print(f"\nGenerated {len(programs)} programs")
    
    # Check meta-reasoning results
    stats = search.get_search_statistics()
    meta = stats.get('meta_reasoning', {})
    
    if meta.get('enabled'):
        print(f"\nLLM Meta-Reasoning:")
        print(f"  Strategy: {meta.get('strategy')}")
        print(f"  Suggested operations: {meta.get('suggested_operations', [])[:5]}")
        print(f"  Confidence: {meta.get('confidence', 0):.2f}")
    else:
        print("\nLLM meta-reasoning was not enabled or failed to load")
    
    print(f"\nSearch Statistics:")
    print(f"  Human reasoning candidates: {stats.get('human_reasoning_candidates', 0)}")
    print(f"  Episodic candidates: {stats.get('episodic_candidates', 0)}")
    print(f"  Neural guided candidates: {stats.get('neural_guided_candidates', 0)}")


def example_with_different_models():
    """Example 2: Compare different LLM models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Comparing Different Models")
    print("=" * 60)
    
    train_pairs = [
        (np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]), 
         np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]))
    ]
    
    models = [
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct"  # Uncomment if you have access
    ]
    
    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            search = EnhancedSearch(
                llm_config={
                    'use_llm_reasoning': True,
                    'llm_model_name': model_name,
                    'llm_temperature': 0.3
                }
            )
            
            programs = search.synthesize_enhanced(train_pairs, max_programs=5)
            stats = search.get_search_statistics()
            meta = stats.get('meta_reasoning', {})
            
            print(f"Strategy: {meta.get('strategy', 'N/A')}")
            print(f"Confidence: {meta.get('confidence', 0):.2f}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")


def example_without_llm():
    """Example 3: Running without LLM for comparison."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Without LLM (Baseline)")
    print("=" * 60)
    
    train_pairs = [
        (np.array([[1, 2, 3]]), np.array([[3, 2, 1]]))
    ]
    
    # Disable LLM
    search = EnhancedSearch(
        llm_config={'use_llm_reasoning': False}
    )
    
    print("\nSynthesizing programs WITHOUT LLM guidance...")
    programs = search.synthesize_enhanced(train_pairs, max_programs=10)
    
    print(f"Generated {len(programs)} programs")
    
    stats = search.get_search_statistics()
    print(f"\nMeta-reasoning enabled: {stats.get('meta_reasoning', {}).get('enabled', False)}")


def example_custom_config():
    """Example 4: Custom LLM configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 60)
    
    train_pairs = [
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]))
    ]
    
    # Custom config with higher temperature for more creative suggestions
    search = EnhancedSearch(
        llm_config={
            'use_llm_reasoning': True,
            'llm_model_name': 'microsoft/Phi-3-mini-4k-instruct',
            'llm_temperature': 0.7,  # Higher = more creative
            'llm_max_tokens': 256,   # Shorter responses
        }
    )
    
    print("\nUsing custom config (higher temperature)...")
    programs = search.synthesize_enhanced(train_pairs, max_programs=10)
    
    stats = search.get_search_statistics()
    meta = stats.get('meta_reasoning', {})
    
    print(f"Strategy: {meta.get('strategy', 'N/A')}")
    print(f"Operations: {meta.get('suggested_operations', [])[:5]}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM Meta-Reasoning Examples")
    print("=" * 60)
    print("\nNote: First run will download model (~2GB)")
    print()
    
    # Run examples
    try:
        example_simple_transformation()
        example_without_llm()
        example_custom_config()
        
        # Only run model comparison if user wants
        response = input("\nRun model comparison? (downloads multiple models, y/n): ")
        if response.lower() == 'y':
            example_with_different_models()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
