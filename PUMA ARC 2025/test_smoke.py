#!/usr/bin/env python3
"""Quick smoke test for solver with new llm_config and runtime_flags."""
import sys
from pathlib import Path

# Add KAGGLE to path to access kaggle_setup
sys.path.insert(0, str(Path(__file__).parent / "KAGGLE"))

from kaggle_setup import get_solver

def test_basic_init():
    """Test solver initializes with default config."""
    print("Test 1: Default initialization...")
    solver = get_solver()
    print(f"  ✓ Solver created: {type(solver).__name__}")
    print(f"  ✓ LLM config: {solver._llm_config}")
    print(f"  ✓ Runtime flags: {solver._runtime_flags}")
    print()

def test_llm_disabled():
    """Test solver with LLM explicitly disabled."""
    print("Test 2: LLM disabled...")
    solver = get_solver(llm_options={"enabled": False})
    enabled = solver._llm_config.get('enabled', True) if solver._llm_config else False
    print(f"  ✓ LLM enabled: {enabled}")
    print()

def test_rft_first_mode():
    """Test solver with RFT-first runtime flag."""
    print("Test 3: RFT-first mode...")
    solver = get_solver(runtime_flags={"rft_first": True})
    print(f"  ✓ RFT-first: {solver._runtime_flags.get('rft_first', False)}")
    print()

def test_custom_llm_path():
    """Test solver with custom model path."""
    print("Test 4: Custom LLM path...")
    custom_path = "/tmp/fake_model.gguf"
    solver = get_solver(llm_options={"model_path": custom_path})
    path = solver._llm_config.get('model_path', 'none') if solver._llm_config else 'none'
    print(f"  ✓ Model path: {path}")
    print()

if __name__ == "__main__":
    print("=== Solver Smoke Tests ===\n")
    try:
        test_basic_init()
        test_llm_disabled()
        test_rft_first_mode()
        test_custom_llm_path()
        print("✓ All smoke tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
