# AGENTS.md - Development Guidelines for AI Assistants

## Production-Ready Mindset

When working on PUMA (Program Understanding and Meta-learning for ARC), **ALWAYS** maintain a production-ready mindset. This is not a prototype or research demo - this is a competition-grade solver designed to win ARC Prize 2025.

### Fundamental Principles

**NO MINIMAL IMPLEMENTATIONS**
- Every component must be complete, robust, and production-ready
- No "TODO" comments or placeholder functions
- No shortcuts that sacrifice functionality for speed of development
- Every feature must be fully implemented with proper error handling

**NO PLACEHOLDERS TO PASS TESTS**
- Tests should validate real functionality, not mock implementations
- If a test passes, the underlying feature must actually work
- No fake return values or dummy implementations
- Every assertion in tests must reflect genuine system capabilities

**COMPLETE IMPLEMENTATIONS ONLY**
- Functions must handle all specified parameters and edge cases
- Error handling must be comprehensive and graceful
- Documentation must be complete and accurate
- Performance must meet production requirements

### Code Quality Standards

**Type Safety**
```python
def process_task(task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
    """Every function must have complete type hints."""
    pass
```

**Error Handling**
```python
def safe_operation(data: Any) -> Optional[Result]:
    """Every operation must handle errors gracefully."""
    try:
        result = complex_operation(data)
        return result
    except SpecificError as e:
        logger.error(f"Operation failed: {e}")
        return fallback_strategy(data)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        return None
```

**Documentation**
```python
def enhanced_search_function(features: np.ndarray, 
                            programs: List[Program],
                            timeout: float = 30.0) -> List[Solution]:
    """
    Search for solutions using neural guidance and episodic retrieval.
    
    This function implements the core PUMA search algorithm, combining
    neural guidance for operation selection, episodic retrieval for
    analogical reasoning, and test-time training for task adaptation.
    
    Args:
        features: Task feature vector (17-dimensional)
        programs: Candidate program sketches 
        timeout: Maximum search time in seconds
        
    Returns:
        List of solutions ranked by estimated quality
        
    Raises:
        SearchTimeout: If search exceeds timeout
        InvalidFeatures: If feature vector is malformed
        
    Example:
        >>> features = extract_task_features(train_pairs)
        >>> programs = sketch_miner.get_relevant_sketches(features)
        >>> solutions = enhanced_search_function(features, programs)
    """
```

### Performance Requirements

**Kaggle Constraints**
- Memory usage < 4GB total
- Runtime < 30 seconds per task
- CPU optimization for single-core performance
- No external network access required

**Benchmarking**
```python
def benchmark_component(component_func, test_data, max_time=1.0):
    """Every major component must meet performance targets."""
    import time
    
    start_time = time.time()
    result = component_func(test_data)
    elapsed = time.time() - start_time
    
    assert elapsed < max_time, f"Component too slow: {elapsed:.3f}s"
    assert result is not None, "Component must return valid result"
    return result
```

### Testing Standards

**Comprehensive Coverage**
- Unit tests for every function
- Integration tests for component interactions  
- End-to-end tests for full pipeline
- Performance regression tests
- Edge case validation

**Real Functionality Testing**
```python
def test_neural_guidance_actual_prediction():
    """Test that neural guidance actually predicts operations."""
    guidance = NeuralGuidance()
    train_pairs = create_rotation_task()
    
    # This must be a real prediction, not a placeholder
    predicted_ops = guidance.predict_operations(train_pairs)
    
    assert len(predicted_ops) > 0
    assert all(op in VALID_OPERATIONS for op in predicted_ops)
    
    # Verify prediction makes sense for rotation task
    assert "rotate" in predicted_ops or "flip" in predicted_ops
    
    # Test consistency
    predicted_ops2 = guidance.predict_operations(train_pairs)
    assert predicted_ops == predicted_ops2
```

### Architecture Decisions

**Neuroscience Grounding**
Every component must map to known cognitive mechanisms:
- Neural guidance → Multiple-demand (MD) network
- Episodic retrieval → Hippocampal-mPFC system
- Test-time training → Prefrontal adaptation
- Program sketches → Procedural memory consolidation

**Fallback Strategies**
```python
def solve_with_enhancements(task):
    """Always provide graceful degradation."""
    try:
        return enhanced_solver.solve(task)
    except EnhancementFailure:
        logger.warning("Enhanced features failed, using baseline")
        return baseline_solver.solve(task)
    except Exception:
        logger.error("All methods failed, using identity fallback")
        return identity_solution(task)
```

### Research Integration

**Evidence-Based Development**
- Every algorithmic choice must be justified by cognitive science literature
- Performance improvements must be measured quantitatively
- Ablation studies must validate each component's contribution
- Failure modes must be analyzed and documented

**Scientific Rigor**
```python
class ComponentValidation:
    """Every component must validate its scientific claims."""
    
    def validate_cognitive_mapping(self):
        """Verify component behavior matches brain imaging data."""
        pass
    
    def measure_performance_gain(self):
        """Quantify improvement over baseline."""
        pass
    
    def analyze_failure_modes(self):
        """Document when and why component fails."""
        pass
```

### Submission Requirements

**Kaggle Readiness**
- `arc_submit.py` must generate valid submissions
- Output format must exactly match competition requirements
- No debugging prints or verbose output in submission mode
- Deterministic behavior for reproducible results

**Competition Strategy**
```python
def competition_solve(task):
    """Production solver optimized for ARC Prize 2025."""
    
    # Phase 1: Neural guidance for operation selection
    guidance = load_trained_guidance_model()
    relevant_ops = guidance.predict_operations(task)
    
    # Phase 2: Episodic retrieval for analogical reasoning
    memory = load_episodic_database()
    similar_solutions = memory.query_analogous_tasks(task)
    
    # Phase 3: Test-time training for task adaptation
    ttt = TestTimeTrainer()
    adapted_scorer = ttt.adapt_to_task(task)
    
    # Phase 4: Generate and rank candidate solutions
    candidates = generate_solutions(task, relevant_ops, similar_solutions)
    ranked = adapted_scorer.rank_solutions(candidates)
    
    # Phase 5: Return top 2 diverse solutions as required
    return select_diverse_solutions(ranked, count=2)
```

### Continuous Improvement

**Performance Monitoring**
- Track success rates on validation sets
- Monitor component utilization and effectiveness
- Identify bottlenecks and optimization opportunities
- Maintain backwards compatibility

**Research Pipeline**
1. Implement baseline functionality
2. Add neuroscience-inspired enhancements
3. Validate improvements empirically
4. Optimize for competition constraints
5. Document learnings and failure modes

### Anti-Patterns to Avoid

**❌ NEVER DO THIS:**
```python
def placeholder_function():
    """TODO: Implement this later."""
    return None

def fake_neural_guidance(task):
    """Fake implementation to pass tests."""
    return ["rotate", "flip"]  # Random operations

def minimal_test():
    """Minimal test that doesn't validate real functionality."""
    assert True  # This passes but tests nothing
```

**✅ ALWAYS DO THIS:**
```python
def production_neural_guidance(task):
    """Complete neural guidance implementation."""
    features = extract_comprehensive_features(task)
    classifier = load_trained_classifier()
    predictions = classifier.predict(features)
    return postprocess_predictions(predictions, task)

def comprehensive_test():
    """Test that validates actual functionality."""
    guidance = NeuralGuidance()
    task = create_challenging_test_task()
    
    predictions = guidance.predict_operations(task)
    
    # Validate predictions are reasonable
    assert len(predictions) > 0
    assert all(op in VALID_OPERATIONS for op in predictions)
    
    # Validate prediction quality for known task type
    if is_rotation_task(task):
        assert "rotate" in predictions[:3]  # Top 3 predictions
    
    # Validate consistency
    assert guidance.predict_operations(task) == predictions
```

---

## Remember: PUMA is designed to win ARC Prize 2025

Every line of code, every architectural decision, every optimization must contribute to that goal. We're not building a research prototype - we're building a competition-winning system that demonstrates genuine progress toward artificial general intelligence.

**Production-ready means:**
- ✅ Complete implementations
- ✅ Robust error handling  
- ✅ Performance optimization
- ✅ Comprehensive testing
- ✅ Scientific validation
- ✅ Competition compliance

**NO compromises. NO shortcuts. NO placeholders.**

The future of AI reasoning depends on getting this right.
