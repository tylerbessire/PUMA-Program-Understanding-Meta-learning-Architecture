# Contributing to PUMA

PUMA is a production-ready ARC solver designed for the ARC Prize 2025 competition. We maintain high standards for code quality, performance, and scientific rigor.

## Development Philosophy

### Production-Ready Mindset
- **No minimal implementations**: Every component must be complete and robust
- **No TODOs**: Code should be finished when submitted
- **No placeholders**: All functionality must be implemented
- **Performance first**: Optimize for Kaggle's compute constraints
- **Reliability**: Graceful degradation and comprehensive error handling

### Code Quality Standards
- **Type hints**: All functions must have complete type annotations
- **Documentation**: Comprehensive docstrings for all public interfaces
- **Testing**: Full test coverage for all components
- **Error handling**: Robust fallback mechanisms
- **Logging**: Appropriate logging for debugging and monitoring

## Contributing Guidelines

### 1. Feature Development
- Start with research justification from cognitive neuroscience literature
- Implement complete solution with benchmarks
- Provide comprehensive tests
- Document architectural decisions

### 2. Code Structure
```python
def function_name(param: Type) -> ReturnType:
    """
    Complete docstring with purpose, parameters, returns, and examples.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        SpecificError: When this error occurs
    """
    # Implementation with error handling
    try:
        result = complex_operation(param)
        return result
    except SpecificError as e:
        logger.error(f"Failed to process {param}: {e}")
        return fallback_result()
```

### 3. Testing Requirements
- Unit tests for all functions
- Integration tests for component interactions
- End-to-end tests for solver pipeline
- Performance benchmarks
- Edge case coverage

### 4. Performance Optimization
- Profile before optimizing
- Measure impact quantitatively
- Consider Kaggle compute limits (~$0.42 per task)
- Optimize for common cases first
- Maintain graceful degradation

## Component-Specific Guidelines

### Neural Components
- **Training data**: Use only public ARC training set
- **Model size**: Keep models lightweight (< 1MB)
- **Inference speed**: < 100ms per prediction
- **Fallbacks**: Always provide heuristic alternatives

### Memory Systems
- **Storage efficiency**: Compact representations
- **Retrieval speed**: Sub-second query times
- **Scalability**: Handle 1000+ stored episodes
- **Persistence**: Robust serialization/deserialization

### Search Algorithms
- **Completeness**: Explore space systematically
- **Efficiency**: Prune search space intelligently
- **Diversity**: Generate varied candidate solutions
- **Termination**: Clear stopping criteria

## Research Integration

### Cognitive Science Grounding
- Every feature should map to known cognitive mechanisms
- Cite relevant neuroscience literature
- Validate against human reasoning patterns
- Consider developmental and lesion studies

### Algorithmic Innovations
- Justify design choices with empirical evidence
- Compare against baseline implementations
- Measure performance on diverse task types
- Document failure modes and limitations

## Testing and Validation

### Required Test Categories
1. **Unit Tests**: Individual function correctness
2. **Integration Tests**: Component interactions
3. **Performance Tests**: Speed and memory usage
4. **Regression Tests**: Prevent degradation
5. **End-to-End Tests**: Full solver pipeline

### Benchmarking Protocol
```python
def test_component_performance():
    """Benchmark component against baseline."""
    baseline_time = measure_baseline()
    enhanced_time = measure_enhanced()
    
    assert enhanced_time < baseline_time * 1.5  # Max 50% slowdown
    assert enhanced_accuracy > baseline_accuracy  # Must improve accuracy
```

### Test Data Management
- Use consistent test sets across experiments
- Separate validation from training data
- Document data preprocessing steps
- Version control test datasets

## Documentation Standards

### Code Documentation
- Module-level docstrings explaining purpose
- Class docstrings with usage examples
- Function docstrings with type information
- Inline comments for complex logic

### Architecture Documentation
- Component interaction diagrams
- Data flow specifications
- Performance characteristics
- Extension points

### Research Documentation
- Theoretical justification for approaches
- Experimental validation results
- Comparison with literature baselines
- Future research directions

## Submission Requirements

### Code Quality Checklist
- [ ] All type hints complete
- [ ] All docstrings present
- [ ] All tests passing
- [ ] Performance benchmarks meet targets
- [ ] No TODOs or placeholders
- [ ] Error handling comprehensive
- [ ] Documentation updated

### Performance Checklist
- [ ] Memory usage < 4GB
- [ ] Runtime < 30s per task
- [ ] Success rate improvement demonstrated
- [ ] Graceful degradation verified
- [ ] Kaggle compatibility tested

### Research Checklist
- [ ] Cognitive justification provided
- [ ] Literature review complete
- [ ] Experimental validation performed
- [ ] Failure analysis documented
- [ ] Future work identified

## Review Process

### Code Review
1. **Functionality**: Does it work correctly?
2. **Performance**: Does it meet speed/memory requirements?
3. **Quality**: Is it well-documented and tested?
4. **Integration**: Does it fit the architecture?
5. **Research**: Is it scientifically sound?

### Research Review
1. **Novelty**: What's new compared to existing work?
2. **Validity**: Are experiments well-designed?
3. **Significance**: Does it improve solver performance?
4. **Clarity**: Is the contribution well-explained?

## Getting Started

1. **Environment Setup**
   ```bash
   git clone https://github.com/tylerbessire/PUMA.git
   cd PUMA
   pip install -r requirements.txt
   python tools/verify_layout.py
   ```

2. **Run Tests**
   ```bash
   python -m pytest tests/ -v
   ```

3. **Train Components**
   ```bash
   ./scripts/train_guidance.sh
   ./scripts/mine_sketches.sh
   ./scripts/build_memory.sh
   ```

4. **Benchmark Performance**
   ```bash
   python tools/benchmark.py --test_data data/arc-agi_evaluation_challenges.json
   ```

Remember: PUMA is designed to win ARC Prize 2025. Every contribution should move us closer to that goal while maintaining scientific rigor and code quality.
