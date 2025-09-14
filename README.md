# PUMA: Program Understanding & Meta-learning Architecture

This repository contains an advanced solver for the **ARC Prize 2025** competition (ARC‑AGI‑2), implementing the complete blueprint from neuroscience-inspired research. It combines symbolic reasoning with neural guidance, episodic retrieval, program sketches, and test-time training to achieve superior performance on abstract reasoning tasks.

## Behavioral Approach with Relational Frame Theory

<p align="center">
  <img src="docs/images/rft_behavioral_approach.svg" alt="Behavioral RFT approach" width="400"/>
</p>

We are implementing a behavioral perspective grounded in **Relational Frame Theory (RFT)** to tackle ARC through explicit relational reasoning. RFT models cognition as networks of learned relational frames, providing a principled foundation for understanding spatial and contextual relationships between objects.

### RFT Implementation Strategy

Our RFT approach focuses on learning explicit relational contexts between objects:

- **Relational Fact Extraction**: Parse visual scenes to identify objects and their spatial relationships (e.g., “blue square is always at top position”)
- **Contextual Rule Learning**: Extract invariant relationships across training examples (e.g., “if blue square at top, then red square at position (blue_y + 1, blue_x)”)
- **Compositional Reasoning**: Combine learned relational frames to generate predictions for novel configurations
- **Behavioral Generalization**: Apply relational rules systematically rather than relying on pattern matching

This approach complements the neural components by providing explicit, interpretable relational knowledge that can be composed and reasoned about symbolically.

For more details, see <profile/README.md>.

## Key Features

### Neuroscience-Inspired Architecture

- **Neural guidance**: Predicts relevant DSL operations using task features
- **Episodic retrieval**: Maintains database of solved tasks for analogical reasoning
- **Program sketches**: Mines common operation sequences as macro-operators
- **Test-time training**: Adapts scoring functions to each specific task
- **Multi-demand network analog**: Prioritizes candidate programs using learned heuristics

### Enhanced Capabilities

- **Object-centric parsing** with connected component analysis
- **Compact DSL** with composable primitives (rotate, flip, translate, recolor, etc.)
- **Relational reasoning** through explicit fact extraction and rule learning
- **Two-attempt diversity** as required by ARC Prize 2025 rules
- **Fallback resilience** with graceful degradation to baseline methods
- **Performance monitoring** with detailed statistics and benchmarking
- **Beam search with constraint propagation** for deeper program synthesis

## Directory Structure

```
arc_solver_project/
│
├── arc_solver/                # Core solver package
│   ├── grid.py                # Grid operations and utilities
│   ├── objects.py             # Connected component extraction
│   ├── dsl.py                 # Domain-specific language primitives
│   ├── heuristics.py          # Heuristic rule inference
│   ├── search.py              # Basic brute-force search
│   ├── solver.py              # Main solver interface with enhancements
│   ├── enhanced_search.py     # Neural-guided program synthesis
│   ├── features.py            # Task feature extraction
│   ├── ttt.py                 # Test-time training utilities
│   ├── io_utils.py            # JSON loading and submission helpers
│   └── neural/                # Neural components
│       ├── guidance.py        # Neural operation prediction
│       ├── episodic.py        # Episodic retrieval system
│       └── sketches.py        # Program sketch mining
│
├── arc_submit.py              # Command-line submission script
├── tools/                     # Training and benchmarking utilities
│   ├── train_guidance.py
│   ├── mine_sketches.py
│   ├── build_memory.py
│   └── benchmark.py
├── tests/                     # Unit and integration tests
└── README.md                  # This file
```

## Quick Start

### Basic Usage (Kaggle-ready)

```bash
# Generate submission file (uses enhanced solver by default)
python arc_submit.py

# Use baseline solver only (if needed)
ARC_USE_BASELINE=1 python arc_submit.py
```

### Training Neural Components

```bash
# Train neural guidance (requires training data)
python tools/train_guidance.py

# Or setup environment with defaults
python tools/benchmark.py
```

### Python API

```python
from arc_solver.solver import solve_task_enhanced, ARCSolver

# Solve a single task with full enhancements
result = solve_task_enhanced(task)

# Configure solver behavior
solver = ARCSolver(use_enhancements=True)
result = solver.solve_task(task)
```

### Public Evaluation Runner

```bash
scripts/eval_public.sh
```

Or via Makefile:

```bash
make eval_public
```

## How It Works

### Enhanced Pipeline

1. **Feature Extraction**: Extract task-level features (colors, objects, transformations)
1. **Relational Context Analysis**: Identify spatial and contextual relationships between objects using RFT principles
1. **Neural Guidance**: Predict which DSL operations are likely relevant
1. **Episodic Retrieval**: Query database for similar previously solved tasks
1. **Sketch-Based Search**: Use mined program templates with parameter filling
1. **Rule-Based Reasoning**: Apply learned relational facts to generate candidate solutions
1. **Test-Time Adaptation**: Fine-tune scoring function using task demonstrations
1. **Program Selection**: Rank and select top 2 diverse candidate programs

### Fallback Strategy

If enhanced components fail, the solver gracefully falls back to:

- Heuristic single-step transformations
- Brute-force enumeration of 2-step programs
- Identity transformation as last resort

## Configuration

The solver supports extensive configuration through environment variables and config files:

### Environment Variables

- `ARC_USE_BASELINE=1`: Force baseline solver only
- `ARC_DISABLE_ENHANCEMENTS=1`: Disable enhanced features

### Configuration File

```json
{
  "use_neural_guidance": true,
  "use_episodic_retrieval": true,
  "use_program_sketches": true,
  "use_test_time_training": true,
  "max_programs": 256,
  "timeout_per_task": 30.0
}
```

## Neural Components

### Neural Guidance

- **Purpose**: Predict which DSL operations are relevant for a given task
- **Architecture**: Simple MLP with task-level features
- **Training**: Uses extracted features from training demonstrations
- **Output**: Operation relevance scores to guide search

### Episodic Retrieval

- **Purpose**: Reuse solutions from similar previously solved tasks
- **Method**: Task signature matching with feature-based similarity
- **Storage**: JSON-based database of solved programs with metadata
- **Retrieval**: Cosine similarity on numerical features + boolean feature matching

### Program Sketches

- **Purpose**: Capture common operation sequences as reusable templates
- **Mining**: Extract frequent 1-step and 2-step operation patterns
- **Usage**: Instantiate sketches with different parameter combinations
- **Adaptation**: Learn from successful programs during solving

### Test-Time Training

- **Purpose**: Adapt scoring function to each specific task
- **Method**: Fine-tune lightweight scorer on task demonstrations
- **Features**: Program length, operation types, success rate, complexity
- **Augmentation**: Generate synthetic training examples via transformations

## Performance and Evaluation

### Benchmarking

```python
from benchmark import Benchmark, SolverConfig

config = SolverConfig()
benchmark = Benchmark(config)
results = benchmark.run_benchmark("test_data.json")
print(f"Success rate: {results['performance_stats']['success_rate']:.3f}")
```

### Monitoring

The solver tracks detailed statistics:

- Success rates for enhanced vs baseline methods
- Component usage (episodic hits, neural guidance, TTT adaptation)
- Timing breakdown per component
- Failure mode analysis

## Implementation Notes

### Kaggle Compatibility

- **Offline execution**: No internet access required
- **Dependency-light**: Uses only NumPy for core operations
- **Compute budget**: Optimized for ~$0.42 per task limit
- **Output format**: Exactly 2 attempts per test input as required

### Code Quality

- **Type hints**: Full typing support for better maintainability
- **Documentation**: Comprehensive docstrings and comments
- **Error handling**: Robust fallback mechanisms
- **Testing**: Validation and benchmarking utilities

## Extending the Solver

### Adding New DSL Operations

1. Define operation function in `dsl.py`
1. Add parameter generation in `sketches.py`
1. Update feature extraction in `features.py`
1. Retrain neural guidance if needed

### Improving Neural Components

1. **Better features**: Add domain-specific feature extractors
1. **Advanced models**: Replace MLP with transformer/GNN
1. **Meta-learning**: Implement few-shot adaptation algorithms
1. **Hybrid methods**: Combine symbolic and neural reasoning

### Advanced Techniques

- **Probabilistic programming**: Sample programs from learned distributions
- **Curriculum learning**: Train on tasks of increasing difficulty
- **Multi-agent reasoning**: Ensemble of specialized solvers
- **Causal reasoning**: Incorporate causal structure learning

## Research Foundation

This implementation is based on the research blueprint “ARC Prize 2025 & Human Fluid Intelligence” which draws from cognitive neuroscience findings about:

- **Multiple-demand (MD) network**: Neural guidance mimics executive control
- **Basal ganglia gating**: Operation selection and working memory control
- **Hippocampal-mPFC loop**: Episodic retrieval and schema integration
- **Test-time adaptation**: Rapid task-specific learning from few examples

The solver architecture directly maps these biological systems to computational components.

## Competition Strategy

### Short-term (Immediate)

- Strong symbolic baseline with neural enhancements
- Episodic retrieval for common patterns
- Test-time adaptation for task specialization
- Kaggle-ready submission format

### Medium-term (During Contest)

- Train neural guidance on public training data
- Mine program sketches from successful solutions
- Analyze semi-private feedback for failure modes
- Expand DSL based on discovered patterns

### Long-term (Advanced Research)

- Probabilistic program synthesis
- Hybrid symbolic-neural architecture
- Broader cognitive priors and meta-learning
- Integration with large language models

## License

This code is designed to be open-sourced under an appropriate license as required by ARC Prize 2025 rules.

## Citation

If you use this solver or build upon its ideas, please cite the research blueprint and this implementation.

## Contributing

Contributions are welcome. Focus areas include:

- Neural architecture improvements
- New DSL operations based on failure analysis
- Advanced meta-learning techniques
- Performance optimizations for Kaggle constraints

-----

**Ready to compete in ARC Prize 2025**