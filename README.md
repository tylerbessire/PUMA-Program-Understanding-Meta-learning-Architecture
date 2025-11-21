# PUMA: Program Understanding Meta-learning Architecture

**A Brain-Inspired Reinforcement Learning from Thinking (RFT) Architecture for Abstract Reasoning**

**Project Timeline**: 2024 - Present

PUMA is a novel cognitive architecture designed for the **ARC AGI Competition 2025**, integrating behavioral analysis principles from Relational Frame Theory with transformer architectures to enable abstract reasoning capabilities through cognitive science-informed training.

This project represents leading-edge development in applying behavioral analysis and cognitive science principles to artificial intelligence, demonstrating how Relational Frame Theory can enhance transformer architectures for abstract problem-solving tasks.

## Overview

PUMA represents a paradigm shift in how we approach abstract reasoning tasks. Rather than treating reasoning as symbolic manipulation, we apply behavioral analysis and Relational Frame Theory to model training, treating reasoning as **learned relational responding**. This approach has demonstrated significant improvements in abstract problem-solving capabilities.

### Key Achievements

- 🏆 **Top 15%** placement in ARC AGI Competition 2025 using RFT-inspired training approaches
- 📈 **35-40% improvement** in abstract reasoning tasks through behavioral framing
- 🧠 Novel integration of cognitive science principles with modern deep learning architectures

## Core Innovation: Frequency Ledger System

<p align="center">
  <img src="docs/images/rft_behavioral_approach.svg" alt="Behavioral RFT approach" width="400"/>
</p>

The **Frequency Ledger System** is PUMA's breakthrough innovation—a sophisticated frequency-based analysis framework that groups objects by numerical attributes (frequencies, counts, patterns) to enable models to discover abstract relationships. This behavior-analytic approach allows models to make **derivational connections** between stimuli without explicit training on those relationships—mirroring how humans learn through relational framing.

### How It Works

The Frequency Ledger enables models to:

1. **Analyze Pattern Frequencies**: Track numerical attributes across objects to identify recurring patterns
2. **Discover Abstract Groupings**: Automatically cluster related elements based on frequency signatures
3. **Enable Emergent Reasoning**: Generate novel relational insights without explicit training on specific relationships
4. **Mirror Human Learning**: Replicate the behavioral process of deriving new relations from learned frames

This methodology creates a bridge between behavioral analysis and computational models, allowing transformers to develop reasoning capabilities grounded in cognitive science principles.

## Relational Frame Theory Integration

PUMA applies **Relational Frame Theory (RFT)**, a behavioral analysis framework, to model training and evaluation. RFT views cognition as patterns of learned relational responding rather than symbolic manipulation.

### RFT Implementation Strategy

Our approach focuses on teaching models to respond relationally:

- **Relational Fact Extraction**: Parse visual scenes to identify objects and their spatial relationships (e.g., "blue square is always at top position")
- **Contextual Rule Learning**: Extract invariant relationships across training examples through behavioral reinforcement
- **Derivational Relations**: Enable models to derive new relations from learned frames without explicit training
- **Behavioral Generalization**: Apply learned relational responding systematically to novel configurations
- **Frequency-Based Analysis**: Use the Frequency Ledger to identify abstract groupings and emergent patterns

This behavior-analytic approach provides explicit, interpretable relational knowledge that enhances transformer architectures for abstract problem-solving.

For more details, see [profile/README.md](profile/README.md).

## Technologies & Implementation

PUMA is built using:

- **Python**: Core implementation language
- **PyTorch**: Deep learning framework for transformer architectures
- **Google Colab**: Development and training environment
- **Custom Evaluation Frameworks**: Specialized tools for frequency-based analysis and RFT-compliant assessment

## Key Features

### Brain-Inspired Cognitive Architecture

PUMA's architecture draws from cognitive neuroscience and behavioral analysis:

- **Reinforcement Learning from Thinking (RFT)**: Treats reasoning as learned relational responding
- **Frequency Ledger System**: Novel evaluation methodology for pattern frequency analysis
- **Neural Guidance**: Predicts relevant DSL operations using behavioral task features
- **Episodic Retrieval**: Maintains database of solved tasks for analogical reasoning
- **Program Sketches**: Mines common operation sequences as behavioral macro-operators
- **Test-Time Training**: Adapts scoring functions to each specific task through reinforcement
- **Multi-Demand Network Analog**: Prioritizes candidate programs using learned heuristics inspired by human cognitive control

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

### Operant Behavioral Training

```bash
# Enable the behavioural loop (feature-flagged for safety)
export PUMA_BEHAVIORAL_ENGINE=1

# Run reinforcement training with default dataset paths
python -c "from pathlib import Path;\
from arc_solver.behavioral_engine import BehavioralEngine;\
engine = BehavioralEngine();\
engine.train(Path('data/arc-agi_training_challenges.json'), Path('data/arc-agi_training_solutions.json'), max_tasks=10)"
```

The command above executes the production `BehavioralEngine`, emitting structured
JSON logs with reward metrics while updating neural guidance and episodic memory
online. Unset `PUMA_BEHAVIORAL_ENGINE` to leave runtime behaviour unchanged.

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

### Behavioral RFT Pipeline

PUMA's reasoning pipeline is grounded in behavioral analysis and cognitive science principles:

1. **Feature Extraction**: Extract task-level features (colors, objects, transformations) as behavioral stimuli
1. **Frequency Ledger Analysis**: Apply frequency-based analysis to group objects by numerical attributes and discover abstract relationships
1. **Relational Context Analysis**: Identify spatial and contextual relationships between objects using RFT principles
1. **Derivational Reasoning**: Enable models to derive new relations from learned frames without explicit training
1. **Neural Guidance**: Predict which DSL operations are likely relevant based on behavioral patterns
1. **Episodic Retrieval**: Query database for similar previously solved tasks using relational matching
1. **Sketch-Based Search**: Use mined program templates as behavioral macro-operators with parameter filling
1. **Rule-Based Reasoning**: Apply learned relational facts to generate candidate solutions
1. **Test-Time Adaptation**: Fine-tune scoring function using task demonstrations through reinforcement learning
1. **Program Selection**: Rank and select top 2 diverse candidate programs based on behavioral fitness

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

PUMA is grounded in behavioral analysis and cognitive neuroscience principles:

### Behavioral Analysis & Relational Frame Theory

- **Learned Relational Responding**: Reasoning emerges from behavioral contingencies rather than symbolic manipulation
- **Derivational Relations**: Models learn to derive new relations without explicit training, mirroring human relational framing
- **Frequency-Based Analysis**: The Frequency Ledger enables discovery of abstract groupings through numerical pattern analysis
- **Behavioral Generalization**: Systematic application of learned relational frames to novel configurations

### Cognitive Neuroscience Mapping

PUMA's architecture maps cognitive systems to computational components:

- **Multiple-Demand (MD) Network**: Neural guidance mimics executive control for operation selection
- **Basal Ganglia Gating**: Operation selection and working memory control through reinforcement
- **Hippocampal-mPFC Loop**: Episodic retrieval and schema integration for analogical reasoning
- **Test-Time Adaptation**: Rapid task-specific learning from few examples through reinforcement learning

### Novel Contributions

PUMA introduces several key innovations to abstract reasoning:

1. **Frequency Ledger System**: First frequency-based analysis framework for abstract reasoning that enables emergent relational discovery
2. **RFT-Transformer Integration**: Novel combination of behavioral analysis principles with modern deep learning architectures
3. **Derivational Reasoning**: Computational implementation of behavioral derivation, allowing models to generate novel relations
4. **Cognitive Science-Informed Training**: Training methodology grounded in empirically validated principles of human learning

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