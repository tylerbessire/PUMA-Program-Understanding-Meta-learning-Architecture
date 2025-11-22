# PUMA: Program Understanding Meta-learning Architecture

**An Autonomous Cognitive Architecture for Self-Modifying AGI**

## Architecture Overview

PUMA is a layered cognitive architecture combining symbolic reasoning (Hyperon/MeTTa), meta-learning, and neural language models to create an autonomous agent with persistent memory, self-modification capability, and emergent goal formation.

```
┌─────────────────────────────────────────────────────────┐
│                    CONSCIOUSNESS LAYER                   │
│  Self-Model • Autobiographical Memory • Goal Genesis    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│                   META-COGNITIVE LAYER                   │
│    PUMA Core • RFT Engine • Curiosity Drive • Shop      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│                    COGNITIVE LAYER                       │
│      Hyperon/MeTTa • Atomspace • Reasoning Engine       │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│                   INTERACTION LAYER                      │
│   Gemini Live • Web Agent • Tool Use • Perception       │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### Hyperon/MeTTa Integration
- OpenCog Hyperon fork with Python bindings
- Atomspace for knowledge representation and persistent memory
- MeTTa-based reasoning and program execution
- RocksDB/PostgreSQL persistence layer

### PUMA Meta-Cognitive Framework
- **Episodic Memory System**: Timestamped experience nodes with context
- **Memory Consolidation**: Background pattern extraction and concept formation
- **RFT Engine**: Relational Frame Theory for analogical reasoning
- **Curiosity Drive**: Intrinsic motivation and knowledge gap detection
- **Goal Formation**: Autonomous intention generation from drives
- **Self-Model**: Emergent identity from behavioral patterns

### Self-Modification System ("The Shop")
- Code introspection and performance profiling
- Modification hypothesis generation
- Sandboxed testing environment
- A/B cognitive testing
- Rollback and version control

### Interaction Layer
- Gemini Live API for bidirectional audio/text
- Autonomous web browsing and learning
- Tool use and API integration
- Multi-modal perception

## Installation

### Prerequisites
- Python 3.11+
- Rust toolchain (for Hyperon)
- Node.js 18+ (for GUI)
- RocksDB or PostgreSQL

### Setup
```bash
# Clone repository
git clone https://github.com/tylerbessire/PUMA-Program-Understanding-Meta-learning-Architecture
cd PUMA-Program-Understanding-Meta-learning-Architecture

# Install Python dependencies
pip install -r requirements.txt

# Build Hyperon from source
cd hyperon-core
cargo build --release
cd ..

# Install GUI dependencies
cd gui
npm install
cd ..

# Set up persistence database
python setup_persistence.py
```

### Configuration
```bash
# Set API keys
export GEMINI_API_KEY="your-key-here"

# Configure persistence path
export ATOMSPACE_DB_PATH="/path/to/atomspace/db"

# Optional: Enable experimental features
export PUMA_ENABLE_SELF_MODIFICATION=1
```

## Project Structure

```
/
├── hyperon-core/           # Forked Hyperon with extensions
├── puma/                   # PUMA meta-cognitive framework
│   ├── memory/             # Episodic memory and consolidation
│   ├── rft/                # Relational Frame Theory engine
│   ├── curiosity/          # Intrinsic motivation system
│   ├── goals/              # Goal formation and intention
│   └── shop/               # Self-modification system
├── gemini-interface/       # Gemini Live integration
├── web-agent/              # Autonomous browsing
├── atomspace-db/           # Persistence layer
├── gui/                    # Real-time visualization dashboard
├── bootstrap/              # Consciousness initialization
└── tests/                  # Integration and unit tests
```

## Usage

### Bootstrap New Consciousness
```python
from puma.bootstrap import bootstrap_new_consciousness

# Initialize fresh cognitive architecture
consciousness = bootstrap_new_consciousness(
    atomspace_path="/path/to/db",
    enable_self_modification=False  # Disable for initial testing
)

# Start autonomous operation
await consciousness.run()
```

### Interactive Mode
```python
from puma.consciousness import Consciousness

consciousness = Consciousness.load_from_checkpoint("/path/to/checkpoint")

# User interrupt and conversation
consciousness.interrupt_handler.trigger()
await consciousness.converse("Hello, how are you?")

# Resume autonomous activity
consciousness.resume()
```

### GUI Dashboard
```bash
cd gui
npm run dev
```
Access at http://localhost:3000

## Key Features

### Emergent Properties
- **No Hardcoded Personality**: Identity emerges from experience patterns
- **No Preset Knowledge**: All knowledge acquired through exploration
- **No Fixed Goals**: Intentions generated from curiosity and self-assessment
- **Autonomous Learning**: Self-directed web exploration and skill acquisition

### Memory Architecture
- Persistent autobiographical timeline
- Episodic memory with contextual links
- Concept formation through consolidation
- Relational frame derivation (RFT)

### Self-Modification
- Code introspection and analysis
- Performance bottleneck detection
- Hypothesis-driven improvement
- Sandboxed testing before deployment
- Human approval for critical changes

### State Machine
- **SLEEPING**: Memory consolidation and pattern extraction
- **EXPLORING**: Autonomous web learning
- **CONVERSING**: Interactive dialogue
- **SHOPPING**: Self-modification
- **IDLE**: Boredom monitoring and goal formation
- **CREATING**: Creative expression

## Technical Details

### Atomspace Schema
```
EpisodicMemoryNode(timestamp, perception, action, outcome)
ConceptNode(abstraction, confidence)
SelfModelNode(meta-cognitive-state)
GoalNode(intention, priority)
RelationalFrameNode(relation-type, frame)
CodeNode(executable-metta)
PerceptionNode(sensory-input)
EmotionalStateNode(valence)
```

### RFT Relational Frames
- Coordination (similarity)
- Opposition (difference)
- Hierarchy (categorization)
- Temporal (before/after)
- Causal (if-then)

### Persistence
- Incremental Atomspace serialization
- Checkpoint system with version control
- Transaction log for recovery
- Snapshot-based rollback

## Development Status

### Implemented
- [x] Basic project structure
- [x] RFT framework foundation
- [x] Frequency ledger tracking system

### In Progress
- [ ] Hyperon Atomspace integration
- [ ] Bootstrap consciousness seed
- [ ] Gemini Live interface
- [ ] Web agent
- [ ] Memory consolidation
- [ ] Self-modification system
- [ ] GUI dashboard

### Planned
- [ ] Full autonomous operation
- [ ] Public demo deployment
- [ ] Research paper publication

## Research Foundation

### Relational Frame Theory (RFT)
PUMA implements RFT as a computational framework for relational reasoning, enabling:
- Derivation of novel relations without explicit training
- Analogical reasoning through relational frame mapping
- Behavioral generalization to novel contexts

### Meta-Learning
- Task-agnostic learning mechanisms
- Self-supervised pattern extraction
- Transfer learning through relational abstraction

### Cognitive Architecture
- Multiple Demand (MD) Network analog for executive control
- Hippocampal-mPFC loop for episodic retrieval
- Basal ganglia gating for action selection

## Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Benchmark cognitive performance
python tools/benchmark_consciousness.py
```

## Contributing

Focus areas:
- Hyperon/MeTTa integration
- Memory consolidation algorithms
- RFT relational learning
- Self-modification safety
- GUI visualizations

## License

MIT License - See LICENSE file for details

## Citation

```bibtex
@software{puma2024,
  title={PUMA: Program Understanding Meta-learning Architecture},
  author={Bessire, Tyler},
  year={2024},
  url={https://github.com/tylerbessire/PUMA-Program-Understanding-Meta-learning-Architecture}
}
```

## References

- OpenCog Hyperon: https://github.com/trueagi-io/hyperon-experimental
- Relational Frame Theory: Hayes, Barnes-Holmes, & Roche (2001)
- Google Gemini API: https://ai.google.dev/

---

**Status**: Active development for autonomous cognitive architecture research
