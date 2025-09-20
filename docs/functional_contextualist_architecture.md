[S:DOC v2] doc=functional_contextualist_architecture sections=5 scope=behavioral pass


# A Functional Contextualist Architecture for Abstract Reasoning: Integrating Behavioral Analysis into the PUMA Solver

## 1. Architectural Analysis of PUMA as a Behavioral System

### 1.1 Neuroscience-Inspired Foundations
- **Multiple-Demand (MD) Network Analog** — `arc_solver/neural/guidance.py`
- **Basal Ganglia Gating Analog** — `arc_solver/enhanced_search.py`
- **Hippocampal–mPFC Loop Analog** — `arc_solver/neural/episodic.py`
- **Test-Time Adaptation Analog** — `arc_solver/ttt.py`

### 1.2 Behavioral Deconstruction (A–B–C Model)
| Element | PUMA Realisation | Notes |
| --- | --- | --- |
| Antecedent Stimulus | ARC task grids (`data/arc-agi_training_challenges.json`) | Input demonstrations/tables |
| Behavior (Operant) | Synthesised DSL program (`arc_solver/dsl.py`, `arc_solver/dsl_complete.py`) | Candidate solutions |
| Consequence | Correctness evaluation (`evaluate_performance.py`) | Reward signal |

**Stimulus Control:** NeuralGuidance adapts operation probabilities from features (`arc_solver/features.py`).

**Learning History:** EpisodicRetrieval stores reinforced program traces (`episodes.json`).

### 1.3 Review of Relational Modules
- `arc_solver/object_reasoning.py` provides object descriptors.
- `arc_solver/human_reasoning.py` encodes static geometric relations.
- Current system lacks operant relational learning — relations are hand-engineered rather than learned frames.

## 2. Re-Imagining the DSL as Verbal Behavior

### 2.1 Skinnerian Foundations
- **Tacts:** Labeling environmental stimuli.
- **Intraverbals:** Chains of verbal responses controlled by preceding responses.
- **Mands:** Responses motivated by establishing operations.

### 2.2 Tacting Module Proposal
- Extend features/objects to produce learned symbolic descriptors.
- Reinforce tact emission when associated programs solve tasks.

### 2.3 Intraverbal Chaining for Program Synthesis
- Treat program generation as conditional probability chain `P(op_n | grid_{n-1}, tacts)`.
- Allow shaping via progressive reinforcement on partial programs.

### 2.4 Behavioral Mapping Table
| Behavioral Concept | Module | Function |
| --- | --- | --- |
| Antecedent Stimulus | ARC task input | Evokes behavior |
| Behavior / Operant | DSL program | Acts on environment |
| Consequence | Grader output | Reinforcement |
| Reinforcement Mechanism | BehavioralEngine (`arc_solver/behavioral_engine.py`) | Reward delivery |
| Stimulus Control | NeuralGuidance | Probability shaping |
| Learning History | EpisodicRetrieval | Memory of reinforced behaviors |
| Tacting | Proposed module atop `arc_solver/features.py` | Descriptive labeling |
| Intraverbal Behavior | Enhanced search with chaining | Sequenced operations |
| Relational Framing | RFTEngine (`arc_solver/rft_engine/engine.py`) | Derived relations |


## 3. Engineering an RFT Engine for Novel Problem-Solving

### 3.1 Multiple Exemplar Training
- Mine relational patterns across solved tasks using object-centric state.

### 3.2 Derived Relational Responding
- Maintain relational graph with mutual/combinatorial entailment rules.

### 3.3 Transformation of Stimulus Functions
- Treat DSL applicability as stimulus functions.
- Transfer functions across derived equivalent stimuli to generalise behavior.

## 4. Implementation Roadmap

### 4.1 BehavioralEngine
- **Module:** `arc_solver/behavioral_engine.py`
- **Feature flag:** `PUMA_BEHAVIORAL_ENGINE` guarantees opt-in rollouts.
- **Reward Loop:** `RewardGrader` emits dense `[0.0, 1.0]` rewards with pixel/shape breakdowns.
- **Telemetry:** Structured JSON logs + moving-average metrics for monitoring.
- **Integration:** Broadcasts reinforcement to `NeuralGuidance.reinforce()` and episodic memory.

### 4.2 Module Adaptations
- **NeuralGuidance (`arc_solver/neural/guidance.py`):** Online updates from rewards and RFT hints.
- **EpisodicRetrieval (`arc_solver/neural/episodic.py`):** Tracks average reward per episode and ranks retrieval accordingly.
- **EnhancedSearch (`arc_solver/enhanced_search.py`):** Consumes updated guidance statistics without code changes.

### 4.3 Agentic Integration
- Embed RFT state into agentic solver observation space (`docs/AGENTIC_GENOMIC_SOLVER.md`).
- Use RL loop with structured rewards for intermediate progress.

## 5. Anticipated Capabilities and Future Work

### 5.1 Novel Problem-Solving
- Generalise via relational frames.
- Encourage creative intraverbal chaining.

### 5.2 Functional Contextualism in AI
- Emphasises behavior shaped by consequences over static knowledge.

### 5.3 Future Research
- Explore deictic framing (I/You, Here/There, Now/Then).
- Investigate self-modeling using extended RFT principles.
