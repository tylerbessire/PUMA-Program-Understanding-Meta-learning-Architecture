# RFT-ILP Neuro-Symbolic Architecture for ARC

## Overview

This document outlines a more robust approach to ARC problem-solving using **Relational Frame Theory (RFT)** combined with **Inductive Logic Programming (ILP)** in a neuro-symbolic architecture.

## Current State

The current PUMA solver has:
- ✅ Object inventory with persistent IDs
- ✅ RFT tracking for conflicts
- ✅ Pliance rules generation
- ✅ Learning from failures with memory persistence
- ✅ Entailment reasoning (mutual and combinatorial)

**However**, the solver struggles to generate correct programs because:
- ❌ Rules are too generic (know "size_reduction" but not specific logic)
- ❌ 0 programs generated for complex tasks
- ❌ Pliance rules have 0% accuracy on training pairs
- ❌ Missing structured relational decomposition

## Proposed Solution: RFT-ILP Neuro-Symbolic Architecture

### 4.1 Relational Decomposition into Logical Facts

Instead of generic pattern detection, transform grids into **structured logical facts**:

```prolog
% --- Facts for Input Grid of ex1 ---
input_object(ex1, obj1).
has_property(ex1, obj1, color, red).
has_property(ex1, obj1, size, 9).
has_property(ex1, obj1, shape, square).

input_object(ex1, obj2).
has_property(ex1, obj2, color, blue).
has_property(ex1, obj2, size, 4).

relation(ex1, input, obj1, obj2, is_enclosing).

% --- Facts for Output Grid of ex1 ---
output_object(ex1, obj3).
has_property(ex1, obj3, color, yellow).
has_property(ex1, obj3, size, 9).
has_property(ex1, obj3, shape, square).
```

This creates a **rich, symbolic knowledge base** describing the "before" and "after" states in a structured, relational format perfectly suited for logic-based reasoning.

### 4.2 Inductive Logic Programming (ILP) for Rule Discovery

**ILP is ideal** because its fundamental operation—generalizing from specific facts to abstract rules—is a direct computational analog of relational abstraction.

#### The ILP Learning Task:

- **Goal**: Induce a set of rules (logic program) that can explain the `output_...` facts using the `input_...` facts
- **Background Knowledge**: All DSL primitives framed as logical predicates (e.g., `move(In_Obj, Dx, Dy, Out_Obj)`)
- **Examples**: The `output_...` facts serve as positive examples

#### Example of Rule Induction:

For a task where red objects are removed and blue objects move right:

```prolog
% Rule 1: Blue objects in the input are moved right to become output objects.
output_object(Ex, Out_Obj) :-
    input_object(Ex, In_Obj),
    has_property(Ex, In_Obj, color, blue),
    move(In_Obj, 1, 0, Out_Obj).

% Note: No rule generates output from red input objects,
% effectively implementing their deletion.
```

This learned program:
- Is a **general, symbolic solution**
- Directly implements RFT's **combinatorial entailment**
- Chains together facts and background knowledge to derive outputs

### 4.3 Neuro-Symbolic Architecture for End-to-End Learning

**The Challenge**: ILP can't operate on raw pixels, neural networks struggle with crisp logical generalization.

**The Solution**: Combine both paradigms!

#### Proposed System Architecture:

1. **Neural Perception Front-End** (CNN or ViT)
   - Takes raw input/output grids
   - **Outputs**: Logical facts describing objects, properties, relations
   - Acts as "scene parser" converting pixels → structured symbolic language

2. **Symbolic Reasoning Back-End** (ILP Engine)
   - Receives facts from neural front-end for all demonstration pairs
   - Uses DSL primitives as background knowledge
   - Searches for simplest logic program consistent with facts
   - **Output**: Inferred solution program

3. **Program Executor** (DSL Interpreter)
   - Test input → Neural Front-End → Facts
   - Applies synthesized program to new facts
   - Generates output facts

4. **Grid Renderer**
   - Converts output facts → 2D grid
   - **Output**: Final solution

#### Key Advantage: Search Space Pruning

Neural front-end acts as powerful heuristic, dramatically pruning combinatorial search space:
- Instead of ILP considering all possible grid parses from pixels
- Neural network provides highly probable initial parse (objects + properties)
- Symbolic search focuses on smaller, abstract space of transformations

### 5.1 Complete System Flow

```
Input (JSON task)
    ↓
[Module 1: Neural Perception]
    → Facts: input_object(...), has_property(...), relation(...)
    ↓
[Module 2: ILP Program Synthesizer]
    → Uses DSL as background knowledge
    → Searches for concise logic program
    → Output: Abstract Syntax Tree / Horn clauses
    ↓
[Module 3: Program Execution Pipeline]
    → Test input → Neural Perception → Input facts
    → DSL Interpreter executes program on facts
    → Output: Final output facts
    ↓
[Module 4: Grid Renderer]
    → Facts → 2D grid
    → Output: Competition submission format
```

### Self-Improving Flywheel Effect

1. Better DSL → ILP solves more tasks
2. Solved tasks = perfect training data (noise-free)
3. Better training → Neural Perception produces cleaner facts
4. Better perception → ILP tackles more complex tasks
5. **Loop repeats**, cumulative knowledge grows

This creates the broad generalization necessary to surpass static systems!

## Integration with Current PUMA System

### Current State:
- Object inventory ≈ Input facts generation
- Pliance rules ≈ Simple rule patterns
- RFT tracking ≈ Conflict detection

### Needed Additions:

1. **Fact Generator Module**
   - Convert `ObjectInventory` entries → Prolog-style facts
   - Extract relations between objects
   - Generate both input and output facts

2. **ILP Engine Integration**
   - Python ILP libraries: `popper`, `metagol`, or `pyaleph`
   - Define DSL primitives as predicates
   - Search for consistent programs across demonstrations

3. **Program Verifier**
   - Test synthesized programs on training pairs
   - Measure accuracy before applying to test inputs

4. **Enhanced Learning**
   - Store successful programs in episodic memory
   - Use as background knowledge for future tasks
   - Build library of reusable program fragments

## Benefits Over Current Approach

| Current Approach | RFT-ILP Approach |
|-----------------|------------------|
| Generic patterns ("size_reduction") | Specific logical rules with parameters |
| 0% rule accuracy on training | Verifiable programs that work on training |
| 0 programs generated | Synthesized programs with provable correctness |
| Ad-hoc rule generation | Principled ILP search with guarantees |
| Limited generalization | Compositional, reusable program fragments |

## Next Steps

See `TODO_RFT_ILP.md` for implementation roadmap.

## References

- **Relational Frame Theory**: Hayes, Barnes-Holmes, & Roche (2001)
- **ILP for Program Synthesis**: Muggleton & De Raedt (1994)
- **Neuro-Symbolic AI**: Garcez, Gori, et al. (2019)
- **ARC Challenge**: Chollet (2019)

---

*This architecture represents a novel synthesis of concepts from cognitive psychology (RFT), logic programming (ILP), and deep learning (neuro-symbolic methods).*
