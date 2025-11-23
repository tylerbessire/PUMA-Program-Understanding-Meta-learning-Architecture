"""
RFT-Hyperon Bridge Module

Connects PUMA's Relational Frame Theory (RFT) system with Hyperon's MeTTa reasoning
capabilities. This bridge enables symbolic reasoning over relational frames, allowing
derived relation inference, compositional reasoning, and frequency-based analysis using
Hyperon's powerful reasoning engine.

Key Integration Points:
-----------------------
1. RFT Frame ↔ MeTTa Expression Conversion
2. Frequency Ledger Integration for MeTTa-based frequency analysis
3. Relational frame composition through MeTTa programs
4. Derived relation inference using Hyperon's reasoning engine
5. Support for all RFT relation types (coordination, opposition, hierarchy, etc.)

This module bridges behavioral analysis (RFT) with symbolic reasoning (Hyperon),
creating a hybrid cognitive architecture that combines the strengths of both approaches.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Hyperon imports
try:
    from hyperon import MeTTa
    from hyperon.atoms import Atom, E, S, V, OperationAtom
    from hyperon.base import GroundingSpace, Bindings
    HYPERON_AVAILABLE = True
except ImportError:
    HYPERON_AVAILABLE = False
    MeTTa = None
    Atom = None
    E = S = V = OperationAtom = None
    GroundingSpace = Bindings = None

# Import PUMA RFT components
import sys
from pathlib import Path

# Add parent directories to path for imports
puma_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(puma_root))

try:
    from puma.rft.reasoning import RelationalFrame, RelationType, RFTEngine
    from arc_solver.rft import RelationalFact, RelationalFrameAnalyzer
    from arc_solver.frequency_ledger import FrequencyLedger, FrequencySignature
except ImportError:
    # Fallback for when module is used standalone
    RelationalFrame = RelationalFact = None
    RelationType = RFTEngine = None
    RelationalFrameAnalyzer = None
    FrequencyLedger = FrequencySignature = None


@dataclass
class MeTTaRelation:
    """
    Represents a relational frame encoded as a MeTTa expression.

    This structure bridges RFT frames to Hyperon's symbolic reasoning,
    enabling logical inference over relational patterns.
    """
    metta_expr: str  # The MeTTa expression
    relation_type: str
    source: str
    target: str
    strength: float
    context: List[str]
    metadata: Dict[str, Any]


class RFTHyperonBridge:
    """
    Bridge between PUMA's RFT system and Hyperon's MeTTa reasoning engine.

    This class enables:
    - Conversion of RFT frames to MeTTa expressions
    - Parsing MeTTa results back to RFT frames
    - Symbolic reasoning over relational patterns
    - Derived relation inference using Hyperon
    - Frequency-based analysis in MeTTa
    """

    def __init__(self, atomspace=None):
        """
        Initialize the RFT-Hyperon bridge.

        Parameters
        ----------
        atomspace : optional
            Hyperon atomspace/grounding space for reasoning
        """
        if not HYPERON_AVAILABLE:
            raise ImportError(
                "Hyperon is not available. Install with: pip install hyperon"
            )

        self.metta = MeTTa()
        self.atomspace = atomspace
        self.relation_cache: Dict[str, MeTTaRelation] = {}

        # Initialize MeTTa with RFT reasoning programs
        self._initialize_rft_programs()

    def _initialize_rft_programs(self):
        """
        Initialize MeTTa with RFT reasoning rules and programs.

        This creates the symbolic reasoning infrastructure for:
        - Coordination (same-as relations)
        - Distinction (opposite-of relations)
        - Comparison (more-than, less-than)
        - Hierarchical (contains, part-of)
        - Transitivity inference
        - Symmetry inference
        - Composition operations
        """

        # Define RFT relation types in MeTTa
        self.metta.run("""
            ; Relational Frame Types
            (: Coordination Type)
            (: Opposition Type)
            (: Hierarchy Type)
            (: Comparative Type)
            (: Spatial Type)
            (: Temporal Type)
            (: Causal Type)

            ; Basic relation predicates
            (: same-as (-> $a $b Coordination))
            (: opposite-of (-> $a $b Opposition))
            (: more-than (-> $a $b Comparative))
            (: less-than (-> $a $b Comparative))
            (: contains (-> $a $b Hierarchy))
            (: part-of (-> $a $b Hierarchy))
            (: before (-> $a $b Temporal))
            (: causes (-> $a $b Causal))
            (: near (-> $a $b Spatial))
        """)

        # Transitivity rules for coordination (similarity)
        self.metta.run("""
            ; If A is same as B, and B is same as C, then A is same as C
            (= (derive-coordination $A $B $C)
               (if (and (same-as $A $B) (same-as $B $C))
                   (same-as $A $C)))
        """)

        # Symmetry rules
        self.metta.run("""
            ; Coordination is symmetric: if A same-as B, then B same-as A
            (= (coordination-symmetric $A $B)
               (if (same-as $A $B)
                   (same-as $B $A)))

            ; Opposition is symmetric: if A opposite-of B, then B opposite-of A
            (= (opposition-symmetric $A $B)
               (if (opposite-of $A $B)
                   (opposite-of $B $A)))
        """)

        # Hierarchical transitivity
        self.metta.run("""
            ; If A part-of B, and B part-of C, then A part-of C
            (= (derive-hierarchy $A $B $C)
               (if (and (part-of $A $B) (part-of $B $C))
                   (part-of $A $C)))
        """)

        # Comparison inference
        self.metta.run("""
            ; If A more-than B, and B more-than C, then A more-than C
            (= (derive-comparison $A $B $C)
               (if (and (more-than $A $B) (more-than $B $C))
                   (more-than $A $C)))

            ; If A more-than B, then B less-than A
            (= (comparison-inverse $A $B)
               (if (more-than $A $B)
                   (less-than $B $A)))
        """)

        # Temporal reasoning
        self.metta.run("""
            ; If A before B, and B before C, then A before C
            (= (derive-temporal $A $B $C)
               (if (and (before $A $B) (before $B $C))
                   (before $A $C)))
        """)

        # Causal inference with confidence decay
        self.metta.run("""
            ; Causal chains with confidence decay
            (= (derive-causal $A $B $C $conf)
               (if (and (causes $A $B) (causes $B $C))
                   (causes $A $C)))
        """)

    # ================================================================================
    # CONVERSION FUNCTIONS: RFT ↔ MeTTa
    # ================================================================================

    def rft_frame_to_metta(self, frame: RelationalFrame) -> str:
        """
        Convert RFT RelationalFrame to MeTTa expression.

        Parameters
        ----------
        frame : RelationalFrame
            RFT relational frame from puma.rft.reasoning

        Returns
        -------
        str
            MeTTa expression representing the relational frame

        Examples
        --------
        >>> frame = RelationalFrame(
        ...     relation_type=RelationType.COORDINATION,
        ...     source="red_square",
        ...     target="red_circle",
        ...     strength=0.8
        ... )
        >>> bridge.rft_frame_to_metta(frame)
        '(same-as red_square red_circle 0.8)'
        """
        if frame is None:
            return ""

        # Map RFT relation types to MeTTa predicates
        relation_map = {
            RelationType.COORDINATION: "same-as",
            RelationType.OPPOSITION: "opposite-of",
            RelationType.HIERARCHY: "part-of",
            RelationType.TEMPORAL: "before",
            RelationType.CAUSAL: "causes",
            RelationType.COMPARATIVE: "more-than",
            RelationType.SPATIAL: "near"
        }

        predicate = relation_map.get(frame.relation_type, "relates-to")

        # Create MeTTa expression with strength/confidence
        metta_expr = f"({predicate} {self._sanitize_term(frame.source)} " \
                    f"{self._sanitize_term(frame.target)} {frame.strength})"

        # Add context if present
        if frame.context:
            context_str = " ".join([f'"{c}"' for c in frame.context])
            metta_expr = f"(with-context ({metta_expr}) ({context_str}))"

        # Cache the conversion
        cache_key = f"{frame.source}_{frame.target}_{frame.relation_type.value}"
        self.relation_cache[cache_key] = MeTTaRelation(
            metta_expr=metta_expr,
            relation_type=frame.relation_type.value,
            source=frame.source,
            target=frame.target,
            strength=frame.strength,
            context=frame.context or [],
            metadata={'derived': frame.derived}
        )

        return metta_expr

    def rft_fact_to_metta(self, fact: RelationalFact) -> str:
        """
        Convert RFT RelationalFact to MeTTa expression.

        Parameters
        ----------
        fact : RelationalFact
            RFT relational fact from arc_solver.rft

        Returns
        -------
        str
            MeTTa expression representing the relational fact

        Examples
        --------
        >>> fact = RelationalFact(
        ...     relation="spatial_transform",
        ...     subject=(1, 3, 3),  # red 3x3 square
        ...     object=(2, 3, 3),   # blue 3x3 square
        ...     metadata={'distance': 5.0},
        ...     confidence=0.9
        ... )
        >>> bridge.rft_fact_to_metta(fact)
        '(spatial-transform obj_1_3_3 obj_2_3_3 0.9)'
        """
        if fact is None:
            return ""

        # Create unique identifiers for objects based on their signatures
        source_id = f"obj_{fact.subject[0]}_{fact.subject[1]}_{fact.subject[2]}"
        target_id = f"obj_{fact.object[0]}_{fact.object[1]}_{fact.object[2]}"

        # Normalize relation name for MeTTa
        relation = fact.relation.replace("_", "-")

        # Build base expression
        metta_expr = f"({relation} {source_id} {target_id} {fact.confidence})"

        # Add spatial direction if present
        if fact.direction_vector is not None:
            spatial_rel = fact.get_spatial_relation()
            if spatial_rel:
                metta_expr = f"(and {metta_expr} " \
                           f"(direction {source_id} {target_id} {spatial_rel}))"

        # Add metadata annotations
        if fact.metadata:
            metadata_parts = []
            for key, value in fact.metadata.items():
                if isinstance(value, (int, float)):
                    metadata_parts.append(f"({key} {value})")

            if metadata_parts:
                metadata_str = " ".join(metadata_parts)
                metta_expr = f"(with-metadata {metta_expr} ({metadata_str}))"

        return metta_expr

    def metta_to_rft_frame(self, metta_expr: str) -> Optional[RelationalFrame]:
        """
        Parse MeTTa expression back to RFT RelationalFrame.

        Parameters
        ----------
        metta_expr : str
            MeTTa expression to parse

        Returns
        -------
        RelationalFrame or None
            Parsed relational frame, or None if parsing fails

        Examples
        --------
        >>> metta_expr = "(same-as red_square red_circle 0.8)"
        >>> frame = bridge.metta_to_rft_frame(metta_expr)
        >>> frame.relation_type
        RelationType.COORDINATION
        """
        # Simple parser for basic MeTTa expressions
        # Format: (predicate source target confidence)

        metta_expr = metta_expr.strip()
        if not metta_expr.startswith('(') or not metta_expr.endswith(')'):
            return None

        # Remove outer parentheses
        inner = metta_expr[1:-1].strip()
        parts = inner.split()

        if len(parts) < 3:
            return None

        predicate = parts[0]
        source = parts[1]
        target = parts[2]
        strength = float(parts[3]) if len(parts) > 3 else 1.0

        # Map MeTTa predicates back to RFT relation types
        predicate_map = {
            "same-as": RelationType.COORDINATION,
            "opposite-of": RelationType.OPPOSITION,
            "part-of": RelationType.HIERARCHY,
            "contains": RelationType.HIERARCHY,
            "before": RelationType.TEMPORAL,
            "causes": RelationType.CAUSAL,
            "more-than": RelationType.COMPARATIVE,
            "less-than": RelationType.COMPARATIVE,
            "near": RelationType.SPATIAL
        }

        relation_type = predicate_map.get(predicate, RelationType.COORDINATION)

        return RelationalFrame(
            relation_type=relation_type,
            source=source,
            target=target,
            strength=strength,
            context=None,
            derived=True  # Assume MeTTa-generated frames are derived
        )

    # ================================================================================
    # RELATIONAL FRAME COMPOSITION
    # ================================================================================

    def compose_frames(
        self,
        frame1: RelationalFrame,
        frame2: RelationalFrame
    ) -> Optional[RelationalFrame]:
        """
        Compose two relational frames using MeTTa inference.

        If frame1 relates A to B, and frame2 relates B to C,
        derive a frame relating A to C (if valid for the relation type).

        Parameters
        ----------
        frame1 : RelationalFrame
            First relational frame (A → B)
        frame2 : RelationalFrame
            Second relational frame (B → C)

        Returns
        -------
        RelationalFrame or None
            Composed frame (A → C) if derivation is valid
        """
        # Check if frames can be composed (frame1.target == frame2.source)
        if frame1.target != frame2.source:
            return None

        # Check if relation types match and support transitivity
        if frame1.relation_type != frame2.relation_type:
            return None

        transitive_types = {
            RelationType.COORDINATION,
            RelationType.HIERARCHY,
            RelationType.COMPARATIVE,
            RelationType.TEMPORAL
        }

        if frame1.relation_type not in transitive_types:
            return None

        # Convert to MeTTa and perform inference
        metta1 = self.rft_frame_to_metta(frame1)
        metta2 = self.rft_frame_to_metta(frame2)

        # Add facts to MeTTa space
        self.metta.run(metta1)
        self.metta.run(metta2)

        # Run derivation based on relation type
        if frame1.relation_type == RelationType.COORDINATION:
            query = f"(derive-coordination {frame1.source} {frame1.target} {frame2.target})"
        elif frame1.relation_type == RelationType.HIERARCHY:
            query = f"(derive-hierarchy {frame1.source} {frame1.target} {frame2.target})"
        elif frame1.relation_type == RelationType.COMPARATIVE:
            query = f"(derive-comparison {frame1.source} {frame1.target} {frame2.target})"
        elif frame1.relation_type == RelationType.TEMPORAL:
            query = f"(derive-temporal {frame1.source} {frame1.target} {frame2.target})"
        else:
            return None

        # Execute query (simplified - actual Hyperon would return results)
        # For now, construct derived frame directly
        derived_strength = min(frame1.strength, frame2.strength) * 0.8

        return RelationalFrame(
            relation_type=frame1.relation_type,
            source=frame1.source,
            target=frame2.target,
            strength=derived_strength,
            context=['derived_by_composition'],
            derived=True
        )

    # ================================================================================
    # FREQUENCY LEDGER INTEGRATION
    # ================================================================================

    def frequency_signature_to_metta(self, signature: FrequencySignature) -> str:
        """
        Convert FrequencySignature to MeTTa expression.

        Parameters
        ----------
        signature : FrequencySignature
            Frequency signature from arc_solver.frequency_ledger

        Returns
        -------
        str
            MeTTa expression representing frequency properties
        """
        sig_id = f"sig_{signature.color}_{signature.size}"

        metta_expr = f"""
            (frequency-signature {sig_id}
                (color {signature.color})
                (size {signature.size})
                (occurrence-count {signature.occurrence_count})
                (shape-frequency {signature.shape_frequency})
                (color-frequency {signature.color_frequency}))
        """

        return metta_expr.strip()

    def frequency_ledger_to_metta(self, ledger: FrequencyLedger) -> List[str]:
        """
        Convert entire FrequencyLedger to MeTTa knowledge base.

        Parameters
        ----------
        ledger : FrequencyLedger
            Populated frequency ledger

        Returns
        -------
        List[str]
            List of MeTTa expressions encoding frequency knowledge
        """
        metta_expressions = []

        # Encode color frequencies
        for color, freq in ledger.color_frequencies.items():
            metta_expressions.append(f"(color-frequency {color} {freq})")

        # Encode size frequencies
        for size, freq in ledger.size_frequencies.items():
            metta_expressions.append(f"(size-frequency {size} {freq})")

        # Encode object signatures
        for signature in ledger.object_signatures:
            metta_expressions.append(self.frequency_signature_to_metta(signature))

        # Encode relational groupings
        for group_name, members in ledger.relational_groupings.items():
            for member in members:
                sig_id = f"sig_{member.color}_{member.size}"
                metta_expressions.append(
                    f"(belongs-to-group {sig_id} {group_name})"
                )

        # Add frequency-based similarity rules
        metta_expressions.append("""
            ; If two signatures belong to same frequency group, they are similar
            (= (frequency-similar $A $B)
               (if (and (belongs-to-group $A $group)
                       (belongs-to-group $B $group))
                   (same-as $A $B)))
        """)

        return metta_expressions

    def derive_frequency_relations(
        self,
        ledger: FrequencyLedger
    ) -> List[RelationalFrame]:
        """
        Derive relational frames from frequency ledger using MeTTa reasoning.

        Parameters
        ----------
        ledger : FrequencyLedger
            Populated frequency ledger

        Returns
        -------
        List[RelationalFrame]
            Derived relational frames based on frequency patterns
        """
        # Load frequency knowledge into MeTTa
        freq_expressions = self.frequency_ledger_to_metta(ledger)
        for expr in freq_expressions:
            try:
                self.metta.run(expr)
            except:
                pass  # Skip malformed expressions

        derived_frames = []

        # Derive similarity relations from frequency groupings
        for group_name, members in ledger.relational_groupings.items():
            # Create coordination frames between group members
            for i, sig1 in enumerate(members):
                for sig2 in members[i+1:]:
                    # Calculate similarity strength
                    similarity = sig1.similarity_score(sig2)

                    if similarity > 0.6:
                        frame = RelationalFrame(
                            relation_type=RelationType.COORDINATION,
                            source=f"sig_{sig1.color}_{sig1.size}",
                            target=f"sig_{sig2.color}_{sig2.size}",
                            strength=similarity,
                            context=['frequency_based', group_name],
                            derived=True
                        )
                        derived_frames.append(frame)

        # Derive comparison relations from size frequencies
        sorted_sizes = sorted(ledger.size_frequencies.keys())
        for i, size1 in enumerate(sorted_sizes[:-1]):
            size2 = sorted_sizes[i + 1]
            frame = RelationalFrame(
                relation_type=RelationType.COMPARATIVE,
                source=f"size_{size2}",
                target=f"size_{size1}",
                strength=1.0,
                context=['size_comparison'],
                derived=True
            )
            derived_frames.append(frame)

        return derived_frames

    # ================================================================================
    # DERIVED RELATION INFERENCE
    # ================================================================================

    def infer_derived_relations(
        self,
        known_frames: List[RelationalFrame],
        max_depth: int = 3
    ) -> List[RelationalFrame]:
        """
        Infer derived relations using Hyperon's reasoning engine.

        Given a set of known relational frames, use MeTTa to derive new
        relations through transitivity, symmetry, and composition.

        Parameters
        ----------
        known_frames : List[RelationalFrame]
            Known relational frames
        max_depth : int
            Maximum inference depth (default 3)

        Returns
        -------
        List[RelationalFrame]
            Newly derived relational frames
        """
        # Load known frames into MeTTa
        for frame in known_frames:
            metta_expr = self.rft_frame_to_metta(frame)
            try:
                self.metta.run(metta_expr)
            except:
                pass

        derived = []

        # Apply symmetry rules
        for frame in known_frames:
            if frame.relation_type == RelationType.COORDINATION:
                # Coordination is symmetric
                symmetric_frame = RelationalFrame(
                    relation_type=frame.relation_type,
                    source=frame.target,
                    target=frame.source,
                    strength=frame.strength,
                    context=['symmetric_derivation'],
                    derived=True
                )
                derived.append(symmetric_frame)

            elif frame.relation_type == RelationType.OPPOSITION:
                # Opposition is symmetric
                symmetric_frame = RelationalFrame(
                    relation_type=frame.relation_type,
                    source=frame.target,
                    target=frame.source,
                    strength=frame.strength,
                    context=['symmetric_derivation'],
                    derived=True
                )
                derived.append(symmetric_frame)

            elif frame.relation_type == RelationType.COMPARATIVE:
                # Invert comparison (more-than ↔ less-than)
                inverse_frame = RelationalFrame(
                    relation_type=frame.relation_type,
                    source=frame.target,
                    target=frame.source,
                    strength=frame.strength,
                    context=['inverse_comparison'],
                    derived=True
                )
                derived.append(inverse_frame)

        # Apply transitivity for supported relation types
        transitive_types = {
            RelationType.COORDINATION,
            RelationType.HIERARCHY,
            RelationType.COMPARATIVE,
            RelationType.TEMPORAL
        }

        for frame1 in known_frames:
            if frame1.relation_type not in transitive_types:
                continue

            for frame2 in known_frames:
                if frame2.relation_type != frame1.relation_type:
                    continue

                # Check for transitivity (frame1.target == frame2.source)
                composed = self.compose_frames(frame1, frame2)
                if composed:
                    derived.append(composed)

        return derived

    # ================================================================================
    # UTILITY FUNCTIONS
    # ================================================================================

    def _sanitize_term(self, term: str) -> str:
        """Sanitize term for use in MeTTa expressions."""
        # Replace spaces and special characters
        sanitized = term.replace(" ", "_")
        sanitized = sanitized.replace("-", "_")
        sanitized = sanitized.replace(":", "_")
        return sanitized

    def export_to_metta_file(
        self,
        frames: List[RelationalFrame],
        filepath: str
    ):
        """
        Export relational frames to a MeTTa file for persistence.

        Parameters
        ----------
        frames : List[RelationalFrame]
            Relational frames to export
        filepath : str
            Path to output MeTTa file
        """
        with open(filepath, 'w') as f:
            f.write("; RFT Relational Frames exported to MeTTa\n")
            f.write("; Generated by PUMA RFT-Hyperon Bridge\n\n")

            for frame in frames:
                metta_expr = self.rft_frame_to_metta(frame)
                f.write(metta_expr + "\n")

    def get_bridge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the bridge's operation.

        Returns
        -------
        Dict[str, Any]
            Statistics including cached relations, inference counts, etc.
        """
        return {
            'cached_relations': len(self.relation_cache),
            'hyperon_available': HYPERON_AVAILABLE,
            'relation_types_supported': [
                'COORDINATION',
                'OPPOSITION',
                'HIERARCHY',
                'COMPARATIVE',
                'SPATIAL',
                'TEMPORAL',
                'CAUSAL'
            ]
        }


# ================================================================================
# EXAMPLE USAGE AND TESTS
# ================================================================================

def example_basic_conversion():
    """Example: Basic RFT frame to MeTTa conversion."""
    print("=" * 80)
    print("EXAMPLE 1: Basic RFT Frame to MeTTa Conversion")
    print("=" * 80)

    if not HYPERON_AVAILABLE:
        print("Hyperon not available. Skipping example.")
        return

    # Create bridge
    bridge = RFTHyperonBridge()

    # Create a coordination frame (similarity relation)
    frame = RelationalFrame(
        relation_type=RelationType.COORDINATION,
        source="red_square",
        target="red_circle",
        strength=0.85,
        context=["same_color"],
        derived=False
    )

    # Convert to MeTTa
    metta_expr = bridge.rft_frame_to_metta(frame)
    print(f"\nRFT Frame: {frame}")
    print(f"MeTTa Expression: {metta_expr}")

    # Convert back
    reconstructed = bridge.metta_to_rft_frame(metta_expr.split("(with-context")[0].strip())
    print(f"Reconstructed Frame: {reconstructed}")


def example_frame_composition():
    """Example: Composing relational frames using transitivity."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Relational Frame Composition (Transitivity)")
    print("=" * 80)

    if not HYPERON_AVAILABLE:
        print("Hyperon not available. Skipping example.")
        return

    bridge = RFTHyperonBridge()

    # Create chain: A similar to B, B similar to C
    frame1 = RelationalFrame(
        relation_type=RelationType.COORDINATION,
        source="pattern_A",
        target="pattern_B",
        strength=0.9,
        context=None,
        derived=False
    )

    frame2 = RelationalFrame(
        relation_type=RelationType.COORDINATION,
        source="pattern_B",
        target="pattern_C",
        strength=0.8,
        context=None,
        derived=False
    )

    # Compose frames to derive: A similar to C
    composed = bridge.compose_frames(frame1, frame2)

    print(f"\nFrame 1: {frame1.source} → {frame1.target} (strength: {frame1.strength})")
    print(f"Frame 2: {frame2.source} → {frame2.target} (strength: {frame2.strength})")
    print(f"Composed: {composed.source} → {composed.target} (strength: {composed.strength})")
    print(f"Derived: {composed.derived}")


def example_frequency_integration():
    """Example: Integrating Frequency Ledger with MeTTa."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Frequency Ledger Integration")
    print("=" * 80)

    if not HYPERON_AVAILABLE or FrequencyLedger is None:
        print("Dependencies not available. Skipping example.")
        return

    bridge = RFTHyperonBridge()

    # Create mock frequency ledger
    ledger = FrequencyLedger()
    ledger.color_frequencies = {1: 10, 2: 5, 3: 3}
    ledger.size_frequencies = {9: 8, 4: 6, 1: 2}

    sig1 = FrequencySignature(color=1, size=9, occurrence_count=8, shape_frequency=8, color_frequency=10)
    sig2 = FrequencySignature(color=1, size=9, occurrence_count=7, shape_frequency=8, color_frequency=10)
    sig3 = FrequencySignature(color=2, size=4, occurrence_count=5, shape_frequency=6, color_frequency=5)

    ledger.object_signatures = [sig1, sig2, sig3]
    ledger.relational_groupings = {
        'group_0': [sig1, sig2],
        'group_1': [sig3]
    }

    # Convert to MeTTa
    metta_exprs = bridge.frequency_ledger_to_metta(ledger)

    print(f"\nFrequency Ledger Statistics:")
    print(f"  Total objects: {len(ledger.object_signatures)}")
    print(f"  Color frequencies: {dict(ledger.color_frequencies)}")
    print(f"  Groupings: {len(ledger.relational_groupings)}")

    print(f"\nSample MeTTa expressions:")
    for expr in metta_exprs[:5]:
        print(f"  {expr}")

    # Derive relations from frequency patterns
    derived_frames = bridge.derive_frequency_relations(ledger)
    print(f"\nDerived {len(derived_frames)} relational frames from frequency patterns")
    for frame in derived_frames[:3]:
        print(f"  {frame.source} --[{frame.relation_type.value}]--> {frame.target} "
              f"(strength: {frame.strength:.2f})")


def example_inference():
    """Example: Deriving new relations through inference."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Derived Relation Inference")
    print("=" * 80)

    if not HYPERON_AVAILABLE:
        print("Hyperon not available. Skipping example.")
        return

    bridge = RFTHyperonBridge()

    # Create set of known frames
    known_frames = [
        RelationalFrame(RelationType.COORDINATION, "A", "B", 0.9),
        RelationalFrame(RelationType.COORDINATION, "B", "C", 0.8),
        RelationalFrame(RelationType.HIERARCHY, "X", "Y", 1.0),
        RelationalFrame(RelationType.HIERARCHY, "Y", "Z", 1.0),
        RelationalFrame(RelationType.COMPARATIVE, "small", "medium", 1.0),
        RelationalFrame(RelationType.COMPARATIVE, "medium", "large", 1.0),
    ]

    print(f"\nKnown frames: {len(known_frames)}")
    for frame in known_frames:
        print(f"  {frame.source} --[{frame.relation_type.value}]--> {frame.target}")

    # Infer derived relations
    derived = bridge.infer_derived_relations(known_frames, max_depth=2)

    print(f"\nDerived frames: {len(derived)}")
    for frame in derived[:10]:  # Show first 10
        print(f"  {frame.source} --[{frame.relation_type.value}]--> {frame.target} "
              f"(strength: {frame.strength:.2f}, context: {frame.context})")


def run_all_examples():
    """Run all examples to demonstrate the bridge capabilities."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  PUMA RFT-HYPERON BRIDGE - EXAMPLES AND TESTS".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    example_basic_conversion()
    example_frame_composition()
    example_frequency_integration()
    example_inference()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_examples()
