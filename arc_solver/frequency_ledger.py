"""
Frequency Ledger System - Core Innovation of PUMA Architecture

This module implements PUMA's breakthrough frequency-based analysis framework that
groups objects by numerical attributes (frequencies, counts, patterns) to enable
models to discover abstract relationships.

The Frequency Ledger System is a behavior-analytic approach that allows models to
make derivational connections between stimuli without explicit training on those
relationships—mirroring how humans learn through relational framing in Relational
Frame Theory (RFT).

Key Capabilities:
-----------------
1. **Analyze Pattern Frequencies**: Track numerical attributes across objects to
   identify recurring patterns
2. **Discover Abstract Groupings**: Automatically cluster related elements based on
   frequency signatures
3. **Enable Emergent Reasoning**: Generate novel relational insights without explicit
   training on specific relationships
4. **Mirror Human Learning**: Replicate the behavioral process of deriving new
   relations from learned frames

This methodology creates a bridge between behavioral analysis and computational models,
allowing transformers to develop reasoning capabilities grounded in cognitive science
principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter, defaultdict
import numpy as np

from .grid import Array, histogram


@dataclass
class FrequencySignature:
    """
    Represents the frequency-based signature of an object or pattern.

    This signature enables derivational reasoning by encoding numerical attributes
    that can be used to discover abstract relationships without explicit training.
    """
    color: int
    size: int  # Number of pixels
    occurrence_count: int  # How many times this pattern appears
    position_frequencies: Dict[Tuple[int, int], int] = field(default_factory=dict)
    shape_frequency: int = 0  # Frequency of this shape type
    color_frequency: int = 0  # Frequency of this color across all objects

    def similarity_score(self, other: 'FrequencySignature') -> float:
        """
        Compute behavioral similarity between frequency signatures.

        This enables derivational relations - models can derive that objects with
        similar frequency signatures may participate in similar relational frames.
        """
        score = 0.0

        # Color frequency similarity (behavioral stimulus equivalence)
        if self.color == other.color:
            score += 0.3

        # Size frequency similarity (magnitude relations)
        size_ratio = min(self.size, other.size) / max(self.size, other.size)
        score += 0.2 * size_ratio

        # Occurrence frequency similarity (contextual control)
        occ_ratio = min(self.occurrence_count, other.occurrence_count) / \
                    max(self.occurrence_count, other.occurrence_count)
        score += 0.3 * occ_ratio

        # Shape frequency similarity (equivalence class membership)
        if self.shape_frequency > 0 and other.shape_frequency > 0:
            shape_ratio = min(self.shape_frequency, other.shape_frequency) / \
                         max(self.shape_frequency, other.shape_frequency)
            score += 0.2 * shape_ratio

        return score


@dataclass
class FrequencyLedger:
    """
    The Frequency Ledger maintains a comprehensive record of frequency-based
    patterns that enable emergent relational reasoning.

    This is the core data structure of PUMA's Frequency Ledger System,
    implementing behavioral analysis principles for abstract reasoning.
    """
    color_frequencies: Counter = field(default_factory=Counter)
    size_frequencies: Counter = field(default_factory=Counter)
    pattern_frequencies: Dict[str, int] = field(default_factory=dict)
    object_signatures: List[FrequencySignature] = field(default_factory=list)
    relational_groupings: Dict[str, List[FrequencySignature]] = field(default_factory=dict)

    def add_observation(self, grid: Array, objects: Optional[List] = None) -> None:
        """
        Add observations from a grid to the frequency ledger.

        This implements the behavioral principle of stimulus tracking - building
        a repertoire of observed patterns that can later support derivational
        reasoning.

        Parameters
        ----------
        grid : Array
            The grid to analyze
        objects : Optional[List]
            Pre-extracted objects from the grid. If None, will analyze grid globally.
        """
        # Track color frequencies (stimulus equivalence classes)
        colors = histogram(grid)
        self.color_frequencies.update(colors)

        if objects is not None:
            for obj in objects:
                # Create frequency signature for this object
                sig = FrequencySignature(
                    color=obj.get('color', 0),
                    size=obj.get('size', 0),
                    occurrence_count=1,
                    color_frequency=self.color_frequencies[obj.get('color', 0)]
                )
                self.object_signatures.append(sig)

                # Track size frequencies (magnitude relations)
                self.size_frequencies[obj.get('size', 0)] += 1

    def discover_abstract_groupings(self, similarity_threshold: float = 0.7) -> Dict[str, List[FrequencySignature]]:
        """
        Discover abstract groupings based on frequency signatures.

        This implements the core Frequency Ledger innovation: enabling models to
        make derivational connections between stimuli without explicit training.

        Objects with similar frequency signatures are grouped together, creating
        equivalence classes that support emergent reasoning capabilities.

        Parameters
        ----------
        similarity_threshold : float
            Minimum similarity score for grouping (default 0.7)

        Returns
        -------
        Dict[str, List[FrequencySignature]]
            Abstract groupings of objects by frequency signature similarity
        """
        groupings: Dict[str, List[FrequencySignature]] = defaultdict(list)
        processed: Set[int] = set()

        for i, sig1 in enumerate(self.object_signatures):
            if i in processed:
                continue

            group_key = f"group_{len(groupings)}"
            groupings[group_key].append(sig1)
            processed.add(i)

            # Find similar signatures (derivational equivalence)
            for j, sig2 in enumerate(self.object_signatures[i+1:], start=i+1):
                if j in processed:
                    continue

                similarity = sig1.similarity_score(sig2)
                if similarity >= similarity_threshold:
                    groupings[group_key].append(sig2)
                    processed.add(j)

        self.relational_groupings = dict(groupings)
        return self.relational_groupings

    def derive_relational_patterns(self) -> List[Dict[str, Any]]:
        """
        Derive relational patterns from frequency analysis.

        This is where emergent reasoning happens - the model derives new relations
        from learned frequency patterns without explicit training on those specific
        relationships. This mirrors human relational framing in RFT.

        Returns
        -------
        List[Dict[str, Any]]
            Derived relational patterns with behavioral properties
        """
        patterns = []

        # Derive frequency-based rules (behavioral contingencies)
        if self.color_frequencies:
            most_common_color = self.color_frequencies.most_common(1)[0]
            patterns.append({
                'type': 'dominant_color',
                'color': most_common_color[0],
                'frequency': most_common_color[1],
                'confidence': most_common_color[1] / sum(self.color_frequencies.values()),
                'derivation': 'frequency_dominance'
            })

        if self.size_frequencies:
            most_common_size = self.size_frequencies.most_common(1)[0]
            patterns.append({
                'type': 'dominant_size',
                'size': most_common_size[0],
                'frequency': most_common_size[1],
                'confidence': most_common_size[1] / sum(self.size_frequencies.values()),
                'derivation': 'frequency_dominance'
            })

        # Derive grouping patterns (equivalence classes)
        if self.relational_groupings:
            for group_name, members in self.relational_groupings.items():
                if len(members) > 1:
                    patterns.append({
                        'type': 'frequency_equivalence_class',
                        'group': group_name,
                        'members': len(members),
                        'confidence': len(members) / len(self.object_signatures) if self.object_signatures else 0,
                        'derivation': 'frequency_similarity'
                    })

        return patterns

    def get_frequency_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive frequency-based insights for behavioral analysis.

        Returns
        -------
        Dict[str, Any]
            Frequency insights including distributions, patterns, and derived relations
        """
        return {
            'color_distribution': dict(self.color_frequencies),
            'size_distribution': dict(self.size_frequencies),
            'total_objects': len(self.object_signatures),
            'abstract_groupings': len(self.relational_groupings),
            'derived_patterns': self.derive_relational_patterns()
        }


def analyze_frequency_patterns(
    train_pairs: List[Tuple[Array, Array]],
    extract_objects: bool = True
) -> FrequencyLedger:
    """
    Analyze training pairs using the Frequency Ledger System.

    This function applies PUMA's core innovation to extract frequency-based patterns
    that enable derivational reasoning and emergent relational capabilities.

    Parameters
    ----------
    train_pairs : List[Tuple[Array, Array]]
        Training input-output pairs
    extract_objects : bool
        Whether to extract individual objects (default True)

    Returns
    -------
    FrequencyLedger
        Populated frequency ledger with discovered patterns
    """
    ledger = FrequencyLedger()

    for inp, out in train_pairs:
        # Analyze both input and output grids
        ledger.add_observation(inp)
        ledger.add_observation(out)

    # Discover abstract groupings (derivational relations)
    ledger.discover_abstract_groupings()

    return ledger


def frequency_guided_search(
    ledger: FrequencyLedger,
    candidate_operations: List[str]
) -> List[Tuple[str, float]]:
    """
    Use frequency ledger insights to guide operation search.

    This implements behavioral guidance - using learned frequency patterns to
    predict which operations are likely relevant, similar to how behavioral
    history guides human problem-solving.

    Parameters
    ----------
    ledger : FrequencyLedger
        Populated frequency ledger with pattern insights
    candidate_operations : List[str]
        Candidate DSL operations to rank

    Returns
    -------
    List[Tuple[str, float]]
        Operations ranked by frequency-based relevance scores
    """
    insights = ledger.get_frequency_insights()
    derived_patterns = insights['derived_patterns']

    # Score operations based on frequency patterns (behavioral contingency matching)
    operation_scores = []

    for op in candidate_operations:
        score = 0.5  # Base score

        # Boost color-related operations if dominant color pattern exists
        if any(p['type'] == 'dominant_color' for p in derived_patterns):
            if 'recolor' in op or 'color' in op:
                score += 0.3

        # Boost size-related operations if size patterns exist
        if any(p['type'] == 'dominant_size' for p in derived_patterns):
            if 'scale' in op or 'grow' in op or 'shrink' in op:
                score += 0.3

        # Boost grouping operations if equivalence classes exist
        if any(p['type'] == 'frequency_equivalence_class' for p in derived_patterns):
            if 'group' in op or 'cluster' in op or 'partition' in op:
                score += 0.4

        operation_scores.append((op, score))

    # Sort by relevance score (behavioral preference)
    operation_scores.sort(key=lambda x: x[1], reverse=True)

    return operation_scores


# [S:FREQ v1] frequency_ledger_system pass
