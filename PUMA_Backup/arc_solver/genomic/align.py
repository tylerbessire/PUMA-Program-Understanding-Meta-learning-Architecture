"""
Sequence alignment utilities for genomic analysis.

This module provides sequence alignment algorithms adapted for ARC
grid analysis, including run-aware scoring and mutation extraction.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum


class EditType(Enum):
    MATCH = "MATCH"
    SUBSTITUTION = "SUB"
    INSERTION = "INS"
    DELETION = "DEL"


@dataclass
class Edit:
    """Represents a single edit operation in an alignment."""
    edit_type: EditType
    pos1: int  # Position in sequence 1
    pos2: int  # Position in sequence 2
    token1: Optional[str] = None
    token2: Optional[str] = None
    score: float = 0.0


@dataclass
class Alignment:
    """Represents a complete sequence alignment."""
    sequence1: List[str]
    sequence2: List[str]
    edits: List[Edit]
    score: float
    
    def get_edit_script(self) -> List[Edit]:
        """Get the edit script (non-MATCH operations only)."""
        return [edit for edit in self.edits if edit.edit_type != EditType.MATCH]


class AlignmentScorer:
    """Scoring scheme for sequence alignment."""
    
    def __init__(self, match_score: float = 2.0, mismatch_penalty: float = -1.0,
                 gap_penalty: float = -2.0, run_bonus: float = 1.0):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.run_bonus = run_bonus
    
    def score_match(self, token1: str, token2: str) -> float:
        """Score a potential match between two tokens."""
        if token1 == token2:
            return self.match_score
        
        # Check if only color differs (neighborhood/edge same)
        parts1 = token1.split(':')
        parts2 = token2.split(':')
        
        if len(parts1) > 1 and len(parts2) > 1:
            if parts1[1:] == parts2[1:]:  # Same neighborhood/edge
                return self.match_score * 0.5  # Partial match
        
        return self.mismatch_penalty
    
    def score_gap(self, length: int = 1) -> float:
        """Score a gap of given length."""
        return self.gap_penalty * length
    
    def score_run_alignment(self, run1: Tuple[str, int], run2: Tuple[str, int]) -> float:
        """Score alignment of two runs (token, length pairs)."""
        token1, len1 = run1
        token2, len2 = run2
        
        base_score = self.score_match(token1, token2)
        
        if token1 == token2:
            # Bonus for aligning runs of same token
            run_alignment_bonus = self.run_bonus * min(len1, len2)
            length_penalty = abs(len1 - len2) * 0.1
            return base_score + run_alignment_bonus - length_penalty
        
        return base_score


def needleman_wunsch(seq1: List[str], seq2: List[str], 
                    scorer: AlignmentScorer = None) -> Alignment:
    """
    Perform global sequence alignment using Needleman-Wunsch algorithm.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        scorer: Scoring scheme
    
    Returns:
        Optimal global alignment
    """
    if scorer is None:
        scorer = AlignmentScorer()
    
    m, n = len(seq1), len(seq2)
    
    # Initialize DP table
    dp = np.zeros((m + 1, n + 1))
    
    # Initialize first row and column
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + scorer.score_gap()
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + scorer.score_gap()
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = dp[i-1][j-1] + scorer.score_match(seq1[i-1], seq2[j-1])
            delete_score = dp[i-1][j] + scorer.score_gap()
            insert_score = dp[i][j-1] + scorer.score_gap()
            
            dp[i][j] = max(match_score, delete_score, insert_score)
    
    # Traceback to get alignment
    edits = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = dp[i-1][j-1] + scorer.score_match(seq1[i-1], seq2[j-1])
            if dp[i][j] == match_score:
                # Match or substitution
                if seq1[i-1] == seq2[j-1]:
                    edit_type = EditType.MATCH
                else:
                    edit_type = EditType.SUBSTITUTION
                
                edits.append(Edit(edit_type, i-1, j-1, seq1[i-1], seq2[j-1]))
                i -= 1
                j -= 1
                continue
        
        if i > 0 and dp[i][j] == dp[i-1][j] + scorer.score_gap():
            # Deletion
            edits.append(Edit(EditType.DELETION, i-1, -1, seq1[i-1], None))
            i -= 1
        elif j > 0:
            # Insertion
            edits.append(Edit(EditType.INSERTION, -1, j-1, None, seq2[j-1]))
            j -= 1
        else:
            break
    
    edits.reverse()
    return Alignment(seq1, seq2, edits, dp[m][n])


def smith_waterman(seq1: List[str], seq2: List[str], 
                  scorer: AlignmentScorer = None) -> Alignment:
    """
    Perform local sequence alignment using Smith-Waterman algorithm.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        scorer: Scoring scheme
    
    Returns:
        Optimal local alignment
    """
    if scorer is None:
        scorer = AlignmentScorer()
    
    m, n = len(seq1), len(seq2)
    
    # Initialize DP table
    dp = np.zeros((m + 1, n + 1))
    
    max_score = 0
    max_i = max_j = 0
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = dp[i-1][j-1] + scorer.score_match(seq1[i-1], seq2[j-1])
            delete_score = dp[i-1][j] + scorer.score_gap()
            insert_score = dp[i][j-1] + scorer.score_gap()
            
            dp[i][j] = max(0, match_score, delete_score, insert_score)
            
            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_i, max_j = i, j
    
    # Traceback from maximum score position
    edits = []
    i, j = max_i, max_j
    
    while i > 0 and j > 0 and dp[i][j] > 0:
        if i > 0 and j > 0:
            match_score = dp[i-1][j-1] + scorer.score_match(seq1[i-1], seq2[j-1])
            if dp[i][j] == match_score:
                if seq1[i-1] == seq2[j-1]:
                    edit_type = EditType.MATCH
                else:
                    edit_type = EditType.SUBSTITUTION
                
                edits.append(Edit(edit_type, i-1, j-1, seq1[i-1], seq2[j-1]))
                i -= 1
                j -= 1
                continue
        
        if i > 0 and dp[i][j] == dp[i-1][j] + scorer.score_gap():
            edits.append(Edit(EditType.DELETION, i-1, -1, seq1[i-1], None))
            i -= 1
        elif j > 0:
            edits.append(Edit(EditType.INSERTION, -1, j-1, None, seq2[j-1]))
            j -= 1
        else:
            break
    
    edits.reverse()
    
    # Extract relevant subsequences
    start1 = min(edit.pos1 for edit in edits if edit.pos1 >= 0)
    end1 = max(edit.pos1 for edit in edits if edit.pos1 >= 0) + 1
    start2 = min(edit.pos2 for edit in edits if edit.pos2 >= 0)
    end2 = max(edit.pos2 for edit in edits if edit.pos2 >= 0) + 1
    
    subseq1 = seq1[start1:end1]
    subseq2 = seq2[start2:end2]
    
    return Alignment(subseq1, subseq2, edits, max_score)


def run_aware_alignment(rle1: List[Tuple[str, int]], rle2: List[Tuple[str, int]],
                       scorer: AlignmentScorer = None) -> List[Edit]:
    """
    Perform alignment on run-length encoded sequences.
    
    Args:
        rle1: First RLE sequence [(token, length), ...]
        rle2: Second RLE sequence [(token, length), ...]
        scorer: Scoring scheme
    
    Returns:
        List of edit operations on runs
    """
    if scorer is None:
        scorer = AlignmentScorer()
    
    m, n = len(rle1), len(rle2)
    dp = np.zeros((m + 1, n + 1))
    
    # Initialize boundaries
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + scorer.score_gap(rle1[i-1][1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + scorer.score_gap(rle2[j-1][1])
    
    # Fill DP table with run-aware scoring
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            run_score = dp[i-1][j-1] + scorer.score_run_alignment(rle1[i-1], rle2[j-1])
            delete_score = dp[i-1][j] + scorer.score_gap(rle1[i-1][1])
            insert_score = dp[i][j-1] + scorer.score_gap(rle2[j-1][1])
            
            dp[i][j] = max(run_score, delete_score, insert_score)
    
    # Traceback
    edits = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            run_score = dp[i-1][j-1] + scorer.score_run_alignment(rle1[i-1], rle2[j-1])
            if dp[i][j] == run_score:
                # Run alignment
                token1, len1 = rle1[i-1]
                token2, len2 = rle2[j-1]
                
                if token1 == token2:
                    edit_type = EditType.MATCH
                else:
                    edit_type = EditType.SUBSTITUTION
                
                edits.append(Edit(edit_type, i-1, j-1, f"{token1}x{len1}", f"{token2}x{len2}"))
                i -= 1
                j -= 1
                continue
        
        if i > 0 and dp[i][j] == dp[i-1][j] + scorer.score_gap(rle1[i-1][1]):
            token, length = rle1[i-1]
            edits.append(Edit(EditType.DELETION, i-1, -1, f"{token}x{length}", None))
            i -= 1
        elif j > 0:
            token, length = rle2[j-1]
            edits.append(Edit(EditType.INSERTION, -1, j-1, None, f"{token}x{length}"))
            j -= 1
        else:
            break
    
    edits.reverse()
    return edits


def merge_adjacent_edits(edits: List[Edit]) -> List[Edit]:
    """
    Merge adjacent edits of the same type into larger operations.
    This helps identify larger-scale patterns like duplications or inversions.
    """
    if not edits:
        return []
    
    merged = []
    current_group = [edits[0]]
    
    for edit in edits[1:]:
        # Check if this edit can be merged with the current group
        last_edit = current_group[-1]
        
        if (edit.edit_type == last_edit.edit_type and
            edit.edit_type != EditType.MATCH):  # Don't merge matches
            current_group.append(edit)
        else:
            # Finish current group and start new one
            if len(current_group) > 1:
                # Create merged edit
                first_edit = current_group[0]
                last_edit = current_group[-1]
                merged_edit = Edit(
                    first_edit.edit_type,
                    first_edit.pos1,
                    first_edit.pos2,
                    f"MERGED_{len(current_group)}",
                    f"MERGED_{len(current_group)}"
                )
                merged.append(merged_edit)
            else:
                merged.extend(current_group)
            
            current_group = [edit]
    
    # Don't forget the last group
    if len(current_group) > 1:
        first_edit = current_group[0]
        merged_edit = Edit(
            first_edit.edit_type,
            first_edit.pos1,
            first_edit.pos2,
            f"MERGED_{len(current_group)}",
            f"MERGED_{len(current_group)}"
        )
        merged.append(merged_edit)
    else:
        merged.extend(current_group)
    
    return merged


def detect_patterns_in_edits(edits: List[Edit]) -> Dict[str, Any]:
    """
    Analyze edit operations to detect higher-level patterns.
    
    Returns:
        Dictionary with detected patterns
    """
    patterns = {
        'substitutions': [],
        'insertions': [],
        'deletions': [],
        'duplications': [],
        'inversions': [],
        'transpositions': []
    }
    
    # Group edits by type
    for edit in edits:
        if edit.edit_type == EditType.SUBSTITUTION:
            patterns['substitutions'].append(edit)
        elif edit.edit_type == EditType.INSERTION:
            patterns['insertions'].append(edit)
        elif edit.edit_type == EditType.DELETION:
            patterns['deletions'].append(edit)
    
    # Look for duplication patterns (repeated insertions)
    insertion_tokens = [edit.token2 for edit in patterns['insertions'] if edit.token2]
    from collections import Counter
    token_counts = Counter(insertion_tokens)
    
    for token, count in token_counts.items():
        if count > 1:
            patterns['duplications'].append({
                'token': token,
                'count': count,
                'positions': [edit.pos2 for edit in patterns['insertions'] 
                             if edit.token2 == token]
            })
    
    # Simple inversion detection (consecutive deletions followed by insertions)
    # This is a simplified heuristic
    del_positions = [edit.pos1 for edit in patterns['deletions']]
    ins_positions = [edit.pos2 for edit in patterns['insertions']]
    
    if del_positions and ins_positions:
        if (max(del_positions) - min(del_positions) == len(del_positions) - 1 and
            max(ins_positions) - min(ins_positions) == len(ins_positions) - 1):
            patterns['inversions'].append({
                'del_range': (min(del_positions), max(del_positions)),
                'ins_range': (min(ins_positions), max(ins_positions))
            })
    
    return patterns