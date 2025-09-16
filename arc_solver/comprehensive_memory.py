"""
Comprehensive Memory System for ARC Solver.

This module provides access to patterns learned from the training data,
including successful transformation patterns and RFT facts.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import os

from .grid import to_array, Array


class ComprehensiveMemory:
    """Memory system with patterns from 1000+ training examples."""
    
    def __init__(self, memory_path: str = "fast_comprehensive_memory.json"):
        self.memory_data = None
        self.pattern_index = defaultdict(list)
        self.size_pattern_index = defaultdict(list)
        self.rft_facts = []
        
        self.load_memory(memory_path)
        self.build_indices()
    
    def load_memory(self, memory_path: str):
        """Load comprehensive memory from file."""
        if os.path.exists(memory_path):
            with open(memory_path, 'r') as f:
                self.memory_data = json.load(f)
            print(f"Loaded comprehensive memory: {len(self.memory_data['successful_examples'])} successful patterns")
        else:
            print(f"Warning: Memory file {memory_path} not found, using empty memory")
            self.memory_data = {
                'successful_examples': [],
                'pattern_counts': {},
                'rft_facts': []
            }
    
    def build_indices(self):
        """Build efficient indices for pattern lookup."""
        if not self.memory_data:
            return
        
        # Index by pattern type
        for example in self.memory_data['successful_examples']:
            pattern_type = example['primary_pattern']
            self.pattern_index[pattern_type].append(example)
            
            # Index by size change for extraction tasks
            if example['size_change']:
                self.size_pattern_index[example['size_change']].append(example)
        
        # Store RFT facts
        self.rft_facts = self.memory_data.get('rft_facts', [])
        
        print(f"Built indices: {len(self.pattern_index)} pattern types, {len(self.rft_facts)} RFT facts")
    
    def find_similar_patterns(self, train_pairs: List[Tuple[Array, Array]]) -> List[Dict[str, Any]]:
        """Find similar transformation patterns from memory."""
        if not train_pairs or not self.memory_data:
            return []
        
        # Analyze the current task
        task_signature = self.analyze_task_signature(train_pairs)
        print(f"DEBUG: Task signature: {task_signature}")
        
        # Find matching patterns
        candidates = []
        
        # Match by primary pattern type
        primary_pattern = task_signature['primary_pattern']
        print(f"DEBUG: Looking for pattern type '{primary_pattern}' in {list(self.pattern_index.keys())}")
        
        if primary_pattern in self.pattern_index:
            candidates.extend(self.pattern_index[primary_pattern])
            print(f"DEBUG: Found {len(self.pattern_index[primary_pattern])} candidates for {primary_pattern}")
        
        # For extraction tasks, always try broader pattern matching  
        if primary_pattern == 'extraction':
            # Add all extraction patterns regardless of exact size
            extraction_patterns = self.pattern_index.get('extraction', [])
            candidates.extend(extraction_patterns)
            print(f"DEBUG: Added {len(extraction_patterns)} extraction patterns")
            
        # Match by size change for extraction tasks
        size_change = task_signature.get('size_change')
        if size_change and size_change in self.size_pattern_index:
            size_matches = self.size_pattern_index[size_change]
            candidates.extend(size_matches)
            print(f"DEBUG: Found {len(size_matches)} size change matches for {size_change}")
        
        # If still no candidates, be more aggressive
        if not candidates:
            if primary_pattern == 'extraction':
                # Try all extraction patterns
                candidates.extend(self.pattern_index.get('extraction', []))
            elif primary_pattern == 'same_size':
                # Try complex same size patterns
                candidates.extend(self.pattern_index.get('complex_same_size', []))
        
        # Remove duplicates and rank by similarity
        unique_candidates = {}
        for candidate in candidates:
            task_id = candidate['task_id']
            if task_id not in unique_candidates:
                similarity = self.calculate_similarity(task_signature, candidate)
                unique_candidates[task_id] = {
                    'pattern': candidate,
                    'similarity': similarity
                }
        
        # Sort by similarity and return top matches
        ranked_candidates = sorted(unique_candidates.values(), 
                                 key=lambda x: x['similarity'], reverse=True)
        
        return [c['pattern'] for c in ranked_candidates[:10]]
    
    def analyze_task_signature(self, train_pairs: List[Tuple[Array, Array]]) -> Dict[str, Any]:
        """Analyze task to create signature for pattern matching."""
        if not train_pairs:
            return {'primary_pattern': 'unknown'}
        
        inp, out = train_pairs[0]
        inp_h, inp_w = inp.shape
        out_h, out_w = out.shape
        
        signature = {
            'input_size': (inp_h, inp_w),
            'output_size': (out_h, out_w),
            'size_change': None,
            'primary_pattern': 'unknown',
            'color_change': False
        }
        
        # Analyze size change
        if inp_h == out_h and inp_w == out_w:
            signature['primary_pattern'] = 'same_size'
            
            # Check for simple transformations
            if np.array_equal(inp, out):
                signature['primary_pattern'] = 'identity'
            elif np.array_equal(np.rot90(inp, 1), out):
                signature['primary_pattern'] = 'rotation'
            elif np.array_equal(np.flipud(inp), out) or np.array_equal(np.fliplr(inp), out):
                signature['primary_pattern'] = 'reflection'
            else:
                # Check for color changes
                inp_colors = set(inp.flatten())
                out_colors = set(out.flatten())
                if inp_colors != out_colors:
                    signature['primary_pattern'] = 'recolor'
                    signature['color_change'] = True
                else:
                    signature['primary_pattern'] = 'complex_same_size'
        else:
            if out_h * out_w < inp_h * inp_w:
                signature['primary_pattern'] = 'extraction'
                signature['size_change'] = f"{inp_h}x{inp_w}_to_{out_h}x{out_w}"
            else:
                signature['primary_pattern'] = 'expansion'
                signature['size_change'] = f"{inp_h}x{inp_w}_to_{out_h}x{out_w}"
        
        return signature
    
    def calculate_similarity(self, task_signature: Dict, candidate: Dict) -> float:
        """Calculate similarity between task signature and candidate pattern."""
        similarity = 0.0
        
        # Primary pattern match (most important)
        if task_signature['primary_pattern'] == candidate['primary_pattern']:
            similarity += 0.5
        
        # Size change match
        if task_signature.get('size_change') == candidate.get('size_change'):
            similarity += 0.3
        
        # Input/output size similarity
        task_input_size = task_signature.get('input_size', (0, 0))
        task_output_size = task_signature.get('output_size', (0, 0))
        
        # This would need to be extracted from candidate data in a real implementation
        # For now, give partial credit
        similarity += 0.2
        
        return similarity
    
    def get_suggested_operations(self, train_pairs: List[Tuple[Array, Array]]) -> List[Tuple[str, Dict]]:
        """Get suggested operations based on memory patterns."""
        similar_patterns = self.find_similar_patterns(train_pairs)
        print(f"DEBUG: Found {len(similar_patterns)} similar patterns")
        
        suggested_ops = []
        confidence_scores = defaultdict(float)
        
        for i, pattern in enumerate(similar_patterns[:5]):  # Debug first 5
            operations = pattern.get('suggested_operations', [])
            pattern_confidence = pattern.get('confidence', 0.5)
            print(f"DEBUG: Pattern {i} has {len(operations)} operations: {operations}")
            
            for op in operations:
                if isinstance(op, list) and len(op) == 2:  # Handle [op_name, params] format
                    op_name, params = op
                    suggested_ops.append((op_name, params))
                    confidence_scores[op_name] += pattern_confidence
                elif isinstance(op, tuple):
                    op_name, params = op
                    suggested_ops.append((op_name, params))
                    confidence_scores[op_name] += pattern_confidence
                elif isinstance(op, str):
                    suggested_ops.append((op, {}))
                    confidence_scores[op] += pattern_confidence
                else:
                    print(f"DEBUG: Unknown operation format: {op} (type: {type(op)})")
        
        # Sort by confidence and remove duplicates
        unique_ops = []
        seen_ops = set()
        
        for op_name, confidence in sorted(confidence_scores.items(), 
                                        key=lambda x: x[1], reverse=True):
            if op_name not in seen_ops:
                # Find the best parameters for this operation
                best_params = {}
                for op_tuple in suggested_ops:
                    if op_tuple[0] == op_name:
                        best_params = op_tuple[1]
                        break
                
                unique_ops.append((op_name, best_params))
                seen_ops.add(op_name)
        
        return unique_ops[:15]  # Return top 15 suggestions
    
    def get_rft_facts_for_pattern(self, pattern_type: str) -> List[Dict]:
        """Get RFT facts relevant to a specific pattern type."""
        relevant_facts = []
        
        for fact in self.rft_facts:
            if (fact.get('type') == f"{pattern_type}_transformation" or
                fact.get('relation') == pattern_type):
                relevant_facts.append(fact)
        
        return relevant_facts
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded memory."""
        if not self.memory_data:
            return {'status': 'no_memory_loaded'}
        
        return {
            'total_successful_patterns': len(self.memory_data['successful_examples']),
            'pattern_type_counts': self.memory_data.get('pattern_counts', {}),
            'rft_facts_count': len(self.rft_facts),
            'indexed_patterns': len(self.pattern_index),
            'indexed_size_changes': len(self.size_pattern_index)
        }


# Global memory instance
_global_memory = None

def get_comprehensive_memory() -> ComprehensiveMemory:
    """Get the global comprehensive memory instance."""
    global _global_memory
    if _global_memory is None:
        _global_memory = ComprehensiveMemory()
    return _global_memory