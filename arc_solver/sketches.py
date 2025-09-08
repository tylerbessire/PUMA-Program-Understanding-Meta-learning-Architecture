"""
Program sketch mining and macro-operation generation.

This module analyzes successful programs from training data to extract
common patterns and create macro-operations (program sketches) that can
be used to accelerate search on new tasks.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter, defaultdict
import json

from .grid import Array
from .dsl import apply_program


class ProgramSketch:
    """Represents a program template with parameterizable operations."""
    
    def __init__(self, operations: List[str], param_constraints: Dict[str, Any]):
        self.operations = operations
        self.param_constraints = param_constraints
        self.frequency = 1
        self.success_rate = 0.0
    
    def instantiate(self, params: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Create a concrete program from this sketch with given parameters."""
        program = []
        for op_name in self.operations:
            if op_name in params:
                program.append((op_name, params[op_name]))
            else:
                # Use default parameters based on constraints
                default_params = self._get_default_params(op_name)
                program.append((op_name, default_params))
        return program
    
    def _get_default_params(self, op_name: str) -> Dict[str, Any]:
        """Get default parameters for an operation."""
        defaults = {
            'rotate': {'k': 1},
            'flip': {'axis': 0},
            'transpose': {},
            'translate': {'dy': 0, 'dx': 0},
            'recolor': {'mapping': {}},
            'crop': {'top': 0, 'left': 0, 'height': 10, 'width': 10},
            'pad': {'out_h': 10, 'out_w': 10},
            'identity': {},
        }
        return defaults.get(op_name, {})


class SketchMiner:
    """Mines common program patterns from successful solutions."""
    
    def __init__(self):
        self.sketches: List[ProgramSketch] = []
        self.operation_sequences: List[List[str]] = []
        self.success_patterns: Dict[str, int] = Counter()
    
    def add_successful_program(self, program: List[Tuple[str, Dict[str, Any]]], 
                             task_signature: str = ""):
        """Add a successful program to the mining database."""
        operations = [op_name for op_name, _ in program]
        self.operation_sequences.append(operations)
        
        # Track successful patterns
        pattern = "->".join(operations)
        self.success_patterns[pattern] += 1
    
    def mine_sketches(self, min_frequency: int = 2) -> List[ProgramSketch]:
        """Extract common operation sequences as sketches."""
        sketches = []
        
        # Mine 1-operation sketches
        op_counts = Counter()
        for ops in self.operation_sequences:
            for op in ops:
                op_counts[op] += 1
        
        for op, count in op_counts.items():
            if count >= min_frequency:
                sketch = ProgramSketch([op], {})
                sketch.frequency = count
                sketches.append(sketch)
        
        # Mine 2-operation sketches
        pair_counts = Counter()
        for ops in self.operation_sequences:
            if len(ops) >= 2:
                for i in range(len(ops) - 1):
                    pair = (ops[i], ops[i + 1])
                    pair_counts[pair] += 1
        
        for (op1, op2), count in pair_counts.items():
            if count >= min_frequency:
                sketch = ProgramSketch([op1, op2], {})
                sketch.frequency = count
                sketches.append(sketch)
        
        # Sort by frequency
        sketches.sort(key=lambda s: s.frequency, reverse=True)
        self.sketches = sketches
        return sketches
    
    def get_relevant_sketches(self, predicted_operations: List[str], 
                            max_sketches: int = 10) -> List[ProgramSketch]:
        """Get sketches that are likely relevant for given predicted operations."""
        relevant = []
        
        for sketch in self.sketches:
            # Check if sketch operations overlap with predictions
            if any(op in predicted_operations for op in sketch.operations):
                relevant.append(sketch)
            
            if len(relevant) >= max_sketches:
                break
        
        return relevant
    
    def save_sketches(self, filepath: str):
        """Save mined sketches to file."""
        sketch_data = []
        for sketch in self.sketches:
            sketch_data.append({
                'operations': sketch.operations,
                'param_constraints': sketch.param_constraints,
                'frequency': sketch.frequency,
                'success_rate': sketch.success_rate,
            })
        
        with open(filepath, 'w') as f:
            json.dump(sketch_data, f, indent=2)
    
    def load_sketches(self, filepath: str):
        """Load sketches from file."""
        try:
            with open(filepath, 'r') as f:
                sketch_data = json.load(f)
            
            self.sketches = []
            for data in sketch_data:
                sketch = ProgramSketch(data['operations'], data['param_constraints'])
                sketch.frequency = data['frequency']
                sketch.success_rate = data['success_rate']
                self.sketches.append(sketch)
        except (FileNotFoundError, json.JSONDecodeError):
            self.sketches = []


class MacroOperations:
    """Defines macro-operations built from common sketches."""
    
    def __init__(self, sketch_miner: SketchMiner):
        self.sketch_miner = sketch_miner
        self.macros = self._build_macros()
    
    def _build_macros(self) -> Dict[str, ProgramSketch]:
        """Build macro-operations from common sketches."""
        macros = {}
        
        # Define some common macro patterns
        macros['rotate_and_flip'] = ProgramSketch(
            ['rotate', 'flip'], 
            {'rotate': {'k': [1, 2, 3]}, 'flip': {'axis': [0, 1]}}
        )
        
        macros['translate_and_recolor'] = ProgramSketch(
            ['translate', 'recolor'],
            {'translate': {'dy': range(-2, 3), 'dx': range(-2, 3)}}
        )
        
        macros['crop_and_pad'] = ProgramSketch(
            ['crop', 'pad'],
            {}
        )
        
        return macros
    
    def get_macro_operations(self) -> List[str]:
        """Get list of available macro operation names."""
        return list(self.macros.keys())
    
    def expand_macro(self, macro_name: str, params: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Expand a macro into its constituent operations."""
        if macro_name not in self.macros:
            return []
        
        return self.macros[macro_name].instantiate(params)


def generate_parameter_grid(operation: str, constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Generate parameter combinations for a given operation."""
    if constraints is None:
        constraints = {}
    
    if operation == 'rotate':
        return [{'k': k} for k in constraints.get('k', [0, 1, 2, 3])]
    elif operation == 'flip':
        return [{'axis': axis} for axis in constraints.get('axis', [0, 1])]
    elif operation == 'transpose':
        return [{}]
    elif operation == 'translate':
        dy_range = constraints.get('dy', range(-2, 3))
        dx_range = constraints.get('dx', range(-2, 3))
        return [{'dy': dy, 'dx': dx} for dy in dy_range for dx in dx_range]
    elif operation == 'identity':
        return [{}]
    else:
        return [{}]  # Default empty params for other operations
