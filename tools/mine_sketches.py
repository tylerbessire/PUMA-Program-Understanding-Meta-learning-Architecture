"""
Program sketch mining tool for ARC solver.

This script analyzes successful program solutions to extract common patterns
and operation sequences that can be used as macro-operations in future solving.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.features import extract_task_features
from arc_solver.sketches import SketchMiner, ProgramSketch
from arc_solver.heuristics import consistent_program_single_step


def load_training_data(challenges_path: str, solutions_path: str = None) -> List[Dict[str, Any]]:
    """Load ARC training challenges and solutions."""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    solutions = {}
    if solutions_path and Path(solutions_path).exists():
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
    
    tasks = []
    for task_id, task_data in challenges.items():
        task_info = {
            'task_id': task_id,
            'train': task_data['train'],
            'test': task_data['test']
        }
        if task_id in solutions:
            task_info['solutions'] = solutions[task_id]
        tasks.append(task_info)
    
    return tasks


def extract_successful_programs(tasks: List[Dict[str, Any]]) -> List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]]:
    """Extract programs that successfully solve training tasks."""
    successful_programs = []
    
    for task in tasks:
        # Convert train pairs to arrays
        train_pairs = []
        for pair in task['train']:
            inp = to_array(pair['input'])
            out = to_array(pair['output'])
            train_pairs.append((inp, out))
        
        if not train_pairs:
            continue
        
        # Try to find simple heuristic solutions
        heuristic_programs = consistent_program_single_step(train_pairs)
        
        for program in heuristic_programs:
            successful_programs.append((task['task_id'], program))
            
        # If we have known solutions, we could parse them here
        # For now, we rely on heuristic discovery
    
    return successful_programs


def analyze_operation_patterns(programs: List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]]) -> Dict[str, Any]:
    """Analyze patterns in successful programs."""
    analysis = {
        'total_programs': len(programs),
        'operation_frequency': Counter(),
        'sequence_frequency': Counter(),
        'program_lengths': [],
        'parameter_patterns': defaultdict(list),
    }
    
    for task_id, program in programs:
        # Record program length
        analysis['program_lengths'].append(len(program))
        
        # Record operation frequencies
        for op_name, params in program:
            analysis['operation_frequency'][op_name] += 1
            
            # Record parameter patterns for each operation
            analysis['parameter_patterns'][op_name].append(params)
        
        # Record operation sequences
        if len(program) > 1:
            sequence = tuple(op_name for op_name, _ in program)
            analysis['sequence_frequency'][sequence] += 1
    
    return analysis


def mine_program_sketches(programs: List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]], 
                         min_frequency: int = 2) -> List[ProgramSketch]:
    """Mine program sketches from successful programs."""
    sketch_miner = SketchMiner()
    
    # Add all successful programs to the miner
    for task_id, program in programs:
        sketch_miner.add_successful_program(program, task_id)
    
    # Mine sketches with minimum frequency threshold
    sketches = sketch_miner.mine_sketches(min_frequency)
    
    return sketches


def save_sketches(sketches: List[ProgramSketch], output_path: str):
    """Save mined sketches to JSON file."""
    sketch_data = []
    for sketch in sketches:
        sketch_data.append({
            'operations': sketch.operations,
            'param_constraints': sketch.param_constraints,
            'frequency': sketch.frequency,
            'success_rate': sketch.success_rate,
        })
    
    with open(output_path, 'w') as f:
        json.dump(sketch_data, f, indent=2)
    
    print(f"Saved {len(sketches)} sketches to {output_path}")


def print_analysis_summary(analysis: Dict[str, Any], sketches: List[ProgramSketch]):
    """Print analysis summary."""
    print(f"\n=== Program Sketch Mining Results ===")
    print(f"Total programs analyzed: {analysis['total_programs']}")
    print(f"Average program length: {np.mean(analysis['program_lengths']):.2f}")
    
    print(f"\nMost common operations:")
    for op, count in analysis['operation_frequency'].most_common(10):
        print(f"  {op}: {count} times")
    
    print(f"\nMost common operation sequences:")
    for seq, count in analysis['sequence_frequency'].most_common(10):
        seq_str = " -> ".join(seq)
        print(f"  {seq_str}: {count} times")
    
    print(f"\nMined sketches:")
    for i, sketch in enumerate(sketches[:10]):  # Show top 10
        ops_str = " -> ".join(sketch.operations)
        print(f"  {i+1}. {ops_str} (frequency: {sketch.frequency})")
    
    if len(sketches) > 10:
        print(f"  ... and {len(sketches) - 10} more sketches")


def main():
    parser = argparse.ArgumentParser(description="Mine program sketches from successful ARC solutions")
    parser.add_argument('--train_json', required=True, help='Path to training challenges JSON')
    parser.add_argument('--solutions_json', help='Path to training solutions JSON (optional)')
    parser.add_argument('--out', required=True, help='Output path for mined sketches JSON')
    parser.add_argument('--min_frequency', type=int, default=2, help='Minimum frequency for sketch mining')
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.train_json}")
    tasks = load_training_data(args.train_json, args.solutions_json)
    print(f"Loaded {len(tasks)} training tasks")
    
    # Extract successful programs using heuristics
    print("Extracting successful programs...")
    successful_programs = extract_successful_programs(tasks)
    print(f"Found {len(successful_programs)} successful programs")
    
    if len(successful_programs) == 0:
        print("No successful programs found to analyze!")
        return
    
    # Analyze patterns in successful programs
    print("Analyzing operation patterns...")
    analysis = analyze_operation_patterns(successful_programs)
    
    # Mine program sketches
    print(f"Mining program sketches (min frequency: {args.min_frequency})...")
    sketches = mine_program_sketches(successful_programs, args.min_frequency)
    
    # Print analysis summary
    print_analysis_summary(analysis, sketches)
    
    # Save sketches
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_sketches(sketches, args.out)
    
    print(f"\nSketch mining complete! Results saved to {args.out}")


if __name__ == "__main__":
    main()
