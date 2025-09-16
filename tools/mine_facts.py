#!/usr/bin/env python3
"""
Facts mining tool for ARC solver.

This script analyzes training data to extract facts about grid patterns,
transformations, and task structure for use in program synthesis.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter, defaultdict
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.features import extract_task_features


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


def extract_grid_facts(grid: np.ndarray) -> Dict[str, Any]:
    """Extract facts from a grid."""
    facts = {}
    
    # Basic properties
    facts['height'] = int(grid.shape[0])
    facts['width'] = int(grid.shape[1])
    facts['area'] = int(grid.size)
    
    # Color properties
    unique_colors = np.unique(grid)
    facts['num_colors'] = len(unique_colors)
    facts['colors'] = [int(c) for c in unique_colors]
    facts['background_color'] = int(unique_colors[0]) if len(unique_colors) > 0 else 0
    
    # Color frequencies
    color_counts = Counter(grid.flatten())
    facts['color_frequencies'] = {int(k): int(v) for k, v in color_counts.items()}
    
    # Pattern analysis
    facts['is_square'] = facts['height'] == facts['width']
    facts['is_single_color'] = len(unique_colors) == 1
    
    # Simple shape detection
    non_bg = grid != facts['background_color']
    facts['num_foreground_pixels'] = int(np.sum(non_bg))
    
    if facts['num_foreground_pixels'] > 0:
        # Bounding box of non-background pixels
        rows, cols = np.where(non_bg)
        facts['bbox'] = {
            'min_row': int(np.min(rows)),
            'max_row': int(np.max(rows)),
            'min_col': int(np.min(cols)),
            'max_col': int(np.max(cols))
        }
        facts['bbox_width'] = facts['bbox']['max_col'] - facts['bbox']['min_col'] + 1
        facts['bbox_height'] = facts['bbox']['max_row'] - facts['bbox']['min_row'] + 1
    
    return facts


def extract_pair_facts(inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
    """Extract facts from an input-output pair."""
    inp_facts = extract_grid_facts(inp)
    out_facts = extract_grid_facts(out)
    
    pair_facts = {
        'input': inp_facts,
        'output': out_facts,
        'transformation': {}
    }
    
    # Transformation facts
    trans = pair_facts['transformation']
    trans['size_changed'] = (inp.shape != out.shape)
    trans['colors_changed'] = (set(inp_facts['colors']) != set(out_facts['colors']))
    trans['same_dimensions'] = (inp.shape == out.shape)
    
    if trans['same_dimensions']:
        trans['pixel_changes'] = int(np.sum(inp != out))
        trans['pixel_change_ratio'] = trans['pixel_changes'] / inp.size
        trans['exact_copy'] = (trans['pixel_changes'] == 0)
    
    # Color mapping analysis
    if trans['same_dimensions'] and trans['colors_changed']:
        color_map = {}
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                old_color = int(inp[i, j])
                new_color = int(out[i, j])
                if old_color in color_map:
                    if color_map[old_color] != new_color:
                        color_map[old_color] = None  # Inconsistent mapping
                else:
                    color_map[old_color] = new_color
        
        # Clean up inconsistent mappings
        consistent_mappings = {k: v for k, v in color_map.items() if v is not None}
        trans['color_mappings'] = consistent_mappings
        trans['has_consistent_color_mapping'] = len(consistent_mappings) > 0
    
    return pair_facts


def extract_task_facts(task: Dict[str, Any]) -> Dict[str, Any]:
    """Extract facts from a complete task."""
    task_id = task['task_id']
    train_pairs = task['train']
    test_pairs = task['test']
    
    facts = {
        'task_id': task_id,
        'num_train_pairs': len(train_pairs),
        'num_test_pairs': len(test_pairs),
        'train_facts': [],
        'test_input_facts': []
    }
    
    # Extract facts from training pairs
    for i, pair in enumerate(train_pairs):
        inp = to_array(pair['input'])
        out = to_array(pair['output'])
        pair_facts = extract_pair_facts(inp, out)
        pair_facts['pair_id'] = i
        facts['train_facts'].append(pair_facts)
    
    # Extract facts from test inputs (no outputs known)
    for i, pair in enumerate(test_pairs):
        inp = to_array(pair['input'])
        inp_facts = extract_grid_facts(inp)
        inp_facts['pair_id'] = i
        facts['test_input_facts'].append(inp_facts)
    
    # Task-level aggregated facts
    if facts['train_facts']:
        # Check consistency across training pairs
        first_pair = facts['train_facts'][0]
        facts['consistent_input_size'] = all(
            p['input']['height'] == first_pair['input']['height'] and 
            p['input']['width'] == first_pair['input']['width']
            for p in facts['train_facts']
        )
        facts['consistent_output_size'] = all(
            p['output']['height'] == first_pair['output']['height'] and 
            p['output']['width'] == first_pair['output']['width']
            for p in facts['train_facts']
        )
        facts['consistent_transformation'] = all(
            p['transformation']['size_changed'] == first_pair['transformation']['size_changed']
            for p in facts['train_facts']
        )
    
    return facts


def mine_facts(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mine facts from all tasks."""
    all_facts = []
    
    print(f"Mining facts from {len(tasks)} tasks...")
    
    for i, task in enumerate(tasks):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing task {i+1}/{len(tasks)}: {task['task_id']}")
        
        try:
            task_facts = extract_task_facts(task)
            all_facts.append(task_facts)
        except Exception as e:
            print(f"Error processing task {task['task_id']}: {e}")
            continue
    
    print(f"Successfully extracted facts from {len(all_facts)} tasks")
    return all_facts


def analyze_facts_coverage(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze coverage of extracted facts."""
    total_tasks = len(facts)
    
    coverage = {
        'total_tasks': total_tasks,
        'tasks_with_facts': sum(1 for f in facts if f.get('train_facts')),
        'average_train_pairs': np.mean([f.get('num_train_pairs', 0) for f in facts]),
        'average_test_pairs': np.mean([f.get('num_test_pairs', 0) for f in facts])
    }
    
    # Count different transformation types
    size_changes = sum(1 for f in facts if f.get('train_facts') and any(
        p.get('transformation', {}).get('size_changed', False) for p in f['train_facts']
    ))
    color_changes = sum(1 for f in facts if f.get('train_facts') and any(
        p.get('transformation', {}).get('colors_changed', False) for p in f['train_facts']
    ))
    
    coverage['size_change_tasks'] = size_changes
    coverage['color_change_tasks'] = color_changes
    coverage['coverage_percentage'] = (coverage['tasks_with_facts'] / total_tasks) * 100
    
    return coverage


def main():
    parser = argparse.ArgumentParser(description="Mine facts from ARC training data")
    parser.add_argument("--train_json", required=True, help="Path to training challenges JSON")
    parser.add_argument("--solutions_json", help="Path to training solutions JSON")
    parser.add_argument("--out", default="facts.jsonl", help="Output facts file (JSONL format)")
    parser.add_argument("--coverage_out", default="facts_coverage.json", help="Output coverage report")
    
    args = parser.parse_args()
    
    print("🔍 ARC Facts Mining Tool")
    print("=" * 50)
    
    # Load training data
    print(f"Loading training data from {args.train_json}")
    tasks = load_training_data(args.train_json, args.solutions_json)
    print(f"Loaded {len(tasks)} tasks")
    
    # Mine facts
    start_time = time.time()
    all_facts = mine_facts(tasks)
    elapsed = time.time() - start_time
    
    # Analyze coverage
    coverage = analyze_facts_coverage(all_facts)
    
    # Save facts as JSONL
    print(f"\nSaving facts to {args.out}")
    with open(args.out, 'w') as f:
        for fact in all_facts:
            f.write(json.dumps(fact) + '\n')
    
    # Save coverage report
    print(f"Saving coverage report to {args.coverage_out}")
    with open(args.coverage_out, 'w') as f:
        json.dump(coverage, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"🎉 Facts mining complete!")
    print(f"Mining time: {elapsed:.1f} seconds")
    print(f"Facts extracted from: {coverage['tasks_with_facts']}/{coverage['total_tasks']} tasks")
    print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
    print(f"Size-changing tasks: {coverage['size_change_tasks']}")
    print(f"Color-changing tasks: {coverage['color_change_tasks']}")
    print(f"Output saved to: {args.out}")


if __name__ == "__main__":
    main()