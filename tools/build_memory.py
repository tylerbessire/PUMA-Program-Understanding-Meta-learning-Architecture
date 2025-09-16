"""
Episodic memory building tool for ARC solver.

This script builds the episodic memory database by solving training tasks
and storing successful programs along with their task signatures for
future analogical retrieval.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.features import compute_task_signature
from arc_solver.neural.episodic import EpisodicRetrieval
from arc_solver.heuristics import consistent_program_single_step, score_candidate


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


def solve_task_and_store(task: Dict[str, Any], episodic_memory: EpisodicRetrieval) -> bool:
    """Attempt to solve a task and store successful programs in episodic memory."""
    task_id = task['task_id']
    
    # Convert train pairs to arrays
    train_pairs = []
    for pair in task['train']:
        inp = to_array(pair['input'])
        out = to_array(pair['output'])
        train_pairs.append((inp, out))
    
    if not train_pairs:
        return False
    
    # Try enhanced solver first, then fallback to heuristics
    successful_programs = []
    
    try:
        # Use enhanced solver for more comprehensive solutions
        from arc_solver.enhanced_search import synthesize_with_enhancements
        enhanced_programs = synthesize_with_enhancements(train_pairs)
        
        # Validate enhanced programs
        for program in enhanced_programs:
            score = score_candidate(program, train_pairs)
            if score > 0.99:  # Require perfect fit
                successful_programs.append(program)
                
    except Exception:
        # Fallback to heuristics if enhanced solver fails
        pass
    
    # If enhanced solver didn't find anything, try heuristics
    if not successful_programs:
        candidate_programs = consistent_program_single_step(train_pairs)
        
        # Filter for programs that actually work
        for program in candidate_programs:
            score = score_candidate(program, train_pairs)
            if score > 0.99:  # Require perfect fit
                successful_programs.append(program)
    
    # Store successful programs in episodic memory
    if successful_programs:
        episodic_memory.add_successful_solution(train_pairs, successful_programs, task_id)
        return True
    
    return False


def build_episodic_memory(tasks: List[Dict[str, Any]], 
                         db_path: str = "models/episodic_memory.json") -> EpisodicRetrieval:
    """Build episodic memory database from training tasks."""
    print(f"Building episodic memory database...")
    
    # Initialize episodic retrieval system
    episodic_memory = EpisodicRetrieval(db_path)
    
    solved_count = 0
    total_count = len(tasks)
    
    for i, task in enumerate(tasks):
        if i % 10 == 0:
            print(f"Processing task {i+1}/{total_count}...")
        
        try:
            success = solve_task_and_store(task, episodic_memory)
            if success:
                solved_count += 1
        except Exception as e:
            print(f"Error processing task {task['task_id']}: {e}")
    
    # Save the database
    episodic_memory.save()
    
    print(f"\nEpisodic memory building complete!")
    print(f"Successfully solved and stored: {solved_count}/{total_count} tasks")
    print(f"Success rate: {solved_count/total_count:.3f}")
    
    # Print database statistics
    stats = episodic_memory.get_stats()
    if stats:
        print(f"\nDatabase statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return episodic_memory


def analyze_memory_coverage(episodic_memory: EpisodicRetrieval, tasks: List[Dict[str, Any]]):
    """Analyze how well the episodic memory covers different task types."""
    print("\n=== Memory Coverage Analysis ===")
    
    # Collect task signatures
    signatures = {}
    for task in tasks:
        train_pairs = []
        for pair in task['train']:
            inp = to_array(pair['input'])
            out = to_array(pair['output'])
            train_pairs.append((inp, out))
        
        if train_pairs:
            signature = compute_task_signature(train_pairs)
            if signature not in signatures:
                signatures[signature] = []
            signatures[signature].append(task['task_id'])
    
    print(f"Total unique task signatures: {len(signatures)}")
    
    # Check coverage for each signature
    covered_signatures = 0
    for signature, task_ids in signatures.items():
        # Create a dummy task to test retrieval
        first_task = next(t for t in tasks if t['task_id'] == task_ids[0])
        train_pairs = []
        for pair in first_task['train']:
            inp = to_array(pair['input'])
            out = to_array(pair['output'])
            train_pairs.append((inp, out))
        
        candidates = episodic_memory.query_for_programs(train_pairs)
        if candidates:
            covered_signatures += 1
    
    coverage_rate = covered_signatures / len(signatures)
    print(f"Signatures with retrievable solutions: {covered_signatures}/{len(signatures)}")
    print(f"Coverage rate: {coverage_rate:.3f}")
    
    # Show most common signatures
    print(f"\nMost common task signatures:")
    sorted_sigs = sorted(signatures.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (sig, task_ids) in enumerate(sorted_sigs[:10]):
        print(f"  {i+1}. {sig} ({len(task_ids)} tasks)")


def main():
    parser = argparse.ArgumentParser(description="Build episodic memory database for ARC solver")
    parser.add_argument('--train_json', required=True, help='Path to training challenges JSON')
    parser.add_argument('--solutions_json', help='Path to training solutions JSON (optional)')
    parser.add_argument('--db_path', default='models/episodic_memory.json', 
                       help='Output path for episodic memory database')
    parser.add_argument('--analyze', action='store_true', 
                       help='Perform coverage analysis after building')
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.train_json}")
    tasks = load_training_data(args.train_json, args.solutions_json)
    print(f"Loaded {len(tasks)} training tasks")
    
    # Ensure output directory exists
    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build episodic memory
    start_time = time.time()
    episodic_memory = build_episodic_memory(tasks, args.db_path)
    build_time = time.time() - start_time
    
    print(f"\nMemory building took {build_time:.2f} seconds")
    print(f"Database saved to {args.db_path}")
    
    # Perform coverage analysis if requested
    if args.analyze:
        analyze_memory_coverage(episodic_memory, tasks)
    
    print(f"\nEpisodic memory building complete!")


if __name__ == "__main__":
    main()
