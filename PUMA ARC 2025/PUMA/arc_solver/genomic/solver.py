"""
Genomic solver implementation.

This module implements the main genomic solver that converts grids to sequences,
performs alignment analysis, and extracts transformation rules.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from ..grid import Array, to_array
from ..common.eval_utils import exact_match
from .hilbert import grid_to_hilbert_sequence, hilbert_order
from .tokenize import tokenize_sequence, run_length_encode
from .align import needleman_wunsch, smith_waterman, AlignmentScorer
from .script import infer_script, consensus_script, apply_recipe, MutationScript


class GenomicSolver:
    """
    Main genomic solver class that coordinates sequence analysis and transformation.
    """
    
    def __init__(self, alignment_method: str = "needleman_wunsch",
                 use_run_length: bool = True,
                 scorer_params: Optional[Dict[str, float]] = None):
        self.alignment_method = alignment_method
        self.use_run_length = use_run_length
        
        # Initialize alignment scorer
        if scorer_params is None:
            scorer_params = {
                'match_score': 2.0,
                'mismatch_penalty': -1.0,
                'gap_penalty': -2.0,
                'run_bonus': 1.0
            }
        self.scorer = AlignmentScorer(**scorer_params)
        
        # Statistics tracking
        self.stats = {
            'tasks_processed': 0,
            'successful_tasks': 0,
            'avg_confidence': 0.0,
            'alignment_scores': []
        }
    
    def solve_task(self, task: Dict[str, Any]) -> List[Array]:
        """
        Solve an ARC task using genomic analysis.
        
        Args:
            task: ARC task dictionary
        
        Returns:
            List of predicted output grids for test cases
        """
        self.stats['tasks_processed'] += 1
        
        # Extract training pairs
        train_pairs = self._extract_train_pairs(task)
        
        if not train_pairs:
            return self._fallback_predictions(task)
        
        # Infer scripts for each training pair
        scripts = []
        for input_grid, output_grid in train_pairs:
            try:
                script = infer_script(input_grid, output_grid, self.alignment_method)
                scripts.append(script)
                self.stats['alignment_scores'].append(script.metadata.get('alignment_score', 0))
            except Exception:
                # Skip problematic pairs
                continue
        
        if not scripts:
            return self._fallback_predictions(task)
        
        # Create consensus script
        consensus = consensus_script(scripts)
        
        # Apply to test cases
        predictions = []
        test_cases = task.get('test', [])
        
        for test_case in test_cases:
            try:
                test_input = to_array(test_case['input'])
                prediction = apply_recipe(consensus, test_input)
                predictions.append(prediction)
            except Exception:
                predictions.append(test_input)  # Fallback to identity
        
        # Validate predictions on training data
        if self._validate_consensus(consensus, train_pairs):
            self.stats['successful_tasks'] += 1
        
        # Update confidence statistics
        if scripts:
            avg_conf = np.mean([s.confidence for s in scripts])
            self.stats['avg_confidence'] = (
                (self.stats['avg_confidence'] * (self.stats['tasks_processed'] - 1) + avg_conf) /
                self.stats['tasks_processed']
            )
        
        return predictions
    
    def _extract_train_pairs(self, task: Dict[str, Any]) -> List[Tuple[Array, Array]]:
        """Extract and validate training pairs from task."""
        pairs = []
        
        for pair_data in task.get('train', []):
            try:
                input_grid = to_array(pair_data['input'])
                output_grid = to_array(pair_data['output'])
                
                # Basic validation
                if input_grid.size == 0 or output_grid.size == 0:
                    continue
                
                pairs.append((input_grid, output_grid))
            except Exception:
                continue
        
        return pairs
    
    def _fallback_predictions(self, task: Dict[str, Any]) -> List[Array]:
        """Generate fallback predictions when genomic analysis fails."""
        predictions = []
        
        for test_case in task.get('test', []):
            try:
                test_input = to_array(test_case['input'])
                predictions.append(test_input)  # Identity transformation
            except Exception:
                predictions.append(np.array([[0]]))  # Minimal fallback
        
        return predictions
    
    def _validate_consensus(self, consensus: MutationScript, 
                           train_pairs: List[Tuple[Array, Array]]) -> bool:
        """Validate consensus script on training pairs."""
        if not train_pairs:
            return False
        
        correct = 0
        for input_grid, expected_output in train_pairs:
            try:
                predicted_output = apply_recipe(consensus, input_grid)
                if exact_match(predicted_output, expected_output):
                    correct += 1
            except Exception:
                continue
        
        accuracy = correct / len(train_pairs)
        return accuracy > 0.5  # Require majority accuracy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver performance statistics."""
        stats = self.stats.copy()
        
        if stats['tasks_processed'] > 0:
            stats['success_rate'] = stats['successful_tasks'] / stats['tasks_processed']
        else:
            stats['success_rate'] = 0.0
        
        if stats['alignment_scores']:
            stats['avg_alignment_score'] = np.mean(stats['alignment_scores'])
            stats['median_alignment_score'] = np.median(stats['alignment_scores'])
        else:
            stats['avg_alignment_score'] = 0.0
            stats['median_alignment_score'] = 0.0
        
        return stats
    
    def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform detailed analysis of a task without solving it.
        Useful for debugging and understanding task characteristics.
        """
        analysis = {
            'train_pairs': len(task.get('train', [])),
            'test_cases': len(task.get('test', [])),
            'scripts': [],
            'consensus': None,
            'patterns': {}
        }
        
        train_pairs = self._extract_train_pairs(task)
        analysis['valid_train_pairs'] = len(train_pairs)
        
        if not train_pairs:
            return analysis
        
        # Analyze each training pair
        for i, (input_grid, output_grid) in enumerate(train_pairs):
            try:
                script = infer_script(input_grid, output_grid, self.alignment_method)
                analysis['scripts'].append({
                    'pair_index': i,
                    'confidence': script.confidence,
                    'mutations': len(script.mutations),
                    'metadata': script.metadata,
                    'description': self._describe_script(script)
                })
            except Exception as e:
                analysis['scripts'].append({
                    'pair_index': i,
                    'error': str(e)
                })
        
        # Create consensus if possible
        valid_scripts = [MutationScript(s['mutations'] if 'mutations' in s else [], 
                                      s.get('confidence', 0), 
                                      s.get('metadata', {}))
                        for s in analysis['scripts'] 
                        if 'mutations' in s]
        
        if valid_scripts:
            try:
                consensus = consensus_script(valid_scripts)
                analysis['consensus'] = {
                    'mutations': len(consensus.mutations),
                    'confidence': consensus.confidence,
                    'metadata': consensus.metadata,
                    'description': self._describe_script(consensus)
                }
            except Exception as e:
                analysis['consensus'] = {'error': str(e)}
        
        return analysis
    
    def _describe_script(self, script: MutationScript) -> str:
        """Generate human-readable description of a script."""
        from .script import script_to_description
        return script_to_description(script)


def solve_task_genomic(task: Dict[str, Any], **kwargs) -> List[Array]:
    """
    Standalone function to solve a task using genomic analysis.
    
    Args:
        task: ARC task dictionary
        **kwargs: Additional parameters for GenomicSolver
    
    Returns:
        List of predicted output grids
    """
    solver = GenomicSolver(**kwargs)
    return solver.solve_task(task)


def solve_task_genomic_dict(task: Dict[str, Any], **kwargs) -> Dict[str, List[List[List[int]]]]:
    """
    Solve an ARC task and return predictions in standard format.
    This matches the interface expected by the solver registry.
    
    Args:
        task: ARC task dictionary
        **kwargs: Additional parameters for GenomicSolver
    
    Returns:
        Dictionary with attempt_1 and attempt_2 predictions
    """
    predictions = solve_task_genomic(task, **kwargs)
    
    # Convert to list format
    pred_lists = []
    for pred in predictions:
        if isinstance(pred, np.ndarray):
            pred_lists.append(pred.astype(int).tolist())
        else:
            pred_lists.append([[0]])  # Fallback
    
    # For now, return the same predictions for both attempts
    # Could be enhanced to generate diverse predictions
    return {
        "attempt_1": pred_lists,
        "attempt_2": pred_lists
    }


def analyze_genomic_patterns(dataset: List[Dict[str, Any]], 
                           sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze genomic patterns across a dataset of tasks.
    
    Args:
        dataset: List of ARC tasks
        sample_size: Optional limit on number of tasks to analyze
    
    Returns:
        Analysis results across the dataset
    """
    if sample_size and len(dataset) > sample_size:
        import random
        dataset = random.sample(dataset, sample_size)
    
    solver = GenomicSolver()
    
    analyses = []
    for task in dataset:
        try:
            analysis = solver.analyze_task(task)
            analyses.append(analysis)
        except Exception:
            continue
    
    # Aggregate statistics
    total_tasks = len(analyses)
    valid_consensuses = len([a for a in analyses if a.get('consensus') and 'error' not in a['consensus']])
    
    confidence_scores = []
    mutation_counts = []
    
    for analysis in analyses:
        if analysis.get('consensus') and 'confidence' in analysis['consensus']:
            confidence_scores.append(analysis['consensus']['confidence'])
        
        for script_info in analysis.get('scripts', []):
            if 'mutations' in script_info:
                mutation_counts.append(script_info['mutations'])
    
    return {
        'total_tasks_analyzed': total_tasks,
        'tasks_with_valid_consensus': valid_consensuses,
        'consensus_success_rate': valid_consensuses / max(1, total_tasks),
        'avg_consensus_confidence': np.mean(confidence_scores) if confidence_scores else 0,
        'avg_mutations_per_script': np.mean(mutation_counts) if mutation_counts else 0,
        'solver_stats': solver.get_statistics()
    }