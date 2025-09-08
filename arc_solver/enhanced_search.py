"""
Enhanced search module integrating neural guidance, episodic retrieval, and TTT.

This module extends the basic search with neural-guided program synthesis,
episodic retrieval of similar solutions, program sketch mining, and test-time
adaptation for better performance on ARC tasks.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy

from .grid import Array, eq
from .dsl import OPS, apply_program
from .heuristics import consistent_program_single_step, score_candidate, diversify_programs
from .guidance import NeuralGuidance
from .memory import EpisodicRetrieval
from .sketches import SketchMiner, generate_parameter_grid
from .ttt import TestTimeTrainer, DataAugmentation


class EnhancedSearch:
    """Enhanced program synthesis search with neural guidance and episodic retrieval."""
    
    def __init__(self, guidance_model_path: Optional[str] = None, 
                 episode_db_path: str = "episodes.json"):
        self.neural_guidance = NeuralGuidance(guidance_model_path)
        self.episodic_retrieval = EpisodicRetrieval(episode_db_path)
        self.sketch_miner = SketchMiner()
        self.test_time_trainer = TestTimeTrainer()
        self.search_stats = {}
        
        # Load any existing sketches
        try:
            self.sketch_miner.load_sketches("sketches.json")
        except:
            pass
    
    def synthesize_enhanced(self, train_pairs: List[Tuple[Array, Array]], 
                           max_programs: int = 256) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Enhanced program synthesis using all available techniques."""
        self.search_stats = {
            'episodic_candidates': 0,
            'heuristic_candidates': 0,
            'sketch_candidates': 0,
            'neural_guided_candidates': 0,
            'ttt_adapted': False,
        }
        
        all_candidates = []
        
        # Step 1: Try episodic retrieval first (fastest)
        episodic_candidates = self._get_episodic_candidates(train_pairs)
        all_candidates.extend(episodic_candidates)
        self.search_stats['episodic_candidates'] = len(episodic_candidates)
        
        # Step 2: Try heuristic single-step programs
        heuristic_candidates = consistent_program_single_step(train_pairs)
        all_candidates.extend(heuristic_candidates)
        self.search_stats['heuristic_candidates'] = len(heuristic_candidates)
        
        # Step 3: Neural-guided search if we need more candidates
        if len(all_candidates) < max_programs // 4:
            neural_candidates = self._neural_guided_search(train_pairs, max_programs // 2)
            all_candidates.extend(neural_candidates)
            self.search_stats['neural_guided_candidates'] = len(neural_candidates)
        
        # Step 4: Sketch-based search if still need more
        if len(all_candidates) < max_programs // 2:
            sketch_candidates = self._sketch_based_search(train_pairs, max_programs // 3)
            all_candidates.extend(sketch_candidates)
            self.search_stats['sketch_candidates'] = len(sketch_candidates)
        
        # Step 5: Test-time adaptation if we have candidates
        if all_candidates:
            all_candidates = self._apply_test_time_adaptation(train_pairs, all_candidates)
            self.search_stats['ttt_adapted'] = True
        
        # Step 6: Score, deduplicate, and select best programs
        final_programs = self._select_best_programs(train_pairs, all_candidates, max_programs)
        
        # Update episodic memory with any successful programs
        successful_programs = [p for p in final_programs if score_candidate(p, train_pairs) > 0.99]
        if successful_programs:
            self.episodic_retrieval.add_successful_solution(train_pairs, successful_programs)
            # Also update sketch miner
            for program in successful_programs:
                self.sketch_miner.add_successful_program(program)
        
        return final_programs
    
    def _get_episodic_candidates(self, train_pairs: List[Tuple[Array, Array]]) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Get candidate programs from episodic retrieval."""
        return self.episodic_retrieval.query_for_programs(train_pairs, max_candidates=10)
    
    def _neural_guided_search(self, train_pairs: List[Tuple[Array, Array]], 
                             max_candidates: int) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Perform neural-guided program search."""
        # Get operation predictions from neural guidance
        predicted_ops = self.neural_guidance.predict_operations(train_pairs)
        operation_scores = self.neural_guidance.score_operations(train_pairs)
        
        candidates = []
        
        # Generate 1-step programs with predicted operations
        for op_name in predicted_ops:
            if op_name == 'identity':
                candidates.append([('identity', {})])
                continue
                
            param_grids = generate_parameter_grid(op_name)
            for params in param_grids:
                program = [(op_name, params)]
                if score_candidate(program, train_pairs) > 0.99:
                    candidates.append(program)
                if len(candidates) >= max_candidates // 2:
                    break
        
        # Generate 2-step programs with high-scoring operations
        high_scoring_ops = [op for op, score in operation_scores.items() if score > 0.3]
        
        for op1 in high_scoring_ops:
            for op2 in high_scoring_ops:
                if len(candidates) >= max_candidates:
                    break
                    
                param_grid1 = generate_parameter_grid(op1)
                param_grid2 = generate_parameter_grid(op2)
                
                # Sample a few parameter combinations
                for params1 in param_grid1[:3]:  # Limit combinations
                    for params2 in param_grid2[:3]:
                        program = [(op1, params1), (op2, params2)]
                        if score_candidate(program, train_pairs) > 0.99:
                            candidates.append(program)
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates
    
    def _sketch_based_search(self, train_pairs: List[Tuple[Array, Array]], 
                           max_candidates: int) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Search using program sketches."""
        predicted_ops = self.neural_guidance.predict_operations(train_pairs)
        relevant_sketches = self.sketch_miner.get_relevant_sketches(predicted_ops)
        
        candidates = []
        
        for sketch in relevant_sketches:
            if len(candidates) >= max_candidates:
                break
                
            # Try different parameter instantiations for this sketch
            for attempt in range(min(10, max_candidates - len(candidates))):
                try:
                    # Generate parameters for each operation in the sketch
                    sketch_params = {}
                    for op_name in sketch.operations:
                        param_grid = generate_parameter_grid(op_name)
                        if param_grid:
                            # Pick a random parameter set
                            idx = np.random.randint(len(param_grid))
                            sketch_params[op_name] = param_grid[idx]
                    
                    program = sketch.instantiate(sketch_params)
                    if score_candidate(program, train_pairs) > 0.99:
                        candidates.append(program)
                
                except Exception:
                    continue
        
        return candidates
    
    def _apply_test_time_adaptation(self, train_pairs: List[Tuple[Array, Array]], 
                                  candidates: List[List[Tuple[str, Dict[str, int]]]]) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Apply test-time training to improve candidate ranking."""
        if len(candidates) < 2:
            return candidates
        
        # Augment training data for better adaptation
        augmented_pairs = DataAugmentation.augment_training_pairs(train_pairs, max_augmentations=20)
        
        # Adapt the scorer to this specific task
        self.test_time_trainer.adapt_to_task(augmented_pairs, candidates)
        
        # Re-score candidates with adapted scorer
        candidate_scores = []
        for program in candidates:
            adapted_score = self.test_time_trainer.score_with_adaptation(program, train_pairs)
            base_score = score_candidate(program, train_pairs)
            # Combine adapted score with base performance
            combined_score = 0.7 * base_score + 0.3 * adapted_score
            candidate_scores.append((combined_score, program))
        
        # Sort by combined score
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [program for _, program in candidate_scores]
    
    def _select_best_programs(self, train_pairs: List[Tuple[Array, Array]], 
                            candidates: List[List[Tuple[str, Dict[str, int]]]], 
                            max_programs: int) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Select and rank the best candidate programs."""
        # Score all candidates
        scored_candidates = []
        for program in candidates:
            score = score_candidate(program, train_pairs)
            scored_candidates.append((score, program))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take only high-scoring programs
        good_programs = [program for score, program in scored_candidates if score > 0.99]
        
        # If no perfect programs, take the best available
        if not good_programs and scored_candidates:
            good_programs = [program for score, program in scored_candidates[:max_programs]]
        
        # Diversify the program set
        final_programs = diversify_programs(good_programs)
        
        return final_programs[:max_programs]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the last search."""
        return self.search_stats.copy()
    
    def save_components(self):
        """Save all learned components."""
        self.episodic_retrieval.save()
        self.sketch_miner.save_sketches("sketches.json")


def predict_two_enhanced(progs: List[List[Tuple[str, Dict[str, int]]]], 
                        test_inputs: List[Array]) -> List[List[Array]]:
    """Enhanced prediction with better fallback strategies."""
    if len(progs) == 0:
        # No programs found, use identity
        picks = [[("identity", {})], [("identity", {})]]
    elif len(progs) == 1:
        # Only one program, use it twice with slight variation if possible
        main_prog = progs[0]
        picks = [main_prog, main_prog]
    else:
        # Use top 2 programs
        picks = progs[:2]
    
    attempts: List[List[Array]] = []
    for program in picks:
        outs: List[Array] = []
        for ti in test_inputs:
            try:
                result = apply_program(ti, program)
                outs.append(result)
            except Exception:
                # Fallback to identity on failure
                outs.append(ti)
        attempts.append(outs)
    
    return attempts


# Integration function to use enhanced search in the main solver
def synthesize_with_enhancements(train_pairs: List[Tuple[Array, Array]], 
                               max_programs: int = 256) -> List[List[Tuple[str, Dict[str, int]]]]:
    """Main function to synthesize programs with all enhancements."""
    
    # Initialize enhanced search (this will be cached across calls in practice)
    enhanced_search = EnhancedSearch()
    
    # Try enhanced synthesis
    programs = enhanced_search.synthesize_enhanced(train_pairs, max_programs)
    
    # Save learned components periodically
    enhanced_search.save_components()
    
    return programs
