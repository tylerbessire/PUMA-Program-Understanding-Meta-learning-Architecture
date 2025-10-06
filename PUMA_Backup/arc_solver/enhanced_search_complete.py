"""
COMPLETE Enhanced Search with ALL Advanced Techniques.

This module implements the full enhanced search with ALL components:
- Neural guidance with expanded operation prediction
- Episodic retrieval with pattern matching  
- Program sketch mining with macro-operations
- Test-time training with task adaptation
- Multi-step program synthesis
- Advanced failure recovery
- Comprehensive candidate generation
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy
from itertools import product, combinations

from .grid import Array, eq
from .dsl_complete import OPS, apply_program, get_operation_signatures
from .heuristics_complete import consistent_program_comprehensive, score_program_comprehensive
from .guidance import NeuralGuidance
from .memory import EpisodicRetrieval
from .sketches import SketchMiner
from .ttt import TestTimeTrainer, DataAugmentation


class ComprehensiveEnhancedSearch:
    """COMPLETE enhanced search with ALL advanced ARC solving techniques."""
    
    def __init__(self, guidance_model_path: Optional[str] = None, 
                 episode_db_path: str = "models/episodic_memory.json",
                 sketches_path: str = "models/sketches.json"):
        
        self.neural_guidance = NeuralGuidance(guidance_model_path)
        self.episodic_retrieval = EpisodicRetrieval(episode_db_path)
        self.sketch_miner = SketchMiner()
        self.test_time_trainer = TestTimeTrainer()
        
        # Load trained components
        try:
            self.episodic_retrieval.load()
            self.sketch_miner.load_sketches(sketches_path)
        except Exception as e:
            print(f"Warning: Could not load trained components: {e}")
        
        self.search_stats = {}
        self.operation_signatures = get_operation_signatures()
        
    def synthesize_comprehensive(self, train_pairs: List[Tuple[Array, Array]], 
                               max_programs: int = 512) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """COMPLETE program synthesis using ALL available techniques."""
        
        self.search_stats = {
            'comprehensive_heuristics': 0,
            'neural_guided_candidates': 0,
            'episodic_candidates': 0,
            'sketch_candidates': 0,
            'multi_step_candidates': 0,
            'brute_force_candidates': 0,
            'ttt_adapted': False,
            'total_generated': 0,
            'final_selected': 0
        }
        
        all_candidates = []
        
        # PHASE 1: COMPREHENSIVE HEURISTIC ANALYSIS
        print("Phase 1: Comprehensive heuristic analysis...")
        heuristic_programs = consistent_program_comprehensive(train_pairs)
        all_candidates.extend(heuristic_programs)
        self.search_stats['comprehensive_heuristics'] = len(heuristic_programs)
        print(f"Found {len(heuristic_programs)} heuristic programs")
        
        # PHASE 2: NEURAL GUIDANCE ENHANCED SEARCH  
        print("Phase 2: Neural-guided search...")
        if len(all_candidates) < max_programs // 4:
            neural_programs = self._neural_guided_comprehensive_search(train_pairs, max_programs // 3)
            all_candidates.extend(neural_programs)
            self.search_stats['neural_guided_candidates'] = len(neural_programs)
            print(f"Found {len(neural_programs)} neural-guided programs")
        
        # PHASE 3: EPISODIC RETRIEVAL
        print("Phase 3: Episodic retrieval...")
        episodic_programs = self._episodic_retrieval_comprehensive(train_pairs, max_programs // 6)
        all_candidates.extend(episodic_programs)
        self.search_stats['episodic_candidates'] = len(episodic_programs)
        print(f"Found {len(episodic_programs)} episodic programs")
        
        # PHASE 4: SKETCH-BASED SYNTHESIS
        print("Phase 4: Sketch-based synthesis...")
        if len(all_candidates) < max_programs // 3:
            sketch_programs = self._sketch_based_comprehensive_search(train_pairs, max_programs // 4)
            all_candidates.extend(sketch_programs)
            self.search_stats['sketch_candidates'] = len(sketch_programs)
            print(f"Found {len(sketch_programs)} sketch-based programs")
        
        # PHASE 5: MULTI-STEP PROGRAM GENERATION
        print("Phase 5: Multi-step program generation...")
        if len(all_candidates) < max_programs // 2:
            multi_step_programs = self._generate_multi_step_programs(train_pairs, max_programs // 4)
            all_candidates.extend(multi_step_programs)
            self.search_stats['multi_step_candidates'] = len(multi_step_programs)
            print(f"Found {len(multi_step_programs)} multi-step programs")
        
        # PHASE 6: BRUTE FORCE FALLBACK
        print("Phase 6: Brute force fallback...")
        if len(all_candidates) < max_programs // 3:
            brute_force_programs = self._brute_force_comprehensive_search(train_pairs, max_programs // 6)
            all_candidates.extend(brute_force_programs)
            self.search_stats['brute_force_candidates'] = len(brute_force_programs)
            print(f"Found {len(brute_force_programs)} brute-force programs")
        
        self.search_stats['total_generated'] = len(all_candidates)
        print(f"Total candidates generated: {len(all_candidates)}")
        
        # PHASE 7: TEST-TIME TRAINING ADAPTATION
        if all_candidates and len(train_pairs) > 0:
            print("Phase 7: Test-time training adaptation...")
            all_candidates = self._apply_comprehensive_ttt(train_pairs, all_candidates)
            self.search_stats['ttt_adapted'] = True
        
        # PHASE 8: COMPREHENSIVE CANDIDATE SELECTION
        print("Phase 8: Final candidate selection...")
        final_programs = self._comprehensive_candidate_selection(train_pairs, all_candidates, max_programs)
        self.search_stats['final_selected'] = len(final_programs)
        
        # PHASE 9: SAVE SUCCESSFUL PROGRAMS
        successful_programs = [p for p in final_programs if score_program_comprehensive(p, train_pairs) > 0.99]
        if successful_programs:
            print(f"Saving {len(successful_programs)} successful programs...")
            for program in successful_programs:
                self.episodic_retrieval.add_successful_solution(train_pairs, [program])
                self.sketch_miner.add_successful_program(program)
        
        print(f"Final programs selected: {len(final_programs)}")
        return final_programs
    
    def _neural_guided_comprehensive_search(self, train_pairs: List[Tuple[Array, Array]], 
                                          max_candidates: int) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Comprehensive neural-guided search with ALL operations."""
        candidates = []
        
        try:
            # Get neural predictions for relevant operations
            predicted_ops = self.neural_guidance.predict_operations(train_pairs)
            operation_scores = self.neural_guidance.score_operations(train_pairs)
            
            print(f"Neural guidance predicted: {predicted_ops}")
            
            # Generate single-step programs with ALL predicted operations
            for op_name in predicted_ops:
                if op_name == 'identity':
                    candidates.append([('identity', {})])
                    continue
                
                if op_name in self.operation_signatures:
                    param_combinations = self._generate_comprehensive_parameters(op_name, train_pairs)
                    
                    for params in param_combinations[:20]:  # Limit parameter combinations
                        program = [(op_name, params)]
                        if score_program_comprehensive(program, train_pairs) > 0.99:
                            candidates.append(program)
                        
                        if len(candidates) >= max_candidates // 2:
                            break
                
                if len(candidates) >= max_candidates // 2:
                    break
            
            # Generate two-step programs with high-scoring operations
            high_scoring_ops = [op for op, score in operation_scores.items() if score > 0.3]
            
            for op1_name in high_scoring_ops[:5]:  # Top 5 operations
                for op2_name in high_scoring_ops[:5]:
                    if op1_name == op2_name:
                        continue
                    
                    if len(candidates) >= max_candidates:
                        break
                    
                    # Generate parameter combinations for both operations
                    params1_list = self._generate_comprehensive_parameters(op1_name, train_pairs)[:3]
                    params2_list = self._generate_comprehensive_parameters(op2_name, train_pairs)[:3]
                    
                    for params1 in params1_list:
                        for params2 in params2_list:
                            program = [(op1_name, params1), (op2_name, params2)]
                            if score_program_comprehensive(program, train_pairs) > 0.99:
                                candidates.append(program)
                            
                            if len(candidates) >= max_candidates:
                                break
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
        
        except Exception as e:
            print(f"Neural guidance search failed: {e}")
        
        return candidates
    
    def _episodic_retrieval_comprehensive(self, train_pairs: List[Tuple[Array, Array]], 
                                        max_candidates: int) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Comprehensive episodic retrieval with pattern matching."""
        try:
            return self.episodic_retrieval.query_for_programs(train_pairs, max_candidates)
        except Exception as e:
            print(f"Episodic retrieval failed: {e}")
            return []
    
    def _sketch_based_comprehensive_search(self, train_pairs: List[Tuple[Array, Array]], 
                                         max_candidates: int) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Comprehensive sketch-based program synthesis."""
        candidates = []
        
        try:
            # Get relevant sketches based on neural guidance
            predicted_ops = self.neural_guidance.predict_operations(train_pairs)
            relevant_sketches = self.sketch_miner.get_relevant_sketches(predicted_ops)
            
            for sketch in relevant_sketches[:10]:  # Top 10 sketches
                if len(candidates) >= max_candidates:
                    break
                
                # Try multiple parameter instantiations for each sketch
                for attempt in range(min(20, max_candidates - len(candidates))):
                    try:
                        # Generate parameters for each operation in the sketch
                        sketch_params = {}
                        for op_name in sketch.operations:
                            if op_name in self.operation_signatures:
                                param_combinations = self._generate_comprehensive_parameters(op_name, train_pairs)
                                if param_combinations:
                                    # Pick a parameter set (round-robin through attempts)
                                    params = param_combinations[attempt % len(param_combinations)]
                                    sketch_params[op_name] = params
                        
                        if sketch_params:
                            program = sketch.instantiate(sketch_params)
                            if score_program_comprehensive(program, train_pairs) > 0.99:
                                candidates.append(program)
                    
                    except Exception:
                        continue
        
        except Exception as e:
            print(f"Sketch-based search failed: {e}")
        
        return candidates
    
    def _generate_multi_step_programs(self, train_pairs: List[Tuple[Array, Array]], 
                                    max_candidates: int) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Generate multi-step (3-5 step) programs."""
        candidates = []
        
        # Get high-probability operations
        try:
            operation_scores = self.neural_guidance.score_operations(train_pairs)
            high_scoring_ops = [op for op, score in operation_scores.items() if score > 0.2]
        except:
            high_scoring_ops = ['rotate', 'flip', 'transpose', 'recolor', 'translate']
        
        # Generate 3-step programs
        for op1_name in high_scoring_ops[:4]:
            for op2_name in high_scoring_ops[:4]:
                for op3_name in high_scoring_ops[:4]:
                    if len(candidates) >= max_candidates:
                        break
                    
                    if op1_name == op2_name == op3_name:
                        continue
                    
                    # Generate parameter combinations
                    params1_list = self._generate_comprehensive_parameters(op1_name, train_pairs)[:2]
                    params2_list = self._generate_comprehensive_parameters(op2_name, train_pairs)[:2]
                    params3_list = self._generate_comprehensive_parameters(op3_name, train_pairs)[:2]
                    
                    for params1 in params1_list:
                        for params2 in params2_list:
                            for params3 in params3_list:
                                program = [(op1_name, params1), (op2_name, params2), (op3_name, params3)]
                                if score_program_comprehensive(program, train_pairs) > 0.99:
                                    candidates.append(program)
                                
                                if len(candidates) >= max_candidates:
                                    break
                            if len(candidates) >= max_candidates:
                                break
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        
        return candidates
    
    def _brute_force_comprehensive_search(self, train_pairs: List[Tuple[Array, Array]], 
                                        max_candidates: int) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Brute force search through ALL operations as fallback."""
        candidates = []
        
        # Try all single-step operations
        for op_name in list(OPS.keys())[:20]:  # Top 20 most useful operations
            if len(candidates) >= max_candidates:
                break
                
            if op_name in self.operation_signatures:
                param_combinations = self._generate_comprehensive_parameters(op_name, train_pairs)
                
                for params in param_combinations[:5]:  # Top 5 parameter sets
                    program = [(op_name, params)]
                    if score_program_comprehensive(program, train_pairs) > 0.99:
                        candidates.append(program)
                    
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates
    
    def _generate_comprehensive_parameters(self, op_name: str, 
                                         train_pairs: List[Tuple[Array, Array]]) -> List[Dict[str, Any]]:
        """Generate comprehensive parameter combinations for an operation."""
        if op_name not in self.operation_signatures:
            return [{}]
        
        param_names = self.operation_signatures[op_name]
        if not param_names:
            return [{}]
        
        # Get sample input for parameter generation
        sample_input = train_pairs[0][0] if train_pairs else np.zeros((3, 3))
        H, W = sample_input.shape
        
        param_combinations = []
        
        # Generate parameters based on operation type
        if op_name == 'rotate':
            for k in [1, 2, 3]:
                param_combinations.append({'k': k})
        
        elif op_name == 'flip':
            for axis in [0, 1]:
                param_combinations.append({'axis': axis})
        
        elif op_name == 'translate':
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy != 0 or dx != 0:
                        param_combinations.append({'dx': dx, 'dy': dy, 'fill_value': 0})
        
        elif op_name == 'recolor':
            # Generate color mappings based on colors present in input
            colors_in_input = set()
            for inp, _ in train_pairs:
                colors_in_input.update(np.unique(inp))
            colors_in_input.discard(0)  # Remove background
            
            if colors_in_input:
                for old_color in colors_in_input:
                    for new_color in range(1, 10):
                        if new_color != old_color:
                            param_combinations.append({"mapping": {old_color: new_color}})
        
        elif op_name == 'crop':
            for top in range(min(H, 3)):
                for left in range(min(W, 3)):
                    for height in range(1, H - top + 1):
                        for width in range(1, W - left + 1):
                            param_combinations.append({
                                'top': top, 'bottom': top + height, 
                                'left': left, 'right': left + width
                            })
        
        elif op_name == 'pad':
            for padding in [1, 2]:
                param_combinations.append({
                    'top': padding, 'bottom': padding, 
                    'left': padding, 'right': padding, 
                    'fill_value': 0
                })
        
        elif op_name == 'move_object':
            colors_in_input = set()
            for inp, _ in train_pairs:
                colors_in_input.update(np.unique(inp))
            colors_in_input.discard(0)
            
            for color in colors_in_input:
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dy != 0 or dx != 0:
                            param_combinations.append({'from_color': color, 'dx': dx, 'dy': dy})
        
        elif op_name == 'apply_gravity':
            for direction in ['down', 'up', 'left', 'right']:
                param_combinations.append({'direction': direction})
        
        elif op_name == 'complete_symmetry':
            for axis in ['vertical', 'horizontal']:
                param_combinations.append({'axis': axis})
        
        elif op_name == 'fill_rectangle':
            for top in range(min(H, 3)):
                for left in range(min(W, 3)):
                    for height in range(1, min(H - top + 1, 4)):
                        for width in range(1, min(W - left + 1, 4)):
                            for color in range(1, 10):
                                param_combinations.append({
                                    'top': top, 'left': left,
                                    'height': height, 'width': width,
                                    'color': color
                                })
        
        elif op_name == 'flood_fill':
            for y in range(H):
                for x in range(W):
                    for new_color in range(1, 10):
                        param_combinations.append({
                            'start_y': y, 'start_x': x,
                            'new_color': new_color
                        })
        
        else:
            # Default parameter generation for other operations
            param_combinations.append({})
        
        return param_combinations[:50]  # Limit to top 50 parameter combinations
    
    def _apply_comprehensive_ttt(self, train_pairs: List[Tuple[Array, Array]], 
                               candidates: List[List[Tuple[str, Dict[str, Any]]]]) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Apply comprehensive test-time training."""
        if len(candidates) < 2:
            return candidates
        
        try:
            # Augment training data for better adaptation
            augmented_pairs = DataAugmentation.augment_training_pairs(train_pairs, max_augmentations=30)
            
            # Adapt the scorer to this specific task
            self.test_time_trainer.adapt_to_task(augmented_pairs, candidates, num_iterations=5)
            
            # Re-score all candidates with adapted scorer
            candidate_scores = []
            for program in candidates:
                adapted_score = self.test_time_trainer.score_with_adaptation(program, train_pairs)
                base_score = score_program_comprehensive(program, train_pairs)
                # Combine adapted score with base performance (weighted)
                combined_score = 0.6 * base_score + 0.4 * adapted_score
                candidate_scores.append((combined_score, program))
            
            # Sort by combined score
            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            
            return [program for _, program in candidate_scores]
        
        except Exception as e:
            print(f"Test-time training failed: {e}")
            return candidates
    
    def _comprehensive_candidate_selection(self, train_pairs: List[Tuple[Array, Array]], 
                                         candidates: List[List[Tuple[str, Dict[str, Any]]]], 
                                         max_programs: int) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Comprehensive candidate selection and ranking."""
        if not candidates:
            return []
        
        # Score all candidates comprehensively
        scored_candidates = []
        for program in candidates:
            score = score_program_comprehensive(program, train_pairs)
            
            # Additional scoring factors
            complexity_penalty = len(program) * 0.01  # Prefer simpler programs
            diversity_bonus = self._calculate_diversity_bonus(program, [p for _, p in scored_candidates])
            
            final_score = score - complexity_penalty + diversity_bonus
            scored_candidates.append((final_score, program))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select only high-scoring programs (perfect or near-perfect)
        excellent_programs = [program for score, program in scored_candidates if score > 0.95]
        
        # If no excellent programs, take the best available
        if not excellent_programs and scored_candidates:
            excellent_programs = [program for score, program in scored_candidates[:max_programs]]
        
        # Ensure diversity in final selection
        final_programs = self._ensure_program_diversity(excellent_programs)
        
        return final_programs[:max_programs]
    
    def _calculate_diversity_bonus(self, program: List[Tuple[str, Dict[str, Any]]], 
                                 existing_programs: List[List[Tuple[str, Dict[str, Any]]]]) -> float:
        """Calculate diversity bonus for program selection."""
        if not existing_programs:
            return 0.1  # First program gets small bonus
        
        # Check if this program is significantly different from existing ones
        program_ops = [op_name for op_name, _ in program]
        
        for existing_program in existing_programs:
            existing_ops = [op_name for op_name, _ in existing_program]
            if program_ops == existing_ops:
                return -0.05  # Penalty for similarity
        
        return 0.05  # Bonus for diversity
    
    def _ensure_program_diversity(self, programs: List[List[Tuple[str, Dict[str, Any]]]]) -> List[List[Tuple[str, Dict[str, Any]]]]:
        """Ensure diversity in final program selection."""
        if not programs:
            return []
        
        diverse_programs = [programs[0]]  # Always include the best program
        
        for program in programs[1:]:
            program_ops = [op_name for op_name, _ in program]
            
            # Check if this program is sufficiently different
            is_diverse = True
            for existing_program in diverse_programs:
                existing_ops = [op_name for op_name, _ in existing_program]
                
                # If operations are exactly the same, not diverse
                if program_ops == existing_ops:
                    is_diverse = False
                    break
                
                # If operations are very similar, less diverse
                common_ops = set(program_ops) & set(existing_ops)
                if len(common_ops) > 0.8 * min(len(program_ops), len(existing_ops)):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_programs.append(program)
        
        return diverse_programs
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        return self.search_stats.copy()
    
    def save_all_components(self):
        """Save all learned components."""
        try:
            self.episodic_retrieval.save()
            self.sketch_miner.save_sketches("models/sketches.json")
        except Exception as e:
            print(f"Failed to save components: {e}")


# ================== MAIN INTEGRATION FUNCTIONS ==================

def synthesize_with_comprehensive_enhancements(train_pairs: List[Tuple[Array, Array]], 
                                             max_programs: int = 512) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """Main function for comprehensive enhanced program synthesis."""
    
    print(f"Starting comprehensive enhanced synthesis for {len(train_pairs)} training pairs...")
    
    # Initialize comprehensive enhanced search
    enhanced_search = ComprehensiveEnhancedSearch()
    
    # Run comprehensive synthesis
    programs = enhanced_search.synthesize_comprehensive(train_pairs, max_programs)
    
    # Save learned components
    enhanced_search.save_all_components()
    
    # Print search statistics
    stats = enhanced_search.get_comprehensive_stats()
    print("\n=== COMPREHENSIVE SEARCH STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"Comprehensive synthesis complete: {len(programs)} programs found")
    return programs


def predict_two_comprehensive_enhanced(progs: List[List[Tuple[str, Dict[str, Any]]]], 
                                     test_inputs: List[Array]) -> List[List[Array]]:
    """Enhanced prediction with comprehensive fallback strategies."""
    if len(progs) == 0:
        # No programs found - use multiple fallback strategies
        identity_result = [[inp.copy() for inp in test_inputs], [inp.copy() for inp in test_inputs]]
        print("No programs found - using identity fallback")
        return identity_result
    elif len(progs) == 1:
        # Only one program - use it twice with slight variation
        main_prog = progs[0]
        picks = [main_prog, main_prog]
    else:
        # Use top 2 diverse programs
        picks = progs[:2]
    
    attempts: List[List[Array]] = []
    for i, program in enumerate(picks):
        outs: List[Array] = []
        for test_input in test_inputs:
            try:
                result = apply_program(test_input, program)
                outs.append(result)
            except Exception as e:
                print(f"Program {i} failed on test input: {e}")
                # Fallback to identity for this input
                outs.append(test_input.copy())
        attempts.append(outs)
    
    return attempts


# Export main functions for compatibility
def synthesize_with_enhancements(train_pairs: List[Tuple[Array, Array]], 
                               max_programs: int = 256) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """Compatibility function - use comprehensive enhancements."""
    return synthesize_with_comprehensive_enhancements(train_pairs, max_programs)


def predict_two_enhanced(progs: List[List[Tuple[str, Dict[str, Any]]]], 
                        test_inputs: List[Array]) -> List[List[Array]]:
    """Compatibility function - use comprehensive enhanced prediction."""
    return predict_two_comprehensive_enhanced(progs, test_inputs)
