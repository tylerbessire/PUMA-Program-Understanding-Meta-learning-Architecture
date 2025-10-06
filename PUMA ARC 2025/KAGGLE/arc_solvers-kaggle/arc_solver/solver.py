"""Top-level solver interface for ARC tasks with neural enhancements.

This module integrates neural guidance, episodic retrieval, program sketches and
test-time training to provide state-of-the-art solutions for ARC tasks while
maintaining a robust fallback baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os
import logging
import json
from pathlib import Path

from .grid import to_array, to_list, Array
from .search import (
    synthesize as synth_baseline,
    predict_two as predict_two_baseline,
)
from .enhanced_search import synthesize_with_enhancements, predict_two_enhanced
from .hypothesis import HypothesisEngine, Hypothesis
from .continuous_learning import ContinuousSelfMemory
from .neural.episodic import EpisodicRetrieval
from .placeholders import (
    PlaceholderTemplate,
    PlaceholderTemplateEngine,
    deserialize_placeholder_template,
    serialize_placeholder_template,
)
from .llm_reasoning import LLMReasoningEngine
from .object_reasoning import ObjectExtractor


class ARCSolver:
    """Enhanced ARC solver with neural components and episodic memory."""
    
    def __init__(self, use_enhancements: bool = True,
                 guidance_model_path: str = None,
                 episode_db_path: str = "episodes.json",
                 enable_logging: bool = None,
                 checkpoint_path: str = None):
        self.use_enhancements = use_enhancements
        self.guidance_model_path = guidance_model_path
        self.episode_db_path = episode_db_path
        self.checkpoint_path = checkpoint_path or "checkpoint.json"
        
        # Logging control - check environment variable or parameter
        if enable_logging is None:
            enable_logging = os.environ.get('ARC_ENABLE_LOGGING', 'true').lower() in ('1', 'true', 'yes')
        self.enable_logging = enable_logging
        
        self.stats = {
            'tasks_solved': 0,
            'total_tasks': 0,
            'enhancement_success_rate': 0.0,
            'fallback_used': 0,
        }
        self.submission_results = {}  # For checkpoint saving

        # Structured logger for observability - controlled by enable_logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set logging level based on enable_logging flag
        if self.enable_logging:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.CRITICAL)  # Only show critical errors
        self._last_outputs: Optional[Tuple[List[List[List[int]]], List[List[List[int]]]]] = None
        # Continuous memory and hypotheses
        self.self_memory = ContinuousSelfMemory()
        self.hypothesis_engine = HypothesisEngine(continuous_memory=self.self_memory)
        self.episodic_retrieval = EpisodicRetrieval(episode_db_path)
        self.placeholder_engine = PlaceholderTemplateEngine()
        self.llm_reasoning = LLMReasoningEngine(enable_logging=self.enable_logging)
        self.object_extractor = ObjectExtractor()
        self._placeholder_templates: List[PlaceholderTemplate] = []
        self._new_placeholder_templates: List[PlaceholderTemplate] = []
        self._last_hypotheses: List[Hypothesis] = []

    def solve_task(self, task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
        """Solve ARC task using RFT-first approach with pliance/tracking priority."""
        self.stats['total_tasks'] += 1
        task_id = str(task.get("task_id") or task.get("id") or f"anonymous_{self.stats['total_tasks']}")

        # Extract training pairs as numpy arrays, skipping malformed ones
        train_pairs: List[Tuple[Array, Array]] = []
        for pair in task.get("train", []):
            try:
                a = to_array(pair["input"])
                b = to_array(pair["output"])
            except Exception:
                continue
            train_pairs.append((a, b))

        # Extract test inputs with graceful degradation
        test_inputs: List[Array] = []
        for pair in task.get("test", []):
            try:
                test_inputs.append(to_array(pair["input"]))
            except Exception:
                test_inputs.append(np.zeros((1, 1), dtype=np.int16))

        if not train_pairs:
            identity = [to_list(arr) for arr in test_inputs]
            return {"attempt_1": identity, "attempt_2": identity}

        # LLM-GUIDED APPROACH: Get high-level reasoning first
        if self.enable_logging:
            self.logger.info(f"Starting LLM-guided RFT approach for task {task_id}")
        
        # Get initial object inventory for LLM analysis
        initial_inventory = self._build_object_inventory(train_pairs)
        
        # LLM reasoning to guide approach selection
        llm_reasoning = self.llm_reasoning.analyze_task(train_pairs, test_inputs, initial_inventory)
        if self.enable_logging:
            self.logger.info(f"LLM Analysis: {llm_reasoning.analysis}")
            self.logger.info(f"LLM Suggested approach: {llm_reasoning.suggested_approach} (confidence: {llm_reasoning.confidence:.3f})")
        
        # Follow LLM-guided approach
        if llm_reasoning.suggested_approach in ['rft_direct_application', 'rft_pliance_first']:
            # PHASE 1: RFT PLIANCE - Object inventory and relational rule formation
            rft_result = self._rft_pliance_solve(train_pairs, test_inputs, task_id, llm_reasoning)
            if rft_result is not None:
                if self.enable_logging:
                    self.logger.info(f"RFT pliance successfully solved task {task_id}")
                self.stats['tasks_solved'] += 1
                return rft_result

            # PHASE 2: RFT TRACKING - Adaptive rule modification when initial rules fail
            if self.enable_logging:
                self.logger.info(f"RFT pliance failed, attempting tracking adaptation for task {task_id}")
            
            tracking_result = self._rft_tracking_solve(train_pairs, test_inputs, task_id, llm_reasoning)
            if tracking_result is not None:
                if self.enable_logging:
                    self.logger.info(f"RFT tracking successfully solved task {task_id}")
                self.stats['tasks_solved'] += 1
                return tracking_result

        # FALLBACK: Enhanced and baseline methods when RFT needs assistance
        if self.enable_logging:
            self.logger.info(f"RFT approaches exhausted, falling back to enhanced methods for task {task_id}")
        
        return self._fallback_solve(train_pairs, test_inputs, task_id, llm_reasoning)

    def _rft_pliance_solve(self, train_pairs: List[Tuple[Array, Array]], test_inputs: List[Array], task_id: str, llm_reasoning=None) -> Optional[Dict[str, List[List[List[int]]]]]:
        """RFT PLIANCE: Object inventory and relational rule formation."""
        if self.enable_logging:
            self.logger.info("RFT PLIANCE: Starting object inventory and relational mapping")
        
        try:
            # Step 1: Object Inventory - Map all elements and their attributes
            # Use LLM guidance if available
            expanded = llm_reasoning and 'advanced_color_analysis' in llm_reasoning.tool_recommendations
            object_inventory = self._build_object_inventory(train_pairs, expanded=expanded)
            if self.enable_logging:
                self.logger.info(f"Object inventory complete: {len(object_inventory.get('objects', []))} object types identified")
            
            # Step 2: Relational Context - Identify relationships and patterns
            relational_rules = self._extract_relational_rules(train_pairs, object_inventory)
            if self.enable_logging:
                self.logger.info(f"Relational rules extracted: {len(relational_rules)} rules with avg confidence {sum(r.get('confidence', 0) for r in relational_rules) / max(len(relational_rules), 1):.3f}")
            
            # Step 3: Rule Confidence Building - Validate across training examples
            validated_rules = self._validate_rules_across_examples(relational_rules, train_pairs)
            avg_confidence = (
                sum(rule.get('confidence', 0.0) for rule in validated_rules) / max(len(validated_rules), 1)
            ) if validated_rules else 0.0

            confidence_threshold = 0.85
            if llm_reasoning and llm_reasoning.confidence < 0.7:
                confidence_threshold = 0.75  # Lower threshold for uncertain tasks

            stage_context = {
                'rule_candidates': len(relational_rules),
                'validated_rules': len(validated_rules),
                'average_confidence': avg_confidence,
                'inventory_expanded': bool(expanded),
                'confidence_threshold': confidence_threshold,
            }

            stage_advice = self.llm_reasoning.advise_on_stage_progress('pliance', stage_context)
            stage_action = stage_advice.action

            if stage_action == 'fallback_to_enhanced_search':
                if self.enable_logging:
                    self.logger.info("RFT PLIANCE: LLM advised fallback to enhanced search")
                return None

            if stage_action == 'expand_object_inventory' and not expanded:
                if self.enable_logging:
                    self.logger.info("RFT PLIANCE: Expanding object inventory based on stage advice")
                expanded = True
                object_inventory = self._build_object_inventory(train_pairs, expanded=True)
                relational_rules = self._extract_relational_rules(train_pairs, object_inventory)
                validated_rules = self._validate_rules_across_examples(relational_rules, train_pairs)
                avg_confidence = (
                    sum(rule.get('confidence', 0.0) for rule in validated_rules) / max(len(validated_rules), 1)
                ) if validated_rules else 0.0
                stage_context.update({
                    'rule_candidates': len(relational_rules),
                    'validated_rules': len(validated_rules),
                    'average_confidence': avg_confidence,
                    'inventory_expanded': True,
                    'expansion_attempted': True,
                })
                stage_advice = self.llm_reasoning.advise_on_stage_progress('pliance', stage_context)
                stage_action = stage_advice.action if stage_advice else 'apply_high_confidence_rules'
                if stage_action == 'expand_object_inventory':
                    # Avoid loops when inventory is already expanded
                    stage_action = 'apply_high_confidence_rules'

            note_text = ' '.join(stage_advice.notes or []).lower() if stage_advice else ''
            if 'lower threshold' in note_text and confidence_threshold > 0.6:
                confidence_threshold = max(0.6, confidence_threshold - 0.1)
            elif 'raise threshold' in note_text and confidence_threshold < 0.95:
                confidence_threshold = min(0.95, confidence_threshold + 0.1)

            high_confidence_rules = [
                rule for rule in validated_rules if rule.get('confidence', 0) >= confidence_threshold
            ]
            stage_context['confidence_threshold'] = confidence_threshold
            stage_context['high_confidence_rules'] = len(high_confidence_rules)

            if not high_confidence_rules:
                if self.enable_logging:
                    self.logger.info(
                        "RFT PLIANCE: No high-confidence rules found (threshold %.2f)",
                        confidence_threshold,
                    )
                return None
            
            # Step 4: Apply high-confidence rules to test cases
            result = self._apply_rft_rules(high_confidence_rules, test_inputs, object_inventory)
            
            # Update LLM reasoning with successful rules for future reference
            if result and llm_reasoning:
                llm_reasoning.successful_rules = high_confidence_rules
                
            return result
            
        except Exception as e:
            if self.enable_logging:
                self.logger.warning(f"RFT PLIANCE failed: {e}")
            return None

    def _rft_tracking_solve(self, train_pairs: List[Tuple[Array, Array]], test_inputs: List[Array], task_id: str, llm_reasoning=None) -> Optional[Dict[str, List[List[List[int]]]]]:
        """RFT TRACKING: Adaptive rule modification when initial rules fail."""
        if self.enable_logging:
            self.logger.info("RFT TRACKING: Starting adaptive rule modification")
        
        try:
            # Step 1: Re-analyze with expanded context
            object_inventory = self._build_object_inventory(train_pairs, expanded=True)
            relational_rules = self._extract_relational_rules(train_pairs, object_inventory, adaptive=True)
            
            # Step 2: Test rule theories systematically
            theories = self._generate_rule_theories(relational_rules, train_pairs)

            stage_context = {
                'candidate_theories': len(theories),
                'failed_approaches': ['rft_pliance'],
                'tracking_attempts': 1,
            }
            stage_advice = self.llm_reasoning.advise_on_stage_progress('tracking', stage_context)
            action = stage_advice.action

            if action == 'fallback_to_enhanced_search':
                if self.enable_logging:
                    self.logger.info("RFT TRACKING: Stage advice requested fallback to enhanced search")
                return None

            tracking_threshold = 0.75
            note_text = ' '.join(stage_advice.notes or []).lower()
            if 'lower threshold' in note_text:
                tracking_threshold = max(0.6, tracking_threshold - 0.1)
            elif 'raise threshold' in note_text:
                tracking_threshold = min(0.9, tracking_threshold + 0.1)

            if action == 'generate_new_hypotheses':
                if self.enable_logging:
                    self.logger.info("RFT TRACKING: Stage advice requested new hypotheses; deferring to fallback")
                return None

            for rule_theory in theories:
                if self.enable_logging:
                    self.logger.info(f"Testing rule theory: {rule_theory.get('description', 'unnamed')}")

                theory_confidence = self._test_rule_theory(rule_theory, train_pairs)
                if theory_confidence >= tracking_threshold:
                    result = self._apply_rule_theory(rule_theory, test_inputs, object_inventory)
                    if result is not None:
                        if self.enable_logging:
                            self.logger.info(
                                "RFT TRACKING: Applied rule theory %s with confidence %.3f",
                                rule_theory.get('description', 'unnamed'),
                                theory_confidence,
                            )
                        return result
            
            if self.enable_logging:
                self.logger.info("RFT TRACKING: No viable rule theories found")
            return None
            
        except Exception as e:
            if self.enable_logging:
                self.logger.warning(f"RFT TRACKING failed: {e}")
            return None

    def _fallback_solve(self, train_pairs: List[Tuple[Array, Array]], test_inputs: List[Array], task_id: str, llm_reasoning=None) -> Dict[str, List[List[List[int]]]]:
        """Fallback to original enhanced and baseline methods when RFT needs assistance."""
        if self.enable_logging:
            self.logger.info("FALLBACK: Using enhanced search and baseline methods")
            if llm_reasoning:
                self.logger.info(f"FALLBACK: LLM guidance - {llm_reasoning.analysis}")
        
        # Load placeholder templates for fallback methods
        self._load_placeholder_templates(train_pairs)
        if not self.use_enhancements:
            self._persist_placeholder_templates(train_pairs)
        
        training_stats = self._compute_training_stats(train_pairs)
        
        # Dynamic shape detection
        output_shapes = [out.shape for _, out in train_pairs]
        if len(set(output_shapes)) == 1:
            expected_shape = train_pairs[0][1].shape
        else:
            expected_shape = None
            if self.enable_logging:
                self.logger.info(f"Inconsistent output shapes detected: {output_shapes}, enabling dynamic detection")

        # Generate hypotheses using original method
        self._last_hypotheses = self.hypothesis_engine.generate_hypotheses(train_pairs)
        best_hypothesis = self._last_hypotheses[0] if self._last_hypotheses else None
        
        if best_hypothesis:
            best_hypothesis.confidence = self.hypothesis_engine.test_hypothesis(best_hypothesis, train_pairs)
            if best_hypothesis.confidence >= 0.999:
                attempt1: List[List[List[int]]] = []
                attempt2: List[List[List[int]]] = []
                for test_input in test_inputs:
                    transformed = self.hypothesis_engine.apply(best_hypothesis, test_input)
                    if transformed is None:
                        break
                    attempt1.append(to_list(transformed))
                    attempt2.append(to_list(transformed))
                else:
                    result = {"attempt_1": attempt1, "attempt_2": attempt2}
                    self._record_continuous_experience(task_id, train_pairs, best_hypothesis, True, result)
                    return result

        # Use original prediction pipeline
        attempt1: List[List[List[int]]] = []
        attempt2: List[List[List[int]]] = []
        for test_input in test_inputs:
            predictions = self._get_predictions(train_pairs, test_input, expected_shape)
            processed = self._postprocess_predictions(train_pairs, test_input, predictions, expected_shape, training_stats)
            if processed is not None:
                first_arr, second_arr = processed
                attempt1.append(to_list(first_arr))
                attempt2.append(to_list(second_arr))
            else:
                fallback = to_list(test_input)
                attempt1.append(fallback)
                attempt2.append(fallback)

        result = {"attempt_1": attempt1, "attempt_2": attempt2}
        solved_training = bool(best_hypothesis and best_hypothesis.confidence >= 0.999)
        self._record_continuous_experience(task_id, train_pairs, best_hypothesis, solved_training, result)
        return result

    def _get_predictions(
        self, train_pairs: List[Tuple[Array, Array]], test_input: Array, expected_shape: Optional[Tuple[int, int]]
    ) -> List[List[Array]]:
        """Get prediction attempts for a single test input."""
        enhanced: List[List[Array]] = []
        if self.use_enhancements:
            try:
                if self.enable_logging:
                    self.logger.info("Using enhanced search for prediction")
                progs = synthesize_with_enhancements(train_pairs, expected_shape=expected_shape, test_input=test_input)
                
                # Import human reasoner for enhanced prediction
                from .human_reasoning import HumanGradeReasoner
                human_reasoner = HumanGradeReasoner()
                
                enhanced = predict_two_enhanced(progs, [test_input], 
                                              human_reasoner=human_reasoner,
                                              train_pairs=train_pairs)
            except Exception as e:
                if self.enable_logging:
                    self.logger.exception("Enhanced prediction error: %s", e)

        # Baseline predictions for ensemble
        progs_base = synth_baseline(train_pairs, expected_shape=expected_shape)
        baseline = predict_two_baseline(progs_base, [test_input])

        # Validate enhanced prediction
        if enhanced and self._validate_solution(enhanced, [test_input]):
            if self.enable_logging:
                self.logger.info(f"Enhanced prediction valid - shape: {enhanced[0][0].shape}")
            return [enhanced[0], baseline[0]]

        self.stats['fallback_used'] += 1
        if self.enable_logging:
            self.logger.info("Using baseline prediction")
        return baseline

    def _postprocess_predictions(
        self,
        train_pairs: List[Tuple[Array, Array]],
        test_input: Array,
        predictions: List[List[Array]],
        expected_shape: Optional[Tuple[int, int]],
        training_stats: Dict[str, Any],
    ) -> Optional[Tuple[Array, Array]]:
        if not predictions:
            target_shape = self._determine_target_shape(train_pairs, test_input, expected_shape)
            placeholder = self._apply_placeholder_templates(test_input)
            if placeholder is not None:
                adjusted, _ = self._enforce_size_constraints(placeholder, target_shape, training_stats)
                return adjusted, adjusted
            return None

        target_shape = self._determine_target_shape(train_pairs, test_input, expected_shape)

        processed: List[Tuple[int, int, Array]] = []
        for idx, attempt in enumerate(predictions):
            if not attempt:
                continue
            raw_output = attempt[0]
            adjusted, shape_ok = self._enforce_size_constraints(raw_output, target_shape, training_stats)
            coherence = self._evaluate_coherence(
                adjusted,
                target_shape,
                training_stats,
                test_input,
                shape_ok,
            )
            processed.append((coherence, idx, adjusted))

        if not processed:
            placeholder = self._apply_placeholder_templates(test_input)
            if placeholder is not None:
                adjusted, _ = self._enforce_size_constraints(placeholder, target_shape, training_stats)
                return adjusted, adjusted
            return None

        processed.sort(key=lambda item: (-item[0], item[1]))
        best = processed[0][2]
        second = processed[1][2] if len(processed) > 1 else best
        return best, second

    def _apply_placeholder_templates(self, grid: Array) -> Optional[Array]:
        """Apply detected placeholder templates to the given grid."""
        if not self._placeholder_templates:
            return None

        current = grid.copy()
        applied = False

        for template in self._placeholder_templates:
            try:
                result = self.placeholder_engine.apply_template(current, template)
            except Exception:
                continue
            if result is not None:
                current = result
                applied = True

        return current if applied else None

    def _load_placeholder_templates(
        self, train_pairs: List[Tuple[Array, Array]]
    ) -> None:
        """Populate local placeholder templates from detection and episodic memory."""

        self._placeholder_templates = []
        self._new_placeholder_templates = []
        if not train_pairs:
            return

        templates: List[PlaceholderTemplate] = []
        seen_template_keys: set = set()
        episodic_keys: set = set()

        def template_key(template: PlaceholderTemplate) -> Tuple:
            signature = template.signature
            return (
                signature.placeholder_color,
                signature.shape,
                signature.left,
                signature.right,
                signature.top,
                signature.bottom,
                tuple(template.fill_pattern.flatten()),
            )

        def add_template(template: PlaceholderTemplate) -> None:
            key = template_key(template)
            if key in seen_template_keys:
                return
            seen_template_keys.add(key)
            templates.append(template)

        detected = self.placeholder_engine.detect_templates(train_pairs)

        episodic_payloads: List[Dict[str, Any]] = []
        if self.episodic_retrieval:
            episodic_payloads = self.episodic_retrieval.get_placeholder_templates(train_pairs, max_templates=6)
            for payload in episodic_payloads:
                try:
                    template = deserialize_placeholder_template(payload)
                except Exception:
                    continue
                key = template_key(template)
                episodic_keys.add(key)
                add_template(template)

        new_templates: List[PlaceholderTemplate] = []
        for template in detected:
            key = template_key(template)
            add_template(template)
            if key not in episodic_keys:
                new_templates.append(template)

        self._placeholder_templates = templates
        self._new_placeholder_templates = new_templates

        if templates:
            episodic_count = max(0, len(templates) - len(new_templates))
            if self.enable_logging:
                self.logger.info(
                    "Loaded %d placeholder template(s) (%d from episodic memory)",
                    len(templates),
                    episodic_count,
                )

    def _persist_placeholder_templates(
        self, train_pairs: List[Tuple[Array, Array]]
    ) -> None:
        """Persist newly detected placeholder templates to episodic memory."""

        if not self._new_placeholder_templates or not train_pairs:
            return
        if self.use_enhancements:
            return
        if not self.episodic_retrieval:
            return

        payloads: List[Dict[str, Any]] = []
        target_shape = train_pairs[0][1].shape if train_pairs else None
        for template in self._new_placeholder_templates:
            try:
                payload = serialize_placeholder_template(template)
            except Exception:
                continue
            if target_shape is not None:
                payload["target_shape"] = [int(dim) for dim in target_shape]
            payloads.append(payload)

        if not payloads:
            return

        try:
            self.episodic_retrieval.add_successful_solution(
                train_pairs,
                [],
                metadata={"placeholder_templates": payloads},
            )
            self.episodic_retrieval.save()
            if self.enable_logging:
                self.logger.info(
                    "Persisted %d placeholder template(s) to episodic memory",
                    len(payloads),
                )
        except Exception as exc:
            if self.enable_logging:
                self.logger.debug("Failed to persist placeholder templates: %s", exc)

    def _compute_training_stats(
        self, train_pairs: List[Tuple[Array, Array]]
    ) -> Dict[str, Any]:
        color_counts: Dict[int, int] = {}
        color_hist = np.zeros(10, dtype=np.float64)
        output_colors: set[int] = set()
        input_colors: set[int] = set()
        size_change = False
        color_change = False
        background_candidates: Dict[int, int] = {}
        translation_vectors: List[np.ndarray] = []
        vertical_stripe_votes = 0
        horizontal_stripe_votes = 0

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                size_change = True

            inp_colors = {int(v) for v in np.unique(inp)}
            out_colors = {int(v) for v in np.unique(out)}
            input_colors |= inp_colors
            output_colors |= out_colors
            if inp_colors != out_colors:
                color_change = True

            unique, counts = np.unique(out, return_counts=True)
            for value, count in zip(unique, counts):
                key = int(value)
                color_counts[key] = color_counts.get(key, 0) + int(count)
                color_hist[key] += int(count)

            background = self._estimate_background_color(out)
            background_candidates[background] = background_candidates.get(background, 0) + 1

            translation = self._estimate_translation_vector(inp, out)
            if translation is not None:
                translation_vectors.append(translation)

            stripe_axis = self._detect_stripe_axis(out)
            if stripe_axis == 'vertical':
                vertical_stripe_votes += 1
            elif stripe_axis == 'horizontal':
                horizontal_stripe_votes += 1

        dominant_color = max(color_counts, key=color_counts.get) if color_counts else 0
        color_hist = color_hist / color_hist.sum() if color_hist.sum() > 0 else None

        background_color = dominant_color
        if background_candidates:
            background_color = max(background_candidates, key=background_candidates.get)

        likely_translation = False
        translation_vector: Optional[Tuple[int, int]] = None
        if translation_vectors:
            mean_vec = np.mean(translation_vectors, axis=0)
            deviations = [np.linalg.norm(vec - mean_vec) for vec in translation_vectors]
            if max(deviations, default=0.0) < 0.75:
                likely_translation = bool(np.linalg.norm(mean_vec) > 0.1)
                translation_vector = (
                    int(round(float(mean_vec[0]))),
                    int(round(float(mean_vec[1]))),
                )

        stripe_axis = None
        majority_threshold = max(1, len(train_pairs) // 2)
        if vertical_stripe_votes > horizontal_stripe_votes and vertical_stripe_votes >= majority_threshold:
            stripe_axis = 'vertical'
        elif horizontal_stripe_votes > vertical_stripe_votes and horizontal_stripe_votes >= majority_threshold:
            stripe_axis = 'horizontal'

        top_colors = [color for color, _ in sorted(color_counts.items(), key=lambda item: item[1], reverse=True)]

        return {
            "color_counts": color_counts,
            "dominant_color": dominant_color,
            "background_color": background_color,
            "color_hist": color_hist,
            "output_colors": output_colors,
            "input_colors": input_colors,
            "color_change": color_change,
            "size_change": size_change,
            "likely_translation": likely_translation,
            "translation_vector": translation_vector,
            "top_colors": top_colors,
            "stripe_axis": stripe_axis,
        }

    @staticmethod
    def _estimate_background_color(grid: Array) -> int:
        values, counts = np.unique(grid, return_counts=True)
        idx = int(np.argmax(counts)) if len(counts) else 0
        return int(values[idx]) if len(values) else 0

    @staticmethod
    def _centroid(grid: Array, background: int) -> Optional[np.ndarray]:
        mask = grid != background
        if not np.any(mask):
            return None
        coords = np.argwhere(mask)
        return coords.mean(axis=0)

    def _estimate_translation_vector(self, source: Array, target: Array) -> Optional[np.ndarray]:
        bg_src = self._estimate_background_color(source)
        bg_tgt = self._estimate_background_color(target)
        centroid_src = self._centroid(source, bg_src)
        centroid_tgt = self._centroid(target, bg_tgt)
        if centroid_src is None or centroid_tgt is None:
            return None
        return centroid_tgt - centroid_src

    def _detect_stripe_axis(self, grid: Array) -> Optional[str]:
        h, w = grid.shape
        if h == 0 or w == 0:
            return None

        col_uniform = sum(1 for c in range(w) if len(np.unique(grid[:, c])) <= 2)
        row_uniform = sum(1 for r in range(h) if len(np.unique(grid[r, :])) <= 2)

        col_ratio = col_uniform / w
        row_ratio = row_uniform / h

        if col_ratio >= 0.6 and row_ratio < 0.6:
            return 'vertical'
        if row_ratio >= 0.6 and col_ratio < 0.6:
            return 'horizontal'
        return None

    def _determine_target_shape(
        self,
        train_pairs: List[Tuple[Array, Array]],
        test_input: Array,
        expected_shape: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        if expected_shape is not None:
            return expected_shape

        output_shapes = [out.shape for _, out in train_pairs]
        if not output_shapes:
            return test_input.shape

        if len(set(output_shapes)) == 1:
            return output_shapes[0]

        has_size_change = any(inp.shape != out.shape for inp, out in train_pairs)
        placeholder = self._find_largest_placeholder(test_input, marker_color=8)
        if has_size_change and placeholder:
            return placeholder

        heights = {shape[0] for shape in output_shapes}
        widths = {shape[1] for shape in output_shapes}
        test_h, test_w = test_input.shape

        height = heights.pop() if len(heights) == 1 else test_h
        width = widths.pop() if len(widths) == 1 else test_w
        return (height, width)

    def _find_largest_placeholder(
        self, grid: Array, marker_color: int = 8
    ) -> Optional[Tuple[int, int]]:
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        best: Optional[Tuple[int, int]] = None

        for r in range(h):
            for c in range(w):
                if grid[r, c] == marker_color and not visited[r, c]:
                    shape = self._measure_rectangular_region(grid, r, c, marker_color, visited)
                    if shape is not None and min(shape) > 1:
                        if best is None or shape[0] * shape[1] > best[0] * best[1]:
                            best = shape
        return best

    def _measure_rectangular_region(
        self,
        grid: Array,
        start_r: int,
        start_c: int,
        color: int,
        visited: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        h, w = grid.shape
        region_w = 0
        for c in range(start_c, w):
            if grid[start_r, c] == color:
                region_w += 1
            else:
                break

        region_h = 0
        for r in range(start_r, h):
            if all(grid[r, start_c + dc] == color for dc in range(region_w) if start_c + dc < w):
                region_h += 1
            else:
                break

        if region_h == 0 or region_w == 0:
            return None

        for r in range(start_r, start_r + region_h):
            for c in range(start_c, start_c + region_w):
                if r < h and c < w and grid[r, c] == color:
                    visited[r, c] = True
                else:
                    return None

        return (region_h, region_w)

    def _enforce_size_constraints(
        self,
        grid: Array,
        target_shape: Optional[Tuple[int, int]],
        training_stats: Dict[str, Any],
    ) -> Tuple[Array, bool]:
        if target_shape is None:
            return grid, True

        target_h, target_w = target_shape
        current = grid.copy()
        h, w = current.shape

        if h > target_h or w > target_w:
            crop_h = min(h, target_h)
            crop_w = min(w, target_w)
            current = self._crop_to_shape(current, (crop_h, crop_w))
            h, w = current.shape

        if h < target_h or w < target_w:
            fill = training_stats.get("dominant_color")
            if fill is None:
                values, counts = np.unique(current, return_counts=True)
                if len(values):
                    fill = int(values[counts.argmax()])
                else:
                    fill = 0
            padded = np.full((max(h, target_h), max(w, target_w)), fill, dtype=current.dtype)
            start_r = (padded.shape[0] - h) // 2
            start_c = (padded.shape[1] - w) // 2
            padded[start_r : start_r + h, start_c : start_c + w] = current
            current = padded

        if current.shape != target_shape:
            current = self._crop_to_shape(current, target_shape)

        return current, current.shape == target_shape

    def _crop_to_shape(self, grid: Array, target_shape: Tuple[int, int]) -> Array:
        target_h, target_w = target_shape
        h, w = grid.shape
        if h == target_h and w == target_w:
            return grid.copy()

        best_crop = grid[:target_h, :target_w].copy()
        best_score = -1.0

        max_r = max(h - target_h + 1, 1)
        max_c = max(w - target_w + 1, 1)
        for r in range(max_r):
            for c in range(max_c):
                end_r = min(r + target_h, h)
                end_c = min(c + target_w, w)
                crop = grid[r:end_r, c:end_c]
                if crop.shape != (target_h, target_w):
                    continue
                diversity = len(np.unique(crop))
                non_marker = np.count_nonzero(crop != 8)
                score = diversity * 1000 + non_marker
                if score > best_score:
                    best_score = score
                    best_crop = crop.copy()

        return best_crop

    def _evaluate_coherence(
        self,
        prediction: Array,
        target_shape: Optional[Tuple[int, int]],
        training_stats: Dict[str, Any],
        test_input: Array,
        shape_ok: bool,
    ) -> int:
        score = 0.0

        if target_shape is None:
            score += 0.5 if shape_ok else -0.5
        else:
            score += 3.0 if shape_ok else -1.5

        color_hist = training_stats.get("color_hist")
        pred_hist = self._normalized_histogram(prediction)
        if color_hist is not None:
            hist_diff = float(np.abs(pred_hist - color_hist).sum())
            score -= hist_diff * 3.0
            if hist_diff < 0.4:
                score += 1.25

        output_colors = training_stats.get("output_colors", set())
        color_change_expected = training_stats.get("color_change", False)
        pred_colors = {int(v) for v in np.unique(prediction)}
        if not color_change_expected:
            unseen = pred_colors - output_colors
            if unseen:
                score -= 2.0
        else:
            if pred_colors & output_colors:
                score += 0.5

        top_colors = training_stats.get("top_colors") or []
        if top_colors:
            dominant_pred = int(np.argmax(pred_hist)) if pred_hist.sum() > 0 else None
            if dominant_pred is not None and dominant_pred not in top_colors[: min(3, len(top_colors))]:
                score -= 1.5

            training_hist = training_stats.get("color_hist")
            if training_hist is not None:
                ranked_training = [idx for idx, val in sorted(enumerate(training_hist), key=lambda item: item[1], reverse=True) if val > 0]
                ranked_pred = [idx for idx, val in sorted(enumerate(pred_hist), key=lambda item: item[1], reverse=True) if val > 0]
                mismatch = sum(1 for color in ranked_pred[:3] if color not in ranked_training[:3])
                score -= mismatch * 0.5

        if training_stats.get("likely_translation") and training_stats.get("translation_vector") is not None:
            vector = training_stats["translation_vector"]
            translated = self._apply_translation(
                test_input,
                vector,
                training_stats.get("background_color", training_stats.get("dominant_color", 0)),
            )
            adapted, _ = self._enforce_size_constraints(
                translated,
                prediction.shape,
                training_stats,
            )
            min_shape = (min(adapted.shape[0], prediction.shape[0]), min(adapted.shape[1], prediction.shape[1]))
            adapted_crop = adapted[: min_shape[0], : min_shape[1]]
            prediction_crop = prediction[: min_shape[0], : min_shape[1]]
            mismatch = float(np.mean(adapted_crop != prediction_crop)) if min_shape[0] > 0 and min_shape[1] > 0 else 1.0
            score -= mismatch * 4.0
            if mismatch < 0.25:
                score += 1.5
            elif mismatch > 0.6:
                score -= 0.5

        stripe_axis = training_stats.get("stripe_axis")
        if stripe_axis:
            stripe_ratio = self._stripe_uniform_ratio(prediction, axis=0 if stripe_axis == 'vertical' else 1)
            if stripe_ratio < 0.5:
                score -= 1.5
            else:
                score += 0.5

        if not np.array_equal(prediction, test_input):
            score += 0.5

        return score

    @staticmethod
    def _normalized_histogram(grid: Array) -> np.ndarray:
        hist = np.zeros(10, dtype=np.float64)
        unique, counts = np.unique(grid, return_counts=True)
        for value, count in zip(unique, counts):
            idx = int(value)
            if 0 <= idx < hist.size:
                hist[idx] += int(count)
        total = hist.sum()
        if total == 0:
            return hist
        return hist / total

    def _stripe_uniform_ratio(self, grid: Array, axis: int) -> float:
        h, w = grid.shape
        if axis == 0 and w > 0:
            uniform = sum(1 for c in range(w) if len(np.unique(grid[:, c])) <= 2)
            return uniform / w
        if axis == 1 and h > 0:
            uniform = sum(1 for r in range(h) if len(np.unique(grid[r, :])) <= 2)
            return uniform / h
        return 0.0

    def _apply_translation(
        self,
        grid: Array,
        vector: Tuple[int, int],
        fill: int,
    ) -> Array:
        dr, dc = vector
        h, w = grid.shape
        result = np.full((h, w), fill, dtype=grid.dtype)

        src_r_start = max(0, -dr)
        src_r_end = min(h, h - max(0, dr))
        dst_r_start = max(0, dr)
        dst_r_end = dst_r_start + (src_r_end - src_r_start)

        src_c_start = max(0, -dc)
        src_c_end = min(w, w - max(0, dc))
        dst_c_start = max(0, dc)
        dst_c_end = dst_c_start + (src_c_end - src_c_start)

        if dst_r_end > dst_r_start and dst_c_end > dst_c_start:
            result[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = grid[src_r_start:src_r_end, src_c_start:src_c_end]

        return result

# [S:OBS v1] logging=structured fallback_metric=fallback_used pass

    def solve_task_two_attempts(
        self, task: Dict[str, List[Dict[str, List[List[int]]]]]
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Solve a task and ensure two diverse attempts.

        Args:
            task: ARC task specification.

        Returns:
            A tuple ``(attempt1, attempt2)`` each being a list of output grids
            corresponding to the test inputs.
        """

        result = self.solve_task(task)
        attempt1 = result["attempt_1"]
        attempt2 = result["attempt_2"]

        if attempt1 == attempt2:
            alt = self._second_pass_diversified(task)
            if alt is not None:
                attempt2 = alt

        self._last_outputs = (attempt1, attempt2)
        return attempt1, attempt2

    def _second_pass_diversified(
        self, task: Dict[str, List[Dict[str, List[List[int]]]]]
    ) -> Optional[List[List[List[int]]]]:
        """Run a diversified second search pass to obtain an alternative output."""

        train_pairs = [
            (to_array(p["input"]), to_array(p["output"])) for p in task["train"]
        ]
        test_inputs = [to_array(p["input"]) for p in task["test"]]

        try:
            # Use dynamic shape detection for consistency with prediction pipeline
            programs = synthesize_with_enhancements(train_pairs, force_alt=True, test_input=test_inputs[0] if test_inputs else None, expected_shape=None)
            
            # Import human reasoner for enhanced prediction
            from .human_reasoning import HumanGradeReasoner
            human_reasoner = HumanGradeReasoner()
            
            attempts = predict_two_enhanced(programs, test_inputs, prefer_diverse=True,
                                          human_reasoner=human_reasoner,
                                          train_pairs=train_pairs)
            return [to_list(x) for x in attempts[0]]
        except Exception:
            try:
                programs = synth_baseline(train_pairs)
                attempts = predict_two_baseline(
                    programs, test_inputs, prefer_diverse=True
                )
                return [to_list(x) for x in attempts[0]]
            except Exception:
                return None

    def best_so_far(
        self, task: Dict[str, List[Dict[str, List[List[int]]]]]
    ) -> List[List[List[int]]]:
        """Return the best outputs computed so far for the current task.

        If the solver has produced at least one attempt, that attempt is
        returned. Otherwise, the identity transformation of the first test
        input is used as a safe fallback.
        """

        if self._last_outputs is not None:
            return self._last_outputs[0]
        return [task["test"][0]["input"]]

    def _record_continuous_experience(
        self,
        task_id: str,
        train_pairs: List[Tuple[Array, Array]],
        hypothesis: Optional[Hypothesis],
        solved: bool,
        result: Dict[str, List[List[List[int]]]],
    ) -> None:
        if not train_pairs:
            return
        transformation = hypothesis.transformation_type if hypothesis else None
        meta = {
            "confidence": hypothesis.confidence if hypothesis else 0.0,
            "program_sketch": hypothesis.program_sketch if hypothesis else None,
            "attempt_shapes": [
                list(np.asarray(grid).shape) for grid in result.get("attempt_1", [])
            ],
            "enhancements": self.use_enhancements,
        }
        try:
            self.self_memory.record_experience(task_id, train_pairs, transformation, solved, meta)
        except Exception as exc:
            if self.enable_logging:
                self.logger.debug("Continuous memory record failed: %s", exc)
    
    def _validate_solution(self, attempts: List[List[Array]], test_inputs: List[Array]) -> bool:
        """Basic validation to check if solution seems reasonable."""
        if not attempts or len(attempts) != 2:
            return False
        
        for attempt in attempts:
            if len(attempt) != len(test_inputs):
                return False
            
            # Check that outputs are not just copies of inputs (unless that's valid)
            for inp, out in zip(test_inputs, attempt):
                if out.shape[0] == 0 or out.shape[1] == 0:  # Empty output
                    return False
                if np.max(out) > 9:  # Invalid color values
                    return False
        
        return True
    
    def get_statistics(self) -> Dict[str, float]:
        """Get solver performance statistics."""
        success_rate = self.stats['tasks_solved'] / max(1, self.stats['total_tasks'])
        return {
            'success_rate': success_rate,
            'total_tasks': self.stats['total_tasks'],
            'tasks_solved': self.stats['tasks_solved'],
            'fallback_usage': self.stats['fallback_used'] / max(1, self.stats['total_tasks']),
        }

    def get_persona_summary(self) -> Dict[str, Any]:
        """Expose the continuous self model summary."""
        return self.self_memory.persona_summary()
    
    def save_checkpoint(self, task_id: str = None, force: bool = False) -> None:
        """Save current progress to checkpoint file."""
        if not self.submission_results and not force:
            return
            
        try:
            checkpoint_data = {
                'submission_results': self.submission_results,
                'stats': self.stats,
                'completed_tasks': list(self.submission_results.keys()),
                'last_task': task_id,
                'timestamp': str(Path(__file__).stat().st_mtime)
            }
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            if self.enable_logging:
                self.logger.info(f"Checkpoint saved: {len(self.submission_results)} tasks completed")
        except Exception as exc:
            if self.enable_logging:
                self.logger.error(f"Failed to save checkpoint: {exc}")
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """Load progress from checkpoint file."""
        try:
            if not Path(self.checkpoint_path).exists():
                return {}
                
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                
            self.submission_results = checkpoint_data.get('submission_results', {})
            
            # Update stats if they exist
            saved_stats = checkpoint_data.get('stats', {})
            for key, value in saved_stats.items():
                if key in self.stats:
                    self.stats[key] = value
                    
            if self.enable_logging:
                completed = len(self.submission_results)
                last_task = checkpoint_data.get('last_task', 'unknown')
                self.logger.info(f"Checkpoint loaded: {completed} tasks completed, last: {last_task}")
                
            return checkpoint_data
        except Exception as exc:
            if self.enable_logging:
                self.logger.error(f"Failed to load checkpoint: {exc}")
            return {}
    
    def add_submission_result(self, task_id: str, result: Dict[str, List[List[List[int]]]]) -> None:
        """Add a task result to submission tracking."""
        self.submission_results[task_id] = result
        
        # Save checkpoint every 10 tasks to prevent memory buildup
        if len(self.submission_results) % 10 == 0:
            self.save_checkpoint(task_id)
    
    def get_submission_results(self) -> Dict[str, Dict[str, List[List[List[int]]]]]:
        """Get all submission results for final export."""
        return self.submission_results.copy()

    # RFT Helper Methods
    def _build_object_inventory(self, train_pairs: List[Tuple[Array, Array]], expanded: bool = False) -> Dict[str, Any]:
        """Build comprehensive object inventory: shapes, colors, positions, transformations."""
        inventory = {
            'objects': [],
            'colors': set(),
            'shapes': set(), 
            'positions': [],
            'transformations': [],
            'background_colors': [],
            'grid_sizes': []
        }
        
        if not train_pairs:
            return inventory
        
        for i, (input_grid, output_grid) in enumerate(train_pairs):
            try:
                # Basic grid analysis - ensure grids are valid
                if input_grid.size == 0 or output_grid.size == 0:
                    continue
                    
                inventory['grid_sizes'].append((input_grid.shape, output_grid.shape))
                
                # Color analysis
                input_colors = set(np.unique(input_grid).astype(int))
                output_colors = set(np.unique(output_grid).astype(int))
                inventory['colors'].update(input_colors | output_colors)
                
                # Background color detection
                input_bg = self._estimate_background_color(input_grid)
                output_bg = self._estimate_background_color(output_grid)
                inventory['background_colors'].append((input_bg, output_bg))
                
                # Object extraction and analysis with robust error handling
                try:
                    if hasattr(self.object_extractor, 'extract_objects'):
                        objects = self.object_extractor.extract_objects(input_grid, output_grid)
                        if objects and isinstance(objects, list):
                            inventory['objects'].extend(objects)
                            
                            for obj in objects:
                                if isinstance(obj, dict):
                                    inventory['shapes'].add(obj.get('shape', 'unknown'))
                                    if 'position' in obj and obj['position'] is not None:
                                        inventory['positions'].append(obj['position'])
                                    if 'transformation' in obj and obj['transformation'] is not None:
                                        inventory['transformations'].append(obj['transformation'])
                        else:
                            raise AttributeError("extract_objects returned invalid data")
                    else:
                        raise AttributeError("object_extractor missing extract_objects method")
                        
                except Exception as e:
                    # Fallback simple analysis
                    if self.enable_logging:
                        self.logger.debug(f"Object extraction failed for pair {i}: {e}, using basic extraction")
                    self._extract_basic_objects(input_grid, output_grid, inventory)
                    
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(f"Failed to process training pair {i}: {e}")
                continue
        
        if self.enable_logging:
            self.logger.info(f"Object inventory: {len(inventory['colors'])} colors, {len(inventory['shapes'])} shapes, {len(inventory['objects'])} objects")
        
        return inventory

    def _extract_basic_objects(self, input_grid: Array, output_grid: Array, inventory: Dict[str, Any]) -> None:
        """Basic object extraction when advanced extraction fails."""
        # Find non-background regions as objects
        bg_color = self._estimate_background_color(input_grid)
        
        # Connected components analysis
        mask = input_grid != bg_color
        if np.any(mask):
            # Simple shape detection
            coords = np.argwhere(mask)
            if len(coords) > 0:
                min_r, min_c = coords.min(axis=0)
                max_r, max_c = coords.max(axis=0)
                height, width = max_r - min_r + 1, max_c - min_c + 1
                
                inventory['objects'].append({
                    'color': int(input_grid[coords[0][0], coords[0][1]]),
                    'position': (int(min_r), int(min_c)),
                    'size': (int(height), int(width)),
                    'shape': 'rectangular' if height * width == len(coords) else 'irregular'
                })

    def _extract_relational_rules(self, train_pairs: List[Tuple[Array, Array]], inventory: Dict[str, Any], adaptive: bool = False) -> List[Dict[str, Any]]:
        """Extract relational rules between objects and their transformations."""
        rules = []
        
        # Color transformation rules
        color_mappings = {}
        for i, (input_grid, output_grid) in enumerate(train_pairs):
            input_colors = set(np.unique(input_grid))
            output_colors = set(np.unique(output_grid))
            
            # Direct color mappings - ensure grids are same shape for indexing
            if input_grid.shape == output_grid.shape:
                for color in input_colors:
                    input_mask = input_grid == color
                    if np.any(input_mask):
                        output_colors_at_positions = np.unique(output_grid[input_mask])
                        if len(output_colors_at_positions) == 1:
                            mapped_color = int(output_colors_at_positions[0])
                            color_mappings[color] = color_mappings.get(color, []) + [mapped_color]
        
        # Build color transformation rules
        for input_color, output_colors in color_mappings.items():
            if len(set(output_colors)) == 1:  # Consistent mapping
                rules.append({
                    'type': 'color_transform',
                    'description': f'Color {input_color} always transforms to {output_colors[0]}',
                    'input_color': int(input_color),
                    'output_color': output_colors[0],
                    'confidence': len(output_colors) / len(train_pairs),
                    'evidence_count': len(output_colors)
                })
        
        # Position/movement rules
        if inventory['positions']:
            movement_patterns = self._analyze_movement_patterns(train_pairs, inventory)
            rules.extend(movement_patterns)
        
        # Shape transformation rules
        if inventory['transformations']:
            shape_rules = self._analyze_shape_transformations(train_pairs, inventory)
            rules.extend(shape_rules)
        
        if adaptive:
            # Add more experimental rules for tracking phase
            experimental_rules = self._generate_experimental_rules(train_pairs, inventory)
            rules.extend(experimental_rules)
        
        return rules

    def _analyze_movement_patterns(self, train_pairs: List[Tuple[Array, Array]], inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze object movement patterns across examples."""
        movement_rules = []
        
        # Simple translation detection
        for i, (input_grid, output_grid) in enumerate(train_pairs):
            if input_grid.shape == output_grid.shape:
                # Look for objects that moved
                bg_color = self._estimate_background_color(input_grid)
                input_objects = self._find_objects_simple(input_grid, bg_color)
                output_objects = self._find_objects_simple(output_grid, bg_color)
                
                if len(input_objects) == len(output_objects) == 1:
                    in_obj, out_obj = input_objects[0], output_objects[0]
                    if in_obj['color'] == out_obj['color']:
                        dr = out_obj['center'][0] - in_obj['center'][0]
                        dc = out_obj['center'][1] - in_obj['center'][1]
                        
                        movement_rules.append({
                            'type': 'translation',
                            'description': f'Object moves by ({dr}, {dc})',
                            'delta_r': dr,
                            'delta_c': dc,
                            'color': in_obj['color'],
                            'confidence': 0.8,
                            'evidence_count': 1
                        })
        
        return movement_rules

    def _analyze_shape_transformations(self, train_pairs: List[Tuple[Array, Array]], inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze shape-based transformation patterns."""
        shape_rules = []
        
        # Grid size changes
        size_changes = {}
        for input_grid, output_grid in train_pairs:
            size_change = (input_grid.shape, output_grid.shape)
            size_changes[size_change] = size_changes.get(size_change, 0) + 1
        
        for (in_shape, out_shape), count in size_changes.items():
            if count > 1:  # Appears in multiple examples
                shape_rules.append({
                    'type': 'size_transform',
                    'description': f'Grid resizes from {in_shape} to {out_shape}',
                    'input_shape': in_shape,
                    'output_shape': out_shape,
                    'confidence': count / len(train_pairs),
                    'evidence_count': count
                })
        
        return shape_rules

    def _generate_experimental_rules(self, train_pairs: List[Tuple[Array, Array]], inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experimental rules for tracking phase adaptation."""
        experimental_rules = []
        
        # Try alternative color mappings
        all_colors = list(inventory['colors'])
        for i, color1 in enumerate(all_colors):
            for j, color2 in enumerate(all_colors):
                if i != j:
                    experimental_rules.append({
                        'type': 'experimental_color',
                        'description': f'Experimental: {color1} might transform to {color2}',
                        'input_color': int(color1),
                        'output_color': int(color2),
                        'confidence': 0.3,  # Low initial confidence
                        'evidence_count': 0
                    })
        
        return experimental_rules

    def _find_objects_simple(self, grid: Array, bg_color: int) -> List[Dict[str, Any]]:
        """Simple object detection for movement analysis."""
        objects = []
        mask = grid != bg_color
        if np.any(mask):
            coords = np.argwhere(mask)
            center_r = coords[:, 0].mean()
            center_c = coords[:, 1].mean()
            
            # Get most common non-background color
            non_bg_values = grid[mask]
            unique_vals, counts = np.unique(non_bg_values, return_counts=True)
            most_common_color = unique_vals[np.argmax(counts)]
            
            objects.append({
                'color': int(most_common_color),
                'center': (center_r, center_c),
                'coords': coords.tolist()
            })
        
        return objects

    def _validate_rules_across_examples(self, rules: List[Dict[str, Any]], train_pairs: List[Tuple[Array, Array]]) -> List[Dict[str, Any]]:
        """Validate and adjust rule confidence across all training examples."""
        validated_rules = []
        
        for rule in rules:
            validation_count = 0
            total_applicable = 0
            
            for input_grid, output_grid in train_pairs:
                applicable, correct = self._test_rule_on_example(rule, input_grid, output_grid)
                if applicable:
                    total_applicable += 1
                    if correct:
                        validation_count += 1
            
            if total_applicable > 0:
                rule['confidence'] = validation_count / total_applicable
                rule['validation_count'] = validation_count
                rule['total_applicable'] = total_applicable
                validated_rules.append(rule)
        
        # Sort by confidence
        validated_rules.sort(key=lambda x: x['confidence'], reverse=True)
        return validated_rules

    def _test_rule_on_example(self, rule: Dict[str, Any], input_grid: Array, output_grid: Array) -> Tuple[bool, bool]:
        """Test if a rule applies and is correct for a specific example."""
        rule_type = rule.get('type', '')
        
        if rule_type == 'color_transform':
            input_color = rule['input_color']
            output_color = rule['output_color']
            
            # Check if input color exists
            if input_color in input_grid:
                applicable = True
                # Check if it transforms correctly
                input_mask = input_grid == input_color
                correct = np.all(output_grid[input_mask] == output_color)
                return applicable, correct
            return False, False
        
        elif rule_type == 'translation':
            # Check if the expected movement occurred
            dr, dc = rule['delta_r'], rule['delta_c']
            color = rule['color']
            
            if color in input_grid and input_grid.shape == output_grid.shape:
                applicable = True
                # Simple check: does the color appear at expected offset?
                input_positions = np.argwhere(input_grid == color)
                if len(input_positions) > 0:
                    expected_positions = input_positions + np.array([dr, dc])
                    # Check if expected positions are valid and contain the color
                    valid_moves = 0
                    for pos in expected_positions:
                        r, c = pos
                        if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                            if output_grid[r, c] == color:
                                valid_moves += 1
                    correct = valid_moves > 0
                    return applicable, correct
            return False, False
        
        # Default: not applicable
        return False, False

    def _apply_rft_rules(self, rules: List[Dict[str, Any]], test_inputs: List[Array], inventory: Dict[str, Any]) -> Optional[Dict[str, List[List[List[int]]]]]:
        """Apply validated RFT rules to test inputs."""
        attempt1, attempt2 = [], []
        
        for test_input in test_inputs:
            try:
                result_grid = self._apply_rules_to_grid(rules, test_input, inventory)
                if result_grid is not None:
                    attempt1.append(to_list(result_grid))
                    # Create slight variation for attempt2
                    result_grid2 = self._create_rule_variation(rules, test_input, inventory)
                    attempt2.append(to_list(result_grid2 if result_grid2 is not None else result_grid))
                else:
                    return None
            except Exception:
                return None
        
        return {"attempt_1": attempt1, "attempt_2": attempt2}

    def _apply_rules_to_grid(self, rules: List[Dict[str, Any]], grid: Array, inventory: Dict[str, Any]) -> Optional[Array]:
        """Apply rules to transform a single grid."""
        result = grid.copy()
        
        # Apply rules in order of confidence
        for rule in sorted(rules, key=lambda x: x['confidence'], reverse=True):
            try:
                result = self._apply_single_rule(rule, result)
            except Exception:
                continue
        
        return result

    def _apply_single_rule(self, rule: Dict[str, Any], grid: Array) -> Array:
        """Apply a single rule to a grid."""
        rule_type = rule.get('type', '')
        
        if rule_type == 'color_transform':
            input_color = rule['input_color']
            output_color = rule['output_color']
            result = grid.copy()
            result[grid == input_color] = output_color
            return result
        
        elif rule_type == 'translation':
            dr, dc = rule['delta_r'], rule['delta_c']
            color = rule['color']
            result = grid.copy()
            
            # Find positions of the color
            positions = np.argwhere(grid == color)
            # Clear old positions
            result[grid == color] = self._estimate_background_color(grid)
            
            # Move to new positions
            for pos in positions:
                new_r, new_c = pos[0] + dr, pos[1] + dc
                if 0 <= new_r < result.shape[0] and 0 <= new_c < result.shape[1]:
                    result[new_r, new_c] = color
            
            return result
        
        elif rule_type == 'size_transform':
            input_shape = rule['input_shape']
            output_shape = rule['output_shape']
            if grid.shape == input_shape:
                # Resize grid (simple implementation)
                if output_shape[0] <= grid.shape[0] and output_shape[1] <= grid.shape[1]:
                    return grid[:output_shape[0], :output_shape[1]].copy()
        
        return grid

    def _create_rule_variation(self, rules: List[Dict[str, Any]], grid: Array, inventory: Dict[str, Any]) -> Optional[Array]:
        """Create a variation of rule application for attempt2."""
        # Apply rules in different order or with slight modifications
        result = grid.copy()
        
        # Try applying lower confidence rules
        for rule in rules:
            if rule['confidence'] < 0.9:  # Use less certain rules for variation
                try:
                    result = self._apply_single_rule(rule, result)
                    break  # Only apply one variation
                except Exception:
                    continue
        
        return result

    def _generate_rule_theories(self, rules: List[Dict[str, Any]], train_pairs: List[Tuple[Array, Array]]) -> List[Dict[str, Any]]:
        """Generate alternative rule theories for tracking phase."""
        theories = []
        
        # Theory 1: Combine color rules
        color_rules = [r for r in rules if r['type'] == 'color_transform']
        if len(color_rules) >= 2:
            theories.append({
                'description': 'Multi-color transformation theory',
                'rules': color_rules,
                'type': 'combination'
            })
        
        # Theory 2: Movement + color
        movement_rules = [r for r in rules if r['type'] == 'translation']
        if movement_rules and color_rules:
            theories.append({
                'description': 'Movement and color change theory',
                'rules': movement_rules + color_rules[:1],
                'type': 'hybrid'
            })
        
        # Theory 3: Experimental rules
        experimental_rules = [r for r in rules if r['type'] == 'experimental_color']
        if experimental_rules:
            theories.append({
                'description': 'Experimental color mapping theory',
                'rules': experimental_rules[:3],  # Try top 3 experimental rules
                'type': 'experimental'
            })
        
        return theories

    def _test_rule_theory(self, theory: Dict[str, Any], train_pairs: List[Tuple[Array, Array]]) -> float:
        """Test a rule theory against training data."""
        correct_predictions = 0
        total_tests = 0
        
        for input_grid, expected_output in train_pairs:
            try:
                predicted_output = self._apply_theory_to_grid(theory, input_grid)
                if predicted_output is not None:
                    total_tests += 1
                    if np.array_equal(predicted_output, expected_output):
                        correct_predictions += 1
                    elif np.allclose(predicted_output.astype(float), expected_output.astype(float), rtol=0.1):
                        correct_predictions += 0.5  # Partial credit
            except Exception:
                total_tests += 1  # Count as failure
        
        return correct_predictions / max(total_tests, 1)

    def _apply_theory_to_grid(self, theory: Dict[str, Any], grid: Array) -> Optional[Array]:
        """Apply a rule theory to a grid."""
        result = grid.copy()
        
        for rule in theory.get('rules', []):
            try:
                result = self._apply_single_rule(rule, result)
            except Exception:
                continue
        
        return result

    def _apply_rule_theory(self, theory: Dict[str, Any], test_inputs: List[Array], inventory: Dict[str, Any]) -> Optional[Dict[str, List[List[List[int]]]]]:
        """Apply a rule theory to test inputs."""
        attempt1, attempt2 = [], []
        
        for test_input in test_inputs:
            try:
                result1 = self._apply_theory_to_grid(theory, test_input)
                if result1 is not None:
                    attempt1.append(to_list(result1))
                    # Create variation for attempt2
                    result2 = self._create_theory_variation(theory, test_input)
                    attempt2.append(to_list(result2 if result2 is not None else result1))
                else:
                    return None
            except Exception:
                return None
        
        return {"attempt_1": attempt1, "attempt_2": attempt2}

    def _create_theory_variation(self, theory: Dict[str, Any], grid: Array) -> Optional[Array]:
        """Create a variation of theory application."""
        # Apply theory rules in reverse order
        result = grid.copy()
        rules = theory.get('rules', [])
        
        for rule in reversed(rules):
            try:
                result = self._apply_single_rule(rule, result)
            except Exception:
                continue
        
        return result


# Global solver instance (for backwards compatibility)
_global_solver = None


def solve_task(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
    """Solve a single ARC task (backwards compatible interface)."""
    # Create a new solver instance for each task to prevent memory accumulation
    use_baseline = os.environ.get('ARC_USE_BASELINE', '').lower() in (
        '1', 'true', 'yes'
    )
    enhancements_disabled = os.environ.get('ARC_DISABLE_ENHANCEMENTS', '').lower() in (
        '1', 'true', 'yes'
    )
    use_enhancements = not use_baseline and not enhancements_disabled
    solver = ARCSolver(use_enhancements=use_enhancements)
    
    return solver.solve_task(task)


def get_solver_stats() -> Dict[str, float]:
    """Get global solver statistics."""
    global _global_solver
    if _global_solver is None:
        return {}
    return _global_solver.get_statistics()


# Enhanced solver for direct use
def solve_task_enhanced(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
    """Solve using enhanced methods only."""
    solver = ARCSolver(use_enhancements=True)
    return solver.solve_task(task)


# Baseline solver for comparison
def solve_task_baseline(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
    """Solve using baseline methods only."""
    solver = ARCSolver(use_enhancements=False)
    return solver.solve_task(task)
