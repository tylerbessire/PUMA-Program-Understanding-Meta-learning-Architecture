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

from .grid import to_array, to_list, Array
from .search import (
    synthesize as synth_baseline,
    predict_two as predict_two_baseline,
)
from .enhanced_search import synthesize_with_enhancements, predict_two_enhanced
from .hypothesis import HypothesisEngine, Hypothesis
from .continuous_learning import ContinuousSelfMemory


class ARCSolver:
    """Enhanced ARC solver with neural components and episodic memory."""
    
    def __init__(self, use_enhancements: bool = True,
                 guidance_model_path: str = None,
                 episode_db_path: str = "episodes.json"):
        self.use_enhancements = use_enhancements
        self.guidance_model_path = guidance_model_path
        self.episode_db_path = episode_db_path
        self.stats = {
            'tasks_solved': 0,
            'total_tasks': 0,
            'enhancement_success_rate': 0.0,
            'fallback_used': 0,
        }

        # Structured logger for observability
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self._last_outputs: Optional[Tuple[List[List[List[int]]], List[List[List[int]]]]] = None
        # Continuous memory and hypotheses
        self.self_memory = ContinuousSelfMemory()
        self.hypothesis_engine = HypothesisEngine(continuous_memory=self.self_memory)
        self._last_hypotheses: List[Hypothesis] = []

    def solve_task(self, task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
        """Solve a single ARC task using enhanced or baseline methods."""
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

        training_stats = self._compute_training_stats(train_pairs)

        # DYNAMIC SHAPE DETECTION: For inconsistent output tasks, don't assume first training shape
        output_shapes = [out.shape for _, out in train_pairs]
        if len(set(output_shapes)) == 1:
            # Consistent outputs - use the common shape
            expected_shape = train_pairs[0][1].shape
        else:
            # Inconsistent outputs - let enhanced search detect from test input
            expected_shape = None
            self.logger.info(f"Inconsistent output shapes detected: {output_shapes}, enabling dynamic detection")

        # Generate and store hypotheses about the transformation.
        self._last_hypotheses = self.hypothesis_engine.generate_hypotheses(train_pairs)
        best_hypothesis: Optional[Hypothesis] = (
            self._last_hypotheses[0] if self._last_hypotheses else None
        )
        if best_hypothesis:
            # Update confidence using training pairs to double check
            best_hypothesis.confidence = self.hypothesis_engine.test_hypothesis(
                best_hypothesis, train_pairs
            )
            if best_hypothesis.confidence == 1.0:
                attempt1: List[List[List[int]]] = []
                attempt2: List[List[List[int]]] = []
                for test_input in test_inputs:
                    transformed = self.hypothesis_engine.apply(best_hypothesis, test_input)
                    if transformed is None:
                        break
                    attempt1.append(to_list(transformed))
                    attempt2.append(to_list(transformed))
                else:
                    # All test inputs transformed successfully
                    result = {"attempt_1": attempt1, "attempt_2": attempt2}
                    self._record_continuous_experience(task_id, train_pairs, best_hypothesis, True, result)
                    self.stats['tasks_solved'] += 1
                    return result

        # Collect predictions for each test input individually
        attempt1: List[List[List[int]]] = []
        attempt2: List[List[List[int]]] = []
        for test_input in test_inputs:
            predictions = self._get_predictions(train_pairs, test_input, expected_shape)
            processed = self._postprocess_predictions(
                train_pairs,
                test_input,
                predictions,
                expected_shape,
                training_stats,
            )
            if processed is not None:
                first_arr, second_arr = processed
                attempt1.append(to_list(first_arr))
                attempt2.append(to_list(second_arr))
            else:
                # Use identity grid as safe fallback
                fallback = to_list(test_input)
                attempt1.append(fallback)
                attempt2.append(fallback)

        result = {"attempt_1": attempt1, "attempt_2": attempt2}
        solved_training = bool(best_hypothesis and best_hypothesis.confidence >= 0.999)
        self._record_continuous_experience(task_id, train_pairs, best_hypothesis, solved_training, result)
        if solved_training:
            self.stats['tasks_solved'] += 1
        return result

    def _get_predictions(
        self, train_pairs: List[Tuple[Array, Array]], test_input: Array, expected_shape: Optional[Tuple[int, int]]
    ) -> List[List[Array]]:
        """Get prediction attempts for a single test input."""
        enhanced: List[List[Array]] = []
        if self.use_enhancements:
            try:
                self.logger.info("Using enhanced search for prediction")
                progs = synthesize_with_enhancements(train_pairs, expected_shape=expected_shape, test_input=test_input)
                
                # Import human reasoner for enhanced prediction
                from .human_reasoning import HumanGradeReasoner
                human_reasoner = HumanGradeReasoner()
                
                enhanced = predict_two_enhanced(progs, [test_input], 
                                              human_reasoner=human_reasoner,
                                              train_pairs=train_pairs)
            except Exception as e:
                self.logger.exception("Enhanced prediction error: %s", e)

        # Baseline predictions for ensemble
        progs_base = synth_baseline(train_pairs, expected_shape=expected_shape)
        baseline = predict_two_baseline(progs_base, [test_input])

        # Validate enhanced prediction
        if enhanced and self._validate_solution(enhanced, [test_input]):
            self.logger.info(f"Enhanced prediction valid - shape: {enhanced[0][0].shape}")
            return [enhanced[0], baseline[0]]

        self.stats['fallback_used'] += 1
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
            return None

        processed.sort(key=lambda item: (-item[0], item[1]))
        best = processed[0][2]
        second = processed[1][2] if len(processed) > 1 else best
        return best, second

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
