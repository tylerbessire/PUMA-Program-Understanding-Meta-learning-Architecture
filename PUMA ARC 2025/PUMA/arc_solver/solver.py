"""Top-level solver interface for ARC tasks with neural enhancements.

This module integrates neural guidance, episodic retrieval, program sketches and
test-time training to provide state-of-the-art solutions for ARC tasks while
maintaining a robust fallback baseline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
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
from puma.rft.feature_flags import RFT_ENABLED, build_limits
from puma.rft.orchestrator import solve_with_rft
from puma.rft.tracking import HypothesisMemory
from puma.rft.demo.grid import build_context as build_rft_grid_context
from puma.rft.explain import explain_last_run
from .enhanced_search import synthesize_with_enhancements, predict_two_enhanced
from .hypothesis import HypothesisEngine, Hypothesis
from .continuous_learning import ContinuousSelfMemory
from .neural.episodic import EpisodicRetrieval
from .failure_learning import FailureLog, ProgramBank
from .placeholders import (
    PlaceholderTemplate,
    PlaceholderTemplateEngine,
    deserialize_placeholder_template,
    serialize_placeholder_template,
)
from .rft_detectors import DetectorOrchestrator, DetectionSignal
from .hypothesis_engine import HypothesisEngine


class ARCSolver:
    """Enhanced ARC solver with neural components and episodic memory."""
    
    def __init__(self, use_enhancements: bool = True,
                 guidance_model_path: str = None,
                 episode_db_path: str = "episodes.json",
                 enable_logging: bool = None,
                 checkpoint_path: str = None,
                 failure_log_path: Optional[str] = None,
                 program_bank_path: Optional[str] = None,
                 llm_config: Optional[Dict[str, Any]] = None,
                 runtime_flags: Optional[Dict[str, Any]] = None):
        self.use_enhancements = use_enhancements
        self.guidance_model_path = guidance_model_path
        self.episode_db_path = episode_db_path
        self.checkpoint_path = checkpoint_path or "checkpoint.json"
        self.hypothesis_engine = HypothesisEngine() # Instantiate the new engine

        # Logging control - check environment variable or parameter
        if enable_logging is None:
            enable_logging = os.environ.get('ARC_ENABLE_LOGGING', 'true').lower() in ('1', 'true', 'yes')
        self.enable_logging = enable_logging

        self.stats = {
            'tasks_solved': 0,
            'total_tasks': 0,
            'enhancement_success_rate': 0.0,
            'fallback_used': 0,
            'program_bank_hits': 0,
            'detector_hints_used': 0,
            'transduction_successes': 0,
            'induction_successes': 0,
        }
        self.submission_results = {}  # For checkpoint saving

        self._llm_config = llm_config
        self._runtime_flags: Dict[str, Any] = dict(runtime_flags) if runtime_flags else {}
        self._runtime_flags.setdefault('rft_mode', 'metadata')
        self._runtime_flags.setdefault('fallback_policy', 'standard')

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
        self._detector_hint: Optional[str] = None
        # Continuous memory and hypotheses
        self.self_memory = ContinuousSelfMemory()
        # self.hypothesis_engine = HypothesisEngine(continuous_memory=self.self_memory) # Old engine is replaced
        self.episodic_retrieval = EpisodicRetrieval(episode_db_path)
        self.placeholder_engine = PlaceholderTemplateEngine()
        self._placeholder_templates: List[PlaceholderTemplate] = []
        self._new_placeholder_templates: List[PlaceholderTemplate] = []
        self._last_hypotheses: List[Hypothesis] = []

        failure_log_path = failure_log_path or os.environ.get('ARC_FAILURE_LOG')
        program_bank_path = program_bank_path or os.environ.get('ARC_PROGRAM_BANK')
        self.failure_log = FailureLog(Path(failure_log_path)) if failure_log_path else None
        self.program_bank = ProgramBank(Path(program_bank_path)) if program_bank_path else None
        if self._runtime_flags.get('disable_rft'):
            self.rft_enabled = False
        elif self._runtime_flags.get('force_rft'):
            self.rft_enabled = True
        else:
            self.rft_enabled = RFT_ENABLED
        self._rft_limits = build_limits()
        self._rft_memory = HypothesisMemory()
        self._last_rft_status: Optional[str] = None

    def solve_task(self, task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
        """Solve a single ARC task using enhanced or baseline methods."""
        self.stats['total_tasks'] += 1
        task_id = str(task.get("task_id") or task.get("id") or f"anonymous_{self.stats['total_tasks']}")

        # --- New Hypothesis Engine ---
        try:
            engine_solution = self.hypothesis_engine.solve(task)
            if engine_solution is not None:
                if self.enable_logging:
                    self.logger.info(f"Task {task_id} solved by HypothesisEngine.")
                solution_list = [to_list(engine_solution)] * len(task.get("test", []))
                return {"attempt_1": solution_list, "attempt_2": solution_list}
        except Exception as e:
            if self.enable_logging:
                self.logger.warning(f"HypothesisEngine failed with error: {e}")
        # --- End New Engine ---

        # Extract training pairs as numpy arrays, skipping malformed ones
        train_pairs: List[Tuple[Array, Array]] = []
        for pair in task.get("train", []):
            try:
                a = to_array(pair["input"])
                b = to_array(pair["output"])
            except Exception:
                continue
            train_pairs.append((a, b))

        self._load_placeholder_templates(train_pairs)
        if not self.use_enhancements:
            self._persist_placeholder_templates(train_pairs)

        # Extract test inputs with graceful degradation
        test_inputs: List[Array] = []
        for pair in task.get("test", []):
            try:
                test_inputs.append(to_array(pair["input"]))
            except Exception:
                test_inputs.append(np.zeros((1, 1), dtype=np.int16))

        self._detector_hint = None
        if train_pairs:
            orchestrator = DetectorOrchestrator()
            fast_signal = orchestrator.scan_example(train_pairs[0][0], train_pairs[0][1])
            if fast_signal is not None:
                self._detector_hint = fast_signal.hypothesis.get("rule_type")
            if fast_signal and fast_signal.confidence > 0.85:
                if self._validate_rft_signal(fast_signal, train_pairs):
                    attempt1, attempt2 = self._apply_rft_signal(fast_signal, test_inputs)
                    if attempt1 and attempt2:
                        self.stats['transduction_successes'] += 1
                        return self._finalize_result(
                            task,
                            task_id,
                            train_pairs,
                            None,
                            test_inputs,
                            attempt1,
                            attempt2,
                            solved_training=True,
                        )

        bank_attempts = self._try_program_bank(task_id, test_inputs)
        if bank_attempts is not None:
            attempt1, attempt2 = bank_attempts
            return self._finalize_result(
                task,
                task_id,
                train_pairs,
                None,
                test_inputs,
                attempt1,
                attempt2,
                solved_training=True,
            )

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
            if self.enable_logging:
                self.logger.info(f"Inconsistent output shapes detected: {output_shapes}, enabling dynamic detection")



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

        # The new HypothesisEngine doesn't use the old solved_training/best_hypothesis variables.
        # We default them to safe values for the finalization step.
        solved_training = False
        best_hypothesis = None
        return self._finalize_result(
            task,
            task_id,
            train_pairs,
            best_hypothesis,
            test_inputs,
            attempt1,
            attempt2,
            solved_training,
        )

    def _get_predictions(
        self, train_pairs: List[Tuple[Array, Array]], test_input: Array, expected_shape: Optional[Tuple[int, int]]
    ) -> List[List[Array]]:
        """Get prediction attempts for a single test input."""
        enhanced: List[List[Array]] = []
        if self.use_enhancements:
            try:
                if self.enable_logging:
                    self.logger.info("Using enhanced search for prediction")
                progs = synthesize_with_enhancements(
                    train_pairs,
                    expected_shape=expected_shape,
                    test_input=test_input,
                    llm_config=self._llm_config,
                )
                
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

    def _try_program_bank(
        self, task_id: str, test_inputs: List[Array]
    ) -> Optional[Tuple[List[List[List[int]]], List[List[List[int]]]]]:
        """Return predictions sourced from the program bank if available."""

        if not self.program_bank:
            return None
        try:
            program = self.program_bank.load(task_id)
        except Exception as exc:
            if self.enable_logging:
                self.logger.debug("Program bank load failed: %s", exc)
            return None
        if not program:
            return None

        attempt1: List[List[List[int]]] = []
        for grid in test_inputs:
            try:
                output = program.apply_to_grid(grid)
            except Exception as exc:
                if self.enable_logging:
                    self.logger.debug("Program bank apply error: %s", exc)
                return None
            attempt1.append(to_list(output))

        attempt2 = [[row[:] for row in output] for output in attempt1]
        self.stats['program_bank_hits'] += 1
        if self.enable_logging:
            self.logger.info("Serving task %%s from program bank", task_id)
        return attempt1, attempt2

    def _maybe_run_rft_adapter(
        self,
        task: Dict[str, Any],
        attempt1: List[List[List[int]]],
        attempt2: List[List[List[int]]],
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Optionally run the RFT orchestrator when the demo flag is present."""

        if not self.rft_enabled:
            return attempt1, attempt2

        metadata = task.get("metadata") or {}
        rft_config = metadata.get("rft_demo")
        if not rft_config or "grid" not in rft_config:
            return attempt1, attempt2
        grid_spec = rft_config["grid"]
        grid = grid_spec.get("grid")
        targets = grid_spec.get("targets", [])
        if not isinstance(grid, list) or not grid:
            return attempt1, attempt2
        try:
            context, rules = build_rft_grid_context(grid, targets, limits=self._rft_limits)
        except Exception:
            return attempt1, attempt2
        context, status = solve_with_rft(
            context,
            rules,
            self._rft_memory,
            outer_budget=self._rft_limits.outer_budget,
        )
        self._last_rft_status = status
        if self.enable_logging:
            self.logger.debug("RFT status=%s\n%s", status, explain_last_run(context))
        if status != "DONE" or not grid_spec.get("inject_output", False):
            return attempt1, attempt2
        solved_grid = context.state.grid
        if solved_grid and isinstance(solved_grid[0][0], str):
            palette: Dict[str, int] = {"black": 0}
            next_id = 1
            for row in solved_grid:
                for color in row:
                    if color not in palette:
                        palette[color] = next_id
                        next_id += 1
            solved_grid = [[palette[color] for color in row] for row in solved_grid]
        return [solved_grid for _ in attempt1], [solved_grid for _ in attempt2]

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

    def _validate_rft_signal(
        self, signal: DetectionSignal, train_pairs: List[Tuple[Array, Array]]
    ) -> bool:
        hypothesis = signal.hypothesis
        rule_type = hypothesis.get("rule_type")

        if rule_type == "rotate_template_match":
            existing = hypothesis.get("templates") or []
            unique_templates: List[Dict[str, Any]] = []
            seen: Set[Tuple[int, bytes]] = set()

            for entry in existing:
                rotation = int(entry.get("rotation", 0))
                template_arr = np.array(entry.get("template"), dtype=int)
                key = (rotation, template_arr.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                unique_templates.append({
                    "rotation": rotation,
                    "template": template_arr.tolist(),
                })

            for inp, expected_out in train_pairs:
                has_template = False
                for entry in unique_templates:
                    template_arr = np.array(entry["template"], dtype=expected_out.dtype)
                    if template_arr.shape == expected_out.shape and np.array_equal(template_arr, expected_out):
                        has_template = True
                        break

                if not has_template:
                    generated = self._generate_templates_for_pair(inp, expected_out)
                    if not generated:
                        return False
                    for entry in generated:
                        rotation = int(entry.get("rotation", 0))
                        template_arr = np.array(entry.get("template"), dtype=int)
                        key = (rotation, template_arr.tobytes())
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_templates.append({
                            "rotation": rotation,
                            "template": template_arr.tolist(),
                        })

            hypothesis["templates"] = unique_templates
            return True

        height = hypothesis.get("height")
        width = hypothesis.get("width")

        if height is None or width is None:
            return False

        height = int(height)
        width = int(width)

        rotation_candidates = hypothesis.get("rotation_candidates")
        legacy_rotation = hypothesis.get("rotation_degrees")
        if rotation_candidates is None:
            if legacy_rotation is None:
                return False
            rotation_candidates = [int(legacy_rotation)]

        valid_rotations = set(int(deg) for deg in rotation_candidates)
        working_rotations: Set[int] = set()

        for inp, expected_out in train_pairs:
            matches = self._find_rotation_matches(
                inp,
                height,
                width,
                valid_rotations,
                target_grid=expected_out,
            )
            if not matches:
                return False
            working_rotations |= {deg for deg, *_ in matches}

        if not working_rotations:
            return False

        hypothesis["rotation_candidates"] = sorted(working_rotations)
        hypothesis.pop("rotation_matches", None)
        return True

    def _apply_rft_signal(
        self, signal: DetectionSignal, test_inputs: List[Array]
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        attempt1: List[List[List[int]]] = []
        attempt2: List[List[List[int]]] = []

        for test_input in test_inputs:
            output = self._apply_rft_rule(test_input, signal.hypothesis)
            if output is None:
                return [], []
            output_list = to_list(output)
            attempt1.append(output_list)
            attempt2.append(output_list)

        return attempt1, attempt2

    def _apply_rft_rule(
        self,
        grid: Array,
        hypothesis: Dict[str, Any],
        target_grid: Optional[Array] = None,
    ) -> Optional[Array]:
        rule_type = hypothesis.get("rule_type")

        if rule_type == "expand_rare_color":
            color = hypothesis.get("source_color")
            if color is None:
                return None
            return np.full_like(grid, color)

        if rule_type == "color_count_fill":
            color = hypothesis.get("color")
            if color is None:
                return None
            return np.full_like(grid, color)

        if rule_type == "color_swap":
            result = grid.copy()
            for c1, c2 in hypothesis.get("swap_pairs", []):
                mask1 = grid == c1
                mask2 = grid == c2
                result[mask1] = c2
                result[mask2] = c1
            return result

        if rule_type == "tile_grid":
            factor = hypothesis.get("tile_factor", 1)
            if isinstance(factor, tuple):
                fh, fw = factor
            else:
                fh = fw = int(factor)
            if fh <= 0 or fw <= 0:
                return None
            return np.tile(grid, (fh, fw))

        if rule_type == "crop_grid":
            crop_h, crop_w = hypothesis.get("crop_factor", (1, 1))
            if crop_h <= 0 or crop_w <= 0:
                return None
            h, w = grid.shape
            return grid[: max(1, h // crop_h), : max(1, w // crop_w)]

        if rule_type == "crop_subgrid":
            top = hypothesis.get("top")
            left = hypothesis.get("left")
            height = hypothesis.get("height")
            width = hypothesis.get("width")
            if None in (top, left, height, width):
                return None
            top = int(top)
            left = int(left)
            height = int(height)
            width = int(width)
            h, w = grid.shape
            if top < 0 or left < 0 or top + height > h or left + width > w:
                return None
            return grid[top : top + height, left : left + width].copy()

        if rule_type == "remove_colors_and_crop":
            removals = hypothesis.get("removals", [])
            removable = list(int(v) for v in removals)

            pruned = grid.copy()
            for color in removable:
                pruned[pruned == color] = hypothesis.get("fill_color", 0)

            top = hypothesis.get("top")
            left = hypothesis.get("left")
            height = hypothesis.get("height")
            width = hypothesis.get("width")
            if None in (top, left, height, width):
                return None
            top = int(top)
            left = int(left)
            height = int(height)
            width = int(width)
            h, w = pruned.shape
            if top < 0 or left < 0 or top + height > h or left + width > w:
                return None

            cropped = pruned[top : top + height, left : left + width]
            return cropped

        if rule_type == "rotate_template_match":
            templates = hypothesis.get("templates", [])
            if not templates:
                return None

            if target_grid is not None:
                matches = self._find_template_matches(grid, templates, target_grid=target_grid)
                return matches[0][3] if matches else None

            matches = self._find_template_matches(grid, templates)
            if matches:
                return matches[0][3]

            return None

        if rule_type == "rotate_extract_adjacent_to_8":
            rotation = int(hypothesis.get("rotation", 0))
            direction = hypothesis.get("direction")
            height = hypothesis.get("height")
            width = hypothesis.get("width")
            row_offset = int(hypothesis.get("row_offset", 0))
            col_offset = int(hypothesis.get("col_offset", 0))

            if direction is None or height is None or width is None:
                return None

            height = int(height)
            width = int(width)

            rotated = np.rot90(grid, (rotation // 90) % 4) if rotation else grid
            coords = np.argwhere(rotated == 8)
            if coords.size == 0:
                return None

            r_min = int(coords[:, 0].min())
            r_max = int(coords[:, 0].max())
            c_min = int(coords[:, 1].min())
            c_max = int(coords[:, 1].max())

            if direction == "right":
                top = r_min + row_offset
                left = c_max + 1 + col_offset
            elif direction == "left":
                top = r_min + row_offset
                left = c_min - width + col_offset
            elif direction == "below":
                top = r_max + 1 + row_offset
                left = c_min + col_offset
            elif direction == "above":
                top = r_min - height + row_offset
                left = c_min + col_offset
            else:
                return None

            if top < 0 or left < 0:
                return None
            if top + height > rotated.shape[0] or left + width > rotated.shape[1]:
                return None

            cropped = rotated[top : top + height, left : left + width].copy()
            return cropped

        if rule_type == "mirror":
            axis = hypothesis.get("axis")
            if axis == "vertical":
                return np.fliplr(grid)
            if axis == "horizontal":
                return np.flipud(grid)
            return None

        if rule_type == "translate_objects":
            vector = hypothesis.get("vector")
            fill = hypothesis.get("background", 0)
            if vector is None:
                return None
            return self._apply_translation(grid, tuple(vector), fill)

        if rule_type == "gravity_drop":
            vector = hypothesis.get("vector")
            fill = hypothesis.get("background", 0)
            if vector is None:
                return None
            return self._apply_translation(grid, tuple(vector), fill)

        if rule_type == "change_background":
            from_color = hypothesis.get("from_color")
            to_color = hypothesis.get("to_color")
            if from_color is None or to_color is None:
                return None
            result = grid.copy()
            result[grid == from_color] = to_color
            return result

        if rule_type == "repeat_pattern":
            pattern_size = hypothesis.get("pattern_size")
            if not pattern_size:
                return None
            ph, pw = pattern_size
            if ph <= 0 or pw <= 0:
                return None
            if grid.shape[0] % ph != 0 or grid.shape[1] % pw != 0:
                return None
            return np.tile(grid[:ph, :pw], (grid.shape[0] // ph, grid.shape[1] // pw))

        if rule_type == "add_border":
            color = hypothesis.get("border_color", 0)
            thickness = int(hypothesis.get("thickness", 1))
            if thickness <= 0:
                return None
            h, w = grid.shape
            result = np.full((h + 2 * thickness, w + 2 * thickness), color, dtype=grid.dtype)
            result[thickness:thickness + h, thickness:thickness + w] = grid
            return result

        if rule_type == "remove_border":
            thickness = int(hypothesis.get("thickness", 1))
            if thickness <= 0:
                return None
            h, w = grid.shape
            if h <= 2 * thickness or w <= 2 * thickness:
                return None
            return grid[thickness:-thickness, thickness:-thickness]

        if rule_type == "encode_color_counts":
            ordering = hypothesis.get("ordering")
            orientation = hypothesis.get("orientation", "row")
            if not ordering:
                return None
            counts: List[int] = []
            for color in ordering:
                occurrences = int(np.sum(grid == color))
                counts.extend([int(color)] * occurrences)
            if orientation == "column":
                return np.array(counts, dtype=grid.dtype).reshape(-1, 1)
            return np.array([counts], dtype=grid.dtype)

        if rule_type == "draw_diagonal":
            color = hypothesis.get("color")
            if color is None:
                return None
            h, w = grid.shape
            result = grid.copy()
            diag_len = min(h, w)
            for i in range(diag_len):
                result[i, i] = color
            return result

        return None

    def _find_rotation_matches(
        self,
        grid: Array,
        height: int,
        width: int,
        rotations: Iterable[int],
        target_grid: Optional[Array] = None,
    ) -> List[Tuple[int, int, int, Array]]:
        matches: List[Tuple[int, int, int, Array]] = []
        for rotation_degrees in rotations:
            k = (rotation_degrees // 90) % 4
            rotated_grid = np.rot90(grid, k) if k else grid
            rh, rw = rotated_grid.shape
            if height > rh or width > rw:
                continue

            for top in range(rh - height + 1):
                for left in range(rw - width + 1):
                    window = rotated_grid[top : top + height, left : left + width]
                    if target_grid is not None:
                        if np.array_equal(window, target_grid):
                            return [(rotation_degrees, top, left, window.copy())]
                    else:
                        matches.append((rotation_degrees, top, left, window.copy()))

        if target_grid is not None:
            return []

        return matches

    def _find_template_matches(
        self,
        grid: Array,
        templates: List[Dict[str, Any]],
        target_grid: Optional[Array] = None,
    ) -> List[Tuple[int, Optional[int], Optional[int], Array]]:
        matches: List[Tuple[int, Optional[int], Optional[int], Array]] = []

        for entry in templates:
            rotation = int(entry.get("rotation", 0))
            template_list = entry.get("template")
            if template_list is None:
                continue

            template_arr = np.array(template_list, dtype=grid.dtype)

            if target_grid is not None:
                if template_arr.shape == target_grid.shape and np.array_equal(template_arr, target_grid):
                    matches.append((rotation, None, None, template_arr.copy()))
                continue

            k = (rotation // 90) % 4
            rotated = np.rot90(grid, k) if rotation else grid
            rh, rw = rotated.shape
            oh, ow = template_arr.shape
            if oh > rh or ow > rw:
                continue

            for top in range(rh - oh + 1):
                window = rotated[top : top + oh]
                for left in range(rw - ow + 1):
                    candidate = window[:, left : left + ow]
                    if np.array_equal(candidate, template_arr):
                        matches.append((rotation, top, left, template_arr.copy()))

        return matches

    def _generate_templates_for_pair(
        self,
        grid: Array,
        output: Array,
    ) -> List[Dict[str, Any]]:
        oh, ow = output.shape
        gh, gw = grid.shape
        if oh == 0 or ow == 0 or oh > gh or ow > gw:
            return []

        templates: List[Dict[str, Any]] = []
        seen: Set[Tuple[int, bytes]] = set()

        for degrees, k in ((0, 0), (90, 1), (180, 2), (270, 3)):
            rotated = np.rot90(grid, k) if k else grid
            rh, rw = rotated.shape
            if oh > rh or ow > rw:
                continue

            for top in range(rh - oh + 1):
                window = rotated[top : top + oh]
                for left in range(rw - ow + 1):
                    candidate = window[:, left : left + ow]
                    if np.array_equal(candidate, output):
                        key = (degrees, output.tobytes())
                        if key in seen:
                            continue
                        seen.add(key)
                        templates.append(
                            {
                                "rotation": degrees,
                                "template": output.tolist(),
                            }
                        )
                        break
                else:
                    continue
                break

        return templates

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
            programs = synthesize_with_enhancements(
                train_pairs,
                force_alt=True,
                test_input=test_inputs[0] if test_inputs else None,
                expected_shape=None,
                llm_config=self._llm_config,
            )
            
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

    def _finalize_result(
        self,
        task: Dict[str, Any],
        task_id: str,
        train_pairs: List[Tuple[Array, Array]],
        best_hypothesis: Optional[Hypothesis],
        test_inputs: List[Array],
        attempt1: List[List[List[int]]],
        attempt2: List[List[List[int]]],
        solved_training: bool,
    ) -> Dict[str, List[List[List[int]]]]:
        """Run final adapters, update bookkeeping, and log failures."""

        if self.rft_enabled:
            attempt1, attempt2 = self._maybe_run_rft_adapter(task, attempt1, attempt2)
        result = {"attempt_1": attempt1, "attempt_2": attempt2, "predictions": attempt1}
        self._record_continuous_experience(
            task_id, train_pairs, best_hypothesis, solved_training, result
        )
        if solved_training:
            self.stats['tasks_solved'] += 1
        self._handle_outcome(
            task_id=task_id,
            test_inputs=test_inputs,
            attempt1=attempt1,
            attempt2=attempt2,
            solved_training=solved_training,
            hypothesis_confidence=best_hypothesis.confidence if best_hypothesis else 0.0,
        )
        return result

    def _handle_outcome(
        self,
        *,
        task_id: str,
        test_inputs: List[Array],
        attempt1: List[List[List[int]]],
        attempt2: List[List[List[int]]],
        solved_training: bool,
        hypothesis_confidence: float,
    ) -> None:
        """Persist failure details when we cannot verify the training data."""

        if solved_training or not self.failure_log:
            return

        base_metadata = {
            "confidence": float(hypothesis_confidence),
            "enhancements": bool(self.use_enhancements),
        }
        attempts = (attempt1, attempt2)
        for attempt_idx, outputs in enumerate(attempts, start=1):
            for test_index, (input_grid, output_grid) in enumerate(zip(test_inputs, outputs)):
                try:
                    output_array = np.asarray(output_grid, dtype=input_grid.dtype)
                    metadata = dict(base_metadata)
                    metadata.update({
                        "attempt": attempt_idx,
                        "test_index": test_index,
                    })
                    self.failure_log.record(
                        task_id=task_id,
                        attempt=attempt_idx,
                        input_grid=input_grid,
                        output_grid=output_array,
                        reason="unsolved_training",
                        metadata=metadata,
                    )
                except Exception as exc:
                    if self.enable_logging:
                        self.logger.debug("Failure logging aborted: %s", exc)
                    return

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
