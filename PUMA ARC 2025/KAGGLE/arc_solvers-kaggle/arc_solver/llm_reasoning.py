"""
LLM-based reasoning integration for ARC solver.
Provides high-level reasoning over all solver tools and approaches.
"""

from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
import json
import logging
import os
import numpy as np
from dataclasses import dataclass, field, asdict

from .grid import Array, to_list

if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from a text response."""

    if not text:
        return None

    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        candidates = [p for p in parts if p.strip() and not p.strip().lower().startswith("json")]
        if candidates:
            text = candidates[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


@dataclass
class LLMReasoning:
    """Structure for LLM reasoning results."""
    analysis: str
    suggested_approach: str
    confidence: float
    reasoning_steps: List[str]
    tool_recommendations: List[str]


@dataclass
class LLMStageAdvice:
    """LLM guidance for a specific pipeline stage."""

    stage: str
    action: str
    reason: str
    confidence: float = 0.6
    notes: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


class LLMReasoningEngine:
    """LLM-powered reasoning engine for ARC task analysis and solution guidance."""
    
    def __init__(self, enable_logging: bool = True, llm_client: Optional['LLMClient'] = None):
        self.enable_logging = enable_logging
        self.reasoning_history: List[Dict[str, Any]] = []
        self._llm_client = llm_client if llm_client is not None else self._maybe_build_llm_client()
        self._conversation_messages: List[Dict[str, str]] = []
        self._stage_log: List[Dict[str, Any]] = []
        self._stage_cache: Dict[str, LLMStageAdvice] = {}
        self._latest_reasoning: Optional[LLMReasoning] = None
        self._json_extractor = _extract_json_from_text
        
    def _maybe_build_llm_client(self) -> Optional['LLMClient']:
        """Lazily construct an LLM client if the environment requests it."""

        provider = os.environ.get("PUMA_LLM_PROVIDER", "").strip().lower()
        if not provider:
            return None

        try:
            from . import llm_client as llm_support  # Local import to avoid heavy deps by default
        except Exception as exc:  # pragma: no cover - import failures are logged
            if self.enable_logging:
                logger.warning("Failed to import LLM support module: %s", exc)
            return None

        try:
            client = llm_support.build_llm_client_from_env()
        except Exception as exc:  # pragma: no cover - provider configuration errors
            if self.enable_logging:
                logger.warning("Failed to configure LLM client: %s", exc)
            return None

        extractor = getattr(llm_support, "extract_json_from_text", None)
        if callable(extractor):
            self._json_extractor = extractor  # Use provider-aware extractor if available

        return client

    def analyze_task(self, train_pairs: List[Tuple[Array, Array]], 
                    test_inputs: List[Array],
                    object_inventory: Dict[str, Any] = None,
                    rft_rules: List[Dict[str, Any]] = None) -> LLMReasoning:
        """Perform high-level LLM analysis of the task."""
        
        # Build structured task description for LLM reasoning
        task_description = self._build_task_description(
            train_pairs, test_inputs, object_inventory, rft_rules
        )

        reasoning_result = self._analyze_with_llm(task_description)
        
        # Store reasoning history
        self.reasoning_history.append({
            'task_description': task_description,
            'reasoning': reasoning_result,
            'timestamp': self._get_timestamp()
        })

        self._latest_reasoning = reasoning_result
        
        return reasoning_result
    
    def _build_task_description(self, train_pairs: List[Tuple[Array, Array]], 
                               test_inputs: List[Array],
                               object_inventory: Dict[str, Any] = None,
                               rft_rules: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build structured description for LLM reasoning."""
        
        description = {
            'task_stats': {
                'num_training_examples': len(train_pairs),
                'num_test_inputs': len(test_inputs),
                'input_shapes': [inp.shape for inp, _ in train_pairs],
                'output_shapes': [out.shape for _, out in train_pairs],
                'test_shapes': [test.shape for test in test_inputs]
            },
            'patterns': {
                'consistent_input_shapes': len(set(inp.shape for inp, _ in train_pairs)) == 1,
                'consistent_output_shapes': len(set(out.shape for _, out in train_pairs)) == 1,
                'size_changes': self._analyze_size_changes(train_pairs),
                'color_changes': self._analyze_color_changes(train_pairs)
            },
            'object_inventory': object_inventory or {},
            'rft_rules': rft_rules or [],
            'complexity_indicators': self._assess_complexity(train_pairs)
        }

        return description

    def _analysis_system_prompt(self) -> str:
        """System prompt describing the ARC reasoning role."""

        return (
            "You are the reasoning orchestrator for the PUMA ARC solver. "
            "Use relational frame theory terminology: pliance (rule following), "
            "tracking (rule adaptation), and testing of hypotheses. "
            "You receive summaries of ARC tasks and heuristic analyses. "
            "Reply with a single JSON object containing:\n"
            "{\n"
            "  \"analysis\": string,\n"
            "  \"suggested_approach\": string in ['rft_direct_application','rft_pliance_first','enhanced_search_primary','tracking_priority','baseline_only'],\n"
            "  \"confidence\": number 0-1,\n"
            "  \"reasoning_steps\": list of strings,\n"
            "  \"tool_recommendations\": list of strings (e.g. 'rft_pliance','rft_tracking','hypothesis_engine','enhanced_search'),\n"
            "  \"stage_advice\": optional list of {\"stage\": string, \"action\": string, \"reason\": string, \"confidence\": number, \"notes\": list of strings}\n"
            "}.\n"
            "Stage values should be 'pliance', 'tracking', 'hypothesis', or 'global'. "
            "Do not include free-form text outside the JSON."
        )

    def _json_default(self, value: Any) -> Any:
        """JSON serializer fallback for numpy/scalar types."""

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:  # pragma: no cover - defensive fallback
                return str(value)
        if isinstance(value, set):
            return sorted(self._json_default(v) for v in value)
        if isinstance(value, tuple):
            return [self._json_default(v) for v in value]
        if isinstance(value, dict):
            return {k: self._json_default(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_default(v) for v in value]
        return value

    def _format_analysis_request(
        self,
        task_description: Dict[str, Any],
        heuristic_reasoning: LLMReasoning,
    ) -> str:
        """Format the analysis payload for the LLM."""

        payload = {
            "task_summary": self._json_default(task_description),
            "heuristic_reasoning": {
                "analysis": heuristic_reasoning.analysis,
                "suggested_approach": heuristic_reasoning.suggested_approach,
                "confidence": heuristic_reasoning.confidence,
                "reasoning_steps": heuristic_reasoning.reasoning_steps,
                "tool_recommendations": heuristic_reasoning.tool_recommendations,
            },
        }

        try:
            return json.dumps(payload, default=self._json_default, indent=2)
        except TypeError:
            # As a last resort, round-trip through sanitised dict
            sanitized = self._json_default(payload)
            return json.dumps(sanitized, indent=2)

    def _analyze_with_llm(self, task_description: Dict[str, Any]) -> LLMReasoning:
        """Combine heuristic reasoning with optional LLM augmentation."""

        heuristic = self._generate_reasoning(task_description)

        if self._llm_client is None:
            if self.enable_logging:
                logger.debug("LLM client unavailable; using heuristic reasoning only")
            self._conversation_messages = []
            return heuristic

        system_prompt = self._analysis_system_prompt()
        user_payload = self._format_analysis_request(task_description, heuristic)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

        try:
            response_text = self._llm_client.complete(messages, response_format="json")
        except Exception as exc:  # pragma: no cover - provider/runtime errors
            if self.enable_logging:
                logger.warning("LLM analysis failed (%s); reverting to heuristic reasoning", exc)
            self._llm_client = None  # Avoid repeated failures this run
            self._conversation_messages = []
            return heuristic

        parsed = self._json_extractor(response_text)
        enriched = self._reasoning_from_json(parsed, heuristic)

        # Maintain conversation history for subsequent stage advice calls
        self._conversation_messages = messages + [
            {"role": "assistant", "content": response_text}
        ]

        return enriched

    def _reasoning_from_json(
        self,
        payload: Optional[Dict[str, Any]],
        fallback: LLMReasoning,
    ) -> LLMReasoning:
        """Convert parsed LLM JSON into an LLMReasoning object."""

        if not isinstance(payload, dict):
            if self.enable_logging:
                logger.debug("LLM response missing JSON payload; using heuristic reasoning")
            return fallback

        analysis = str(payload.get("analysis", fallback.analysis))
        suggested = str(payload.get("suggested_approach", fallback.suggested_approach)).strip()
        if not suggested:
            suggested = fallback.suggested_approach

        try:
            confidence = float(payload.get("confidence", fallback.confidence))
        except (TypeError, ValueError):
            confidence = fallback.confidence
        confidence = max(0.0, min(1.0, confidence))

        steps_raw = payload.get("reasoning_steps", fallback.reasoning_steps)
        if isinstance(steps_raw, list):
            reasoning_steps = [str(step) for step in steps_raw]
        else:
            reasoning_steps = fallback.reasoning_steps

        tools_raw = payload.get("tool_recommendations", fallback.tool_recommendations)
        if isinstance(tools_raw, list):
            tool_recommendations = [str(tool) for tool in tools_raw]
        else:
            tool_recommendations = fallback.tool_recommendations

        stage_entries = payload.get("stage_advice", [])
        if isinstance(stage_entries, dict):
            stage_entries = [stage_entries]

        if isinstance(stage_entries, list):
            for entry in stage_entries:
                advice = self._stage_advice_from_json(entry, default_stage="global", baseline=None)
                if advice:
                    self._stage_cache[advice.stage] = advice
                    self._stage_log.append({
                        "stage": advice.stage,
                        "source": "llm_initial",
                        "advice": advice,
                    })

        enriched = LLMReasoning(
            analysis=analysis,
            suggested_approach=suggested,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            tool_recommendations=tool_recommendations,
        )

        return enriched

    def _format_stage_request(
        self,
        stage: str,
        context: Optional[Dict[str, Any]],
        baseline: LLMStageAdvice,
    ) -> str:
        """Format a follow-up request for stage-specific advice."""

        payload = {
            "stage": stage,
            "baseline_advice": self._json_default(asdict(baseline)),
            "context": self._json_default(context or {}),
        }

        if self._latest_reasoning:
            payload["latest_reasoning"] = {
                "analysis": self._latest_reasoning.analysis,
                "suggested_approach": self._latest_reasoning.suggested_approach,
                "confidence": self._latest_reasoning.confidence,
                "tool_recommendations": self._latest_reasoning.tool_recommendations,
            }

        try:
            return json.dumps(payload, default=self._json_default, indent=2)
        except TypeError:
            sanitized = self._json_default(payload)
            return json.dumps(sanitized, indent=2)

    def _stage_advice_from_json(
        self,
        entry: Optional[Dict[str, Any]],
        default_stage: str,
        baseline: Optional[LLMStageAdvice],
    ) -> Optional[LLMStageAdvice]:
        """Parse stage-level advice from LLM output."""

        if not isinstance(entry, dict):
            return baseline

        stage = str(entry.get("stage", (baseline.stage if baseline else default_stage))).lower()
        action = str(entry.get("action", baseline.action if baseline else "continue_current_approach")).strip()
        reason = str(entry.get("reason", baseline.reason if baseline else "Follow heuristic plan.")).strip()

        try:
            confidence = float(entry.get("confidence", baseline.confidence if baseline else 0.6))
        except (TypeError, ValueError):
            confidence = baseline.confidence if baseline else 0.6
        confidence = max(0.0, min(1.0, confidence))

        notes_raw = entry.get("notes", baseline.notes if baseline else [])
        notes: List[str]
        if isinstance(notes_raw, list):
            notes = [str(item) for item in notes_raw]
        elif notes_raw:
            notes = [str(notes_raw)]
        else:
            notes = []

        return LLMStageAdvice(
            stage=stage,
            action=action or (baseline.action if baseline else "continue_current_approach"),
            reason=reason or (baseline.reason if baseline else "Heuristic guidance"),
            confidence=confidence,
            notes=notes,
            raw=entry,
        )

    def _default_stage_advice(
        self,
        stage: str,
        context: Optional[Dict[str, Any]],
    ) -> LLMStageAdvice:
        """Fallback heuristic advice for solver stages."""

        context = context or {}
        failed = set(context.get("failed_approaches", []) or [])
        stage_lower = stage.lower()

        if stage_lower == "pliance":
            has_rules = bool(context.get("high_confidence_rules", True))
            if has_rules:
                action = "apply_high_confidence_rules"
                reason = "Deploy high-confidence relational rules across test inputs"
                confidence = 0.75
            else:
                action = "expand_object_inventory"
                reason = "No confident rules detected; expand inventory and re-evaluate"
                confidence = 0.55
        elif stage_lower == "tracking":
            theories = context.get("candidate_theories", [])
            if theories:
                action = "evaluate_rule_theories"
                reason = "Test synthesized rule theories against training examples"
                confidence = 0.65
            else:
                action = "fallback_to_enhanced_search"
                reason = "No viable theories; consider enhanced search fallback"
                confidence = 0.5
        elif stage_lower == "hypothesis":
            pending = context.get("pending_hypotheses", 0)
            if pending:
                action = "score_hypotheses"
                reason = "Score remaining hypotheses using guidance metrics"
                confidence = 0.6
            else:
                action = "generate_new_hypotheses"
                reason = "No hypotheses remain; trigger new generation cycle"
                confidence = 0.55
        else:  # global / default
            if 'rft_pliance' in failed:
                if 'rft_tracking' not in failed:
                    action = "switch_to_rft_tracking"
                    reason = "Pliance failed; escalate to adaptive tracking"
                    confidence = 0.7
                else:
                    action = "fallback_to_enhanced_search"
                    reason = "RFT approaches exhausted; use enhanced solver"
                    confidence = 0.65
            elif 'enhanced_search' in failed:
                action = "activate_baseline_fallback"
                reason = "Enhanced search failed; rely on baseline predictions"
                confidence = 0.55
            else:
                action = "continue_current_approach"
                reason = "Current pipeline still viable"
                confidence = 0.6

        return LLMStageAdvice(
            stage=stage_lower or "global",
            action=action,
            reason=reason,
            confidence=confidence,
            notes=[],
            raw={"source": "heuristic"},
        )

    def advise_on_stage_progress(
        self,
        stage: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LLMStageAdvice:
        """Blend heuristic and LLM advice for a specific solver stage."""

        stage_lower = (stage or "global").lower()
        baseline = self._default_stage_advice(stage_lower, context)

        if self._llm_client is None:
            self._stage_cache[stage_lower] = baseline
            self._stage_log.append({
                "stage": stage_lower,
                "source": "heuristic_only",
                "advice": baseline,
            })
            return baseline

        user_payload = self._format_stage_request(stage_lower, context, baseline)

        if not self._conversation_messages:
            # Restart conversation with system prompt if analyze_task was not called
            self._conversation_messages = [{"role": "system", "content": self._analysis_system_prompt()}]

        messages = self._conversation_messages + [{"role": "user", "content": user_payload}]

        try:
            response_text = self._llm_client.complete(messages, response_format="json")
        except Exception as exc:  # pragma: no cover - provider/runtime errors
            if self.enable_logging:
                logger.warning("LLM stage advice failed (%s); using heuristic guidance", exc)
            self._llm_client = None
            self._stage_cache[stage_lower] = baseline
            self._stage_log.append({
                "stage": stage_lower,
                "source": "heuristic_fallback",
                "advice": baseline,
            })
            return baseline

        parsed = self._json_extractor(response_text)
        advice = None

        if isinstance(parsed, dict):
            stage_entries = parsed.get("stage_advice")
            if isinstance(stage_entries, list):
                for entry in stage_entries:
                    advice = self._stage_advice_from_json(entry, stage_lower, baseline)
                    if advice and advice.stage == stage_lower:
                        break
            elif stage_entries is not None:
                advice = self._stage_advice_from_json(stage_entries, stage_lower, baseline)

            if advice is None:
                advice = self._stage_advice_from_json(parsed, stage_lower, baseline)

        if advice is None:
            advice = baseline

        self._conversation_messages = messages + [{"role": "assistant", "content": response_text}]
        self._stage_cache[stage_lower] = advice
        self._stage_log.append({
            "stage": stage_lower,
            "source": "llm_stage",
            "baseline": baseline,
            "advice": advice,
        })

        return advice

    def _analyze_size_changes(self, train_pairs: List[Tuple[Array, Array]]) -> Dict[str, Any]:
        """Analyze size transformation patterns."""
        size_changes = []
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                size_changes.append({
                    'from': inp.shape,
                    'to': out.shape,
                    'type': self._classify_size_change(inp.shape, out.shape)
                })
        
        return {
            'has_size_changes': len(size_changes) > 0,
            'changes': size_changes,
            'consistent_size_change': len(set(str(change) for change in size_changes)) <= 1 if size_changes else True
        }
    
    def _classify_size_change(self, from_shape: Tuple[int, int], to_shape: Tuple[int, int]) -> str:
        """Classify the type of size change."""
        from_area = from_shape[0] * from_shape[1] 
        to_area = to_shape[0] * to_shape[1]
        
        if from_area > to_area:
            return 'extraction_or_crop'
        elif from_area < to_area:
            return 'expansion_or_padding'
        else:
            return 'reshape_same_area'
    
    def _analyze_color_changes(self, train_pairs: List[Tuple[Array, Array]]) -> Dict[str, Any]:
        """Analyze color transformation patterns."""
        color_analysis = {
            'has_color_changes': False,
            'new_colors_introduced': False,
            'colors_removed': False,
            'consistent_color_mappings': True
        }
        
        all_mappings = {}
        
        for inp, out in train_pairs:
            inp_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            
            if inp_colors != out_colors:
                color_analysis['has_color_changes'] = True
                
                if not out_colors.issubset(inp_colors):
                    color_analysis['new_colors_introduced'] = True
                    
                if not inp_colors.issubset(out_colors):
                    color_analysis['colors_removed'] = True
                    
                # Track color mappings
                if inp.shape == out.shape:
                    for color in inp_colors:
                        mask = inp == color
                        if np.any(mask):
                            out_colors_at_pos = set(np.unique(out[mask]))
                            if len(out_colors_at_pos) == 1:
                                mapped_color = list(out_colors_at_pos)[0]
                                if color in all_mappings:
                                    if all_mappings[color] != mapped_color:
                                        color_analysis['consistent_color_mappings'] = False
                                else:
                                    all_mappings[color] = mapped_color
        
        color_analysis['color_mappings'] = all_mappings
        return color_analysis
    
    def _assess_complexity(self, train_pairs: List[Tuple[Array, Array]]) -> Dict[str, Any]:
        """Assess task complexity indicators."""
        complexity = {
            'grid_sizes': 'small',  # small, medium, large
            'color_variety': 'low',  # low, medium, high  
            'shape_complexity': 'simple',  # simple, moderate, complex
            'transformation_type': 'unknown'
        }
        
        # Assess grid sizes
        max_area = max(max(inp.size, out.size) for inp, out in train_pairs)
        if max_area > 400:
            complexity['grid_sizes'] = 'large'
        elif max_area > 100:
            complexity['grid_sizes'] = 'medium'
            
        # Assess color variety
        all_colors = set()
        for inp, out in train_pairs:
            all_colors.update(np.unique(inp))
            all_colors.update(np.unique(out))
        
        if len(all_colors) > 6:
            complexity['color_variety'] = 'high'
        elif len(all_colors) > 3:
            complexity['color_variety'] = 'medium'
        
        return complexity
    
    def _generate_reasoning(self, task_description: Dict[str, Any]) -> LLMReasoning:
        """Generate structured reasoning about the task."""
        
        # Analyze patterns and generate reasoning
        analysis_parts = []
        reasoning_steps = []
        tool_recommendations = []
        
        # Size-based analysis
        if task_description['patterns']['size_changes']['has_size_changes']:
            size_changes = task_description['patterns']['size_changes']['changes']
            if all(change['type'] == 'extraction_or_crop' for change in size_changes):
                analysis_parts.append("Task involves consistent size reduction, likely extraction or cropping pattern.")
                reasoning_steps.append("1. Identify extraction pattern from large grid to smaller target")
                tool_recommendations.extend(['rft_pliance', 'enhanced_search_extraction'])
            elif all(change['type'] == 'expansion_or_padding' for change in size_changes):
                analysis_parts.append("Task involves grid expansion, likely padding or duplication pattern.")
                reasoning_steps.append("1. Analyze padding pattern and fill strategy")
                tool_recommendations.extend(['rft_pliance', 'pattern_synthesis'])
        else:
            analysis_parts.append("Task maintains grid size, transformation is in-place.")
            reasoning_steps.append("1. Focus on in-place transformations: color changes, rotations, reflections")
            tool_recommendations.extend(['rft_pliance', 'hypothesis_engine'])
        
        # Color-based analysis  
        color_info = task_description['patterns']['color_changes']
        if color_info['has_color_changes']:
            if color_info['consistent_color_mappings']:
                analysis_parts.append("Consistent color mappings detected across examples.")
                reasoning_steps.append("2. Apply consistent color transformation rules")
                tool_recommendations.append('rft_color_rules')
            else:
                analysis_parts.append("Complex color transformation pattern detected.")
                reasoning_steps.append("2. Use adaptive color analysis and experimental rules")
                tool_recommendations.extend(['rft_tracking', 'enhanced_search'])
        else:
            analysis_parts.append("No color changes detected, focus on spatial transformations.")
            reasoning_steps.append("2. Analyze spatial patterns: movement, rotation, reflection")
            tool_recommendations.extend(['rft_spatial_rules', 'geometric_analysis'])
        
        # RFT rules analysis
        if task_description['rft_rules']:
            high_conf_rules = [r for r in task_description['rft_rules'] if r.get('confidence', 0) > 0.8]
            if high_conf_rules:
                analysis_parts.append(f"Found {len(high_conf_rules)} high-confidence RFT rules.")
                reasoning_steps.append("3. Apply high-confidence RFT rules first")
                tool_recommendations.insert(0, 'rft_rule_application')
            else:
                analysis_parts.append("RFT rules have low confidence, may need adaptive tracking.")
                reasoning_steps.append("3. Use RFT tracking for rule adaptation")
                tool_recommendations.append('rft_tracking')
        else:
            reasoning_steps.append("3. Generate RFT rules through object inventory and pattern analysis")
            tool_recommendations.insert(0, 'rft_pliance')
        
        # Complexity-based recommendations
        complexity = task_description['complexity_indicators']
        if complexity['grid_sizes'] == 'large':
            reasoning_steps.append("4. Use memory-efficient processing for large grids")
            tool_recommendations.append('memory_optimization')
        
        if complexity['color_variety'] == 'high':
            reasoning_steps.append("5. Consider advanced color pattern analysis")
            tool_recommendations.append('advanced_color_analysis')
        
        # Determine suggested approach
        if 'rft_rule_application' in tool_recommendations:
            suggested_approach = 'rft_direct_application'
            confidence = 0.85
        elif 'rft_pliance' in tool_recommendations[:2]:
            suggested_approach = 'rft_pliance_first'
            confidence = 0.75
        else:
            suggested_approach = 'enhanced_search_primary'
            confidence = 0.65
        
        return LLMReasoning(
            analysis='. '.join(analysis_parts),
            suggested_approach=suggested_approach,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            tool_recommendations=tool_recommendations
        )
    
    def recommend_next_action(self, current_state: Dict[str, Any], 
                            failed_approaches: List[str] = None) -> Dict[str, Any]:
        """Recommend the next high-level action for the solver."""

        failed_approaches = failed_approaches or []
        context = dict(current_state or {})
        context['failed_approaches'] = failed_approaches

        advice = self.advise_on_stage_progress('global', context)

        urgency = 'normal'
        if advice.action in {'switch_to_rft_tracking', 'fallback_to_enhanced_search', 'activate_baseline_fallback'}:
            urgency = 'high'
        elif advice.action in {'evaluate_rule_theories', 'generate_new_hypotheses', 'score_hypotheses'}:
            urgency = 'medium'

        return {
            'stage': advice.stage,
            'action': advice.action,
            'reason': advice.reason,
            'confidence': advice.confidence,
            'notes': advice.notes,
            'alternatives': context.get('candidate_alternatives', []),
            'urgency': urgency,
        }
    
    def generate_explanation(self, solution: Dict[str, Any], 
                           reasoning_used: LLMReasoning) -> str:
        """Generate human-readable explanation of the solution."""
        explanation_parts = [
            f"Solution approach: {reasoning_used.suggested_approach}",
            f"Analysis: {reasoning_used.analysis}",
            "Reasoning steps:"
        ]
        
        for step in reasoning_used.reasoning_steps:
            explanation_parts.append(f"  - {step}")
        
        if solution.get('success', False):
            explanation_parts.append("Solution successfully found and validated.")
        else:
            explanation_parts.append("Solution attempt completed but may need refinement.")
        
        return '\n'.join(explanation_parts)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        import time
        return str(int(time.time()))
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get history of reasoning sessions."""
        return self.reasoning_history.copy()
