"""
LLM Meta-Reasoner for PUMA ARC Solver.

This module implements a meta-reasoning layer that sits ABOVE all existing
search systems (episodic memory, sketches, RFT engine, neural guidance) and
acts as the "prefrontal cortex" of the solver. It makes strategic decisions
about which search methods to use, reasons over similar episodes and relational
facts, and generates operation sequences with parameter suggestions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid import Array
from .features import extract_task_features
from .llm_interface import LLMInterface, LLMConfig


@dataclass
class MetaReasoningResult:
    """Result of LLM meta-reasoning."""

    strategy: str  # Primary search strategy to use
    suggested_operations: List[str]  # Ordered list of operations to try
    operation_parameters: Dict[str, Dict[str, Any]]  # Suggested parameters per operation
    search_priorities: Dict[str, float]  # Weight for each search method
    reasoning: str  # LLM's reasoning explanation
    confidence: float  # Confidence in the recommendation


class LLMMetaReasoner:
    """LLM-powered meta-reasoner that coordinates all PUMA systems.

    This acts as the strategic coordination layer that:
    1. Analyzes task features and similar episodes
    2. Reasons about which search strategies are most promising
    3. Suggests operation sequences based on RFT facts and patterns
    4. Provides parameter recommendations for operations
    5. Prioritizes different search methods dynamically
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        enabled: bool = True
    ):
        """Initialize the meta-reasoner.

        Args:
            llm_config: Configuration for the LLM
            enabled: Whether to enable LLM reasoning (can disable for ablation)
        """
        self.enabled = enabled
        self.llm = LLMInterface(llm_config) if enabled else None
        self.cache: Dict[str, MetaReasoningResult] = {}

    def reason_about_task(
        self,
        train_pairs: List[Tuple[Array, Array]],
        task_features: Dict[str, Any],
        similar_episodes: List[Tuple[Any, float]],
        rft_facts: List[Any],
        predicted_ops: List[str]
    ) -> MetaReasoningResult:
        """Perform meta-reasoning about how to solve the task.

        Args:
            train_pairs: Training input/output pairs
            task_features: Extracted task features
            similar_episodes: List of (episode, similarity) from episodic memory
            rft_facts: Relational facts from RFT engine
            predicted_ops: Operations predicted by neural guidance

        Returns:
            Meta-reasoning result with strategy and suggestions
        """
        if not self.enabled or self.llm is None:
            # Fallback: return default strategy
            return self._default_strategy(task_features, predicted_ops)

        # Create cache key
        cache_key = self._make_cache_key(train_pairs)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Build context from all sources
        context = self._build_context(
            task_features,
            similar_episodes,
            rft_facts,
            predicted_ops
        )

        # Create prompts for LLM
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(context, train_pairs)

        # Get LLM reasoning
        try:
            response = self.llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3  # Lower temperature for more focused reasoning
            )

            result = self._parse_llm_response(response, predicted_ops)

        except Exception as e:
            print(f"Warning: LLM meta-reasoning failed: {e}")
            result = self._default_strategy(task_features, predicted_ops)

        # Cache result
        self.cache[cache_key] = result
        if len(self.cache) > 50:  # Limit cache size
            # Remove oldest entries
            keys = list(self.cache.keys())
            for key in keys[:10]:
                del self.cache[key]

        return result

    def _build_context(
        self,
        task_features: Dict[str, Any],
        similar_episodes: List[Tuple[Any, float]],
        rft_facts: List[Any],
        predicted_ops: List[str]
    ) -> Dict[str, Any]:
        """Build context dictionary from all sources."""
        context = {
            "task_features": self._summarize_features(task_features),
            "similar_episodes": self._summarize_episodes(similar_episodes),
            "rft_facts": self._summarize_rft_facts(rft_facts),
            "predicted_ops": predicted_ops[:10]  # Top 10 operations
        }
        return context

    def _summarize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features for LLM reasoning."""
        return {
            "num_train_pairs": features.get("num_train_pairs", 0),
            "input_shape": (features.get("input_height_mean", 0), features.get("input_width_mean", 0)),
            "output_shape": (features.get("output_height_mean", 0), features.get("output_width_mean", 0)),
            "shape_preserved": bool(features.get("shape_preserved", False)),
            "size_ratio": features.get("size_ratio_mean", 1.0),
            "input_colors": features.get("input_colors_mean", 0),
            "output_colors": features.get("output_colors_mean", 0),
            "likely_rotation": bool(features.get("likely_rotation", 0) > 0.5),
            "likely_reflection": bool(features.get("likely_reflection", 0) > 0.5),
            "likely_translation": bool(features.get("likely_translation", 0) > 0.5),
            "likely_recolor": bool(features.get("likely_recolor", 0) > 0.5),
            "likely_crop": bool(features.get("likely_crop", 0) > 0.5),
            "likely_pad": bool(features.get("likely_pad", 0) > 0.5)
        }

    def _summarize_episodes(self, episodes: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
        """Summarize similar episodes for LLM."""
        summaries = []
        for episode, similarity in episodes[:5]:  # Top 5 episodes
            programs = []
            for program in episode.programs[:3]:  # Top 3 programs per episode
                ops = [op for op, _ in program]
                programs.append(ops)

            summaries.append({
                "similarity": float(similarity),
                "programs": programs,
                "success_count": episode.success_count
            })
        return summaries

    def _summarize_rft_facts(self, facts: List[Any]) -> List[Dict[str, Any]]:
        """Summarize RFT relational facts."""
        if not facts:
            return []

        summaries = []
        for fact in facts[:10]:  # Top 10 facts
            try:
                summaries.append({
                    "frame": getattr(fact, "frame", "unknown"),
                    "context": getattr(fact, "context", "unknown"),
                    "confidence": float(getattr(fact, "confidence", 0.0))
                })
            except Exception:
                continue

        return summaries

    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM."""
        return """You are a meta-reasoning system for solving ARC (Abstraction and Reasoning Corpus) puzzles.

Your role is to analyze a task and strategically decide:
1. Which search strategy is most promising (human_reasoning, episodic, beam_search, neural_guided, sketch_based)
2. Which DSL operations are likely needed (rotate, flip, transpose, translate, recolor, crop, pad, extract_*, etc.)
3. What parameters those operations might need
4. How to prioritize different search methods

You have access to:
- Task features (shape changes, color patterns, transformations)
- Similar previously-solved tasks (episodic memory)
- Relational facts from the RFT engine (spatial relationships)
- Neural guidance predictions

Output your reasoning as JSON with these fields:
{
  "strategy": "primary_strategy_name",
  "suggested_operations": ["op1", "op2", ...],
  "operation_parameters": {"op1": {"param": value}, ...},
  "search_priorities": {"method_name": priority_weight, ...},
  "reasoning": "your explanation",
  "confidence": 0.0-1.0
}

Be concise and focus on the most important patterns."""

    def _create_user_prompt(
        self,
        context: Dict[str, Any],
        train_pairs: List[Tuple[Array, Array]]
    ) -> str:
        """Create user prompt with task context."""
        features = context["task_features"]
        episodes = context["similar_episodes"]
        rft = context["rft_facts"]
        ops = context["predicted_ops"]

        # Sample one training pair for visualization
        example_input = train_pairs[0][0] if train_pairs else None
        example_output = train_pairs[0][1] if train_pairs else None

        prompt = f"""Analyze this ARC task and recommend a solving strategy:

TASK FEATURES:
- Training pairs: {features['num_train_pairs']}
- Input shape: {features['input_shape']}
- Output shape: {features['output_shape']}
- Shape preserved: {features['shape_preserved']}
- Size ratio: {features['size_ratio']:.2f}
- Input colors: {features['input_colors']}, Output colors: {features['output_colors']}
- Likely transformations:
  * Rotation: {features['likely_rotation']}
  * Reflection: {features['likely_reflection']}
  * Translation: {features['likely_translation']}
  * Recolor: {features['likely_recolor']}
  * Crop: {features['likely_crop']}
  * Pad: {features['likely_pad']}

SIMILAR EPISODES: {len(episodes)} found
"""

        if episodes:
            prompt += "Top similar solutions used:\n"
            for i, ep in enumerate(episodes[:3]):
                prompt += f"  {i+1}. (similarity {ep['similarity']:.2f}): {ep['programs']}\n"

        if rft:
            prompt += f"\nRFT RELATIONAL FACTS: {len(rft)} detected\n"
            for i, fact in enumerate(rft[:3]):
                prompt += f"  {i+1}. {fact['frame']}: {fact['context']} (conf: {fact['confidence']:.2f})\n"

        prompt += f"\nNEURAL PREDICTIONS: {', '.join(ops[:10])}\n"

        if example_input is not None and example_output is not None:
            prompt += f"\nEXAMPLE PAIR:\nInput shape: {example_input.shape}, Output shape: {example_output.shape}\n"
            prompt += f"Input colors: {set(example_input.flatten())}, Output colors: {set(example_output.flatten())}\n"

        prompt += "\nBased on this information, what strategy should we use?"

        return prompt

    def _parse_llm_response(
        self,
        response: Dict[str, Any],
        fallback_ops: List[str]
    ) -> MetaReasoningResult:
        """Parse LLM JSON response into MetaReasoningResult."""
        try:
            return MetaReasoningResult(
                strategy=response.get("strategy", "neural_guided"),
                suggested_operations=response.get("suggested_operations", fallback_ops[:5]),
                operation_parameters=response.get("operation_parameters", {}),
                search_priorities=response.get("search_priorities", {
                    "human_reasoning": 0.3,
                    "episodic": 0.25,
                    "neural_guided": 0.2,
                    "beam_search": 0.15,
                    "sketch_based": 0.1
                }),
                reasoning=response.get("reasoning", ""),
                confidence=float(response.get("confidence", 0.5))
            )
        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return self._default_strategy({}, fallback_ops)

    def _default_strategy(
        self,
        features: Dict[str, Any],
        predicted_ops: List[str]
    ) -> MetaReasoningResult:
        """Return default strategy when LLM is not available."""
        return MetaReasoningResult(
            strategy="neural_guided",
            suggested_operations=predicted_ops[:10] if predicted_ops else ["identity"],
            operation_parameters={},
            search_priorities={
                "human_reasoning": 0.3,
                "episodic": 0.25,
                "neural_guided": 0.2,
                "beam_search": 0.15,
                "sketch_based": 0.1
            },
            reasoning="Default strategy (LLM disabled)",
            confidence=0.5
        )

    def _make_cache_key(self, train_pairs: List[Tuple[Array, Array]]) -> str:
        """Create a cache key from training pairs."""
        if not train_pairs:
            return "empty"

        # Use first pair's shapes and some hash of values
        inp, out = train_pairs[0]
        inp_hash = hash(inp.tobytes())
        out_hash = hash(out.tobytes())
        return f"{inp.shape}_{out.shape}_{inp_hash}_{out_hash}"

    def cleanup(self) -> None:
        """Clean up LLM resources."""
        if self.llm is not None:
            self.llm.cleanup()
        self.cache.clear()


def create_meta_reasoner(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    enabled: bool = True
) -> LLMMetaReasoner:
    """Convenience function to create a meta-reasoner.

    Args:
        model_name: HuggingFace model identifier
        enabled: Whether to enable LLM reasoning

    Returns:
        Configured meta-reasoner
    """
    if not enabled:
        return LLMMetaReasoner(enabled=False)

    config = LLMConfig(
        model_name=model_name,
        temperature=0.3,  # Lower for reasoning
        max_tokens=512
    )
    return LLMMetaReasoner(llm_config=config, enabled=True)
