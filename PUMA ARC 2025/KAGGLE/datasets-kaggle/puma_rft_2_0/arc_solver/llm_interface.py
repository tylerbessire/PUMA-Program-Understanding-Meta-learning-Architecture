"""
LLM Interface for PUMA ARC Solver.

This module provides a lightweight interface to local small LLMs (Phi-3-mini,
Qwen2.5-3B, Llama-3.2-3B) with 4-bit quantization for efficient inference.
The interface supports structured generation with system/user prompts and
temperature controls.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""

    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    use_4bit: bool = True
    temperature: float = 0.7
    max_tokens: int = 512
    device: str = "auto"
    cache_dir: Optional[str] = None


class LLMInterface:
    """Interface to local small language models for meta-reasoning.

    Supports:
    - microsoft/Phi-3-mini-4k-instruct
    - Qwen/Qwen2.5-3B-Instruct
    - meta-llama/Llama-3.2-3B-Instruct

    Uses 4-bit quantization for memory efficiency (~2GB RAM).
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM interface.

        Args:
            config: LLM configuration. If None, uses default Phi-3-mini config.
        """
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _lazy_load(self) -> None:
        """Lazy load the model to avoid startup overhead."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Configure 4-bit quantization
            if self.config.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None

            # Load model and tokenizer
            print(f"Loading LLM model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map=self.config.device,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._initialized = True
            print(f"LLM loaded successfully. Memory footprint: ~2GB")

        except ImportError as e:
            print(f"Warning: Failed to import transformers/torch: {e}")
            print("LLM functionality will be disabled. Install with: pip install transformers torch bitsandbytes accelerate")
            self._initialized = False
        except Exception as e:
            print(f"Warning: Failed to load LLM model: {e}")
            self._initialized = False

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text from the LLM.

        Args:
            system_prompt: System message to set context
            user_prompt: User query/instruction
            temperature: Sampling temperature (overrides config if provided)
            max_tokens: Maximum tokens to generate (overrides config if provided)

        Returns:
            Generated text response
        """
        if not self._initialized:
            self._lazy_load()

        if not self._initialized or self.model is None:
            return ""  # Return empty string if model failed to load

        try:
            import torch

            # Format prompt based on model type
            if "Phi-3" in self.config.model_name:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif "Qwen" in self.config.model_name:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif "Llama" in self.config.model_name:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                # Generic fallback
                prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate
            temp = temperature if temperature is not None else self.config.temperature
            max_new_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=temp > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode and extract only the new tokens
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Try to extract just the assistant's response
            if "Assistant:" in full_output:
                response = full_output.split("Assistant:")[-1].strip()
            elif "<|assistant|>" in full_output:
                response = full_output.split("<|assistant|>")[-1].strip()
            else:
                # Fallback: remove the prompt
                response = full_output[len(prompt):].strip()

            return response

        except Exception as e:
            print(f"Warning: LLM generation failed: {e}")
            return ""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON output from the LLM.

        Args:
            system_prompt: System message (should mention JSON output format)
            user_prompt: User query/instruction
            temperature: Sampling temperature

        Returns:
            Parsed JSON dictionary, or empty dict on failure
        """
        response = self.generate(system_prompt, user_prompt, temperature)

        if not response:
            return {}

        # Try to extract JSON from response
        try:
            # Look for JSON markers
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            elif "{" in response and "}" in response:
                # Extract first JSON object
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"Warning: Failed to parse JSON from LLM: {e}")
            print(f"Response was: {response[:200]}")
            return {}

    def cleanup(self) -> None:
        """Free GPU/CPU memory used by the model."""
        if self.model is not None:
            try:
                import torch
                del self.model
                del self.tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self.model = None
        self.tokenizer = None
        self._initialized = False


def create_llm_interface(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    temperature: float = 0.7,
    max_tokens: int = 512
) -> LLMInterface:
    """Convenience function to create an LLM interface.

    Args:
        model_name: HuggingFace model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Configured LLM interface
    """
    config = LLMConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return LLMInterface(config)
