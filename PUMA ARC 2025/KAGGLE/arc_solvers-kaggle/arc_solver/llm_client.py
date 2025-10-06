"""Utility clients for optional LLM integration used by the solver.

This module keeps all optional LLM plumbing separate from the core solver so
that the main code path stays fully offline by default.  Users who want to plug
in an external model (e.g., OpenAI, Anthropic, Ollama) can do so by setting the
appropriate environment variables without modifying the rest of the repo.

When no provider is configured, the solver falls back to deterministic
heuristics.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when an LLM client fails and the caller should fall back."""


class LLMClient:
    """Abstract chat-style interface compatible with common LLM providers."""

    def complete(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Return the raw text from the model.

        Parameters
        ----------
        messages:
            Conversation history in OpenAI-style dicts (role/content).
        response_format:
            Optional hint ("json" or "text") so implementations can request
            structured output when available.
        kwargs:
            Provider-specific overrides such as temperature or max_tokens.
        """

        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    """Wrapper for the OpenAI Responses / ChatCompletions API (if available)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMClientError(
                "openai package is not installed; unable to configure OpenAI client"
            ) from exc

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError("OPENAI_API_KEY is not set")

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        **kwargs,
    ) -> str:
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_output_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        if response_format == "json":  # Prefer explicit JSON when supported.
            params["response_format"] = {"type": "json_object"}

        try:
            response = self._client.responses.create(**params)  # type: ignore
        except Exception as exc:  # pragma: no cover - network errors
            raise LLMClientError(str(exc)) from exc

        # The Responses API exposes output text via .output_text
        try:
            return response.output_text  # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover - older clients
            raise LLMClientError("Unexpected OpenAI response payload") from exc


class OllamaClient(LLMClient):
    """Minimal client for a locally-hosted Ollama server (if available)."""

    def __init__(self, model: str, endpoint: str = "http://localhost:11434") -> None:
        import urllib.request

        self._endpoint = endpoint.rstrip("/") + "/api/chat"
        self._model = model
        self._http = urllib.request  # type: ignore

    def complete(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        **kwargs,
    ) -> str:
        import json as _json

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if response_format == "json":
            payload["format"] = "json"

        data = _json.dumps(payload).encode("utf-8")
        request = self._http.Request(
            self._endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with self._http.urlopen(request, timeout=kwargs.get("timeout", 30)) as resp:
                output = resp.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - runtime/network errors
            raise LLMClientError(str(exc)) from exc

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid output
            raise LLMClientError("Ollama returned invalid JSON") from exc

        messages = parsed.get("message") or {}
        content = messages.get("content")
        if not content:
            raise LLMClientError("Ollama response did not include content")
        return content


def build_llm_client_from_env() -> Optional[LLMClient]:
    """Create an LLM client based on environment variables.

    Supported environment variables:

    - ``PUMA_LLM_PROVIDER``: ``openai`` or ``ollama`` (case-insensitive)
    - ``PUMA_LLM_MODEL``: optional override for the model name
    - ``OPENAI_API_KEY``: required for the OpenAI provider
    - ``OLLAMA_BASE_URL``: optional override for the Ollama endpoint

    Returns ``None`` if no provider is configured or if client creation fails.
    """

    provider = os.environ.get("PUMA_LLM_PROVIDER", "").strip().lower()
    if not provider:
        return None

    model = os.environ.get("PUMA_LLM_MODEL", "gpt-4o-mini")

    try:
        if provider == "openai":
            return OpenAIChatClient(model=model)
        if provider == "ollama":
            endpoint = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaClient(model=model, endpoint=endpoint)
        logger.warning("Unsupported LLM provider '%s' — falling back to heuristics", provider)
    except LLMClientError as exc:
        logger.warning("Failed to configure LLM provider '%s': %s", provider, exc)

    return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object embedded in arbitrary text."""

    import json as _json

    if not text:
        return None

    text = text.strip()
    if text.startswith("```"):
        # Strip fenced code blocks like ```json ... ```
        parts = text.split("```")
        candidates = [p for p in parts if p.strip() and not p.strip().startswith("json")]
        if candidates:
            text = candidates[0].strip()

    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass

    # Fallback: scan for the first '{' and last '}' to attempt a substring parse.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _json.loads(text[start : end + 1])
        except _json.JSONDecodeError:
            return None
    return None
