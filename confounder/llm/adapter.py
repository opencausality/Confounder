"""LiteLLM adapter — multi-provider LLM wrapper with retry logic."""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from confounder.config import ConfounderSettings, get_settings

logger = logging.getLogger(__name__)


class ConfounderProviderError(Exception):
    """Raised when the LLM provider fails after all retries."""


class LLMAdapter:
    """Wrapper around LiteLLM with exponential backoff and transparent logging."""

    def __init__(self, settings: ConfounderSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.resolved_model
        logger.info("LLMAdapter initialised → model=%s, provider=%s",
                     self._model, self._settings.llm_provider.value)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        format_json: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send a prompt to the LLM with retry logic."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self._settings.llm_temperature
        max_retries = self._settings.llm_max_retries
        
        call_kwargs = {**kwargs}
        if format_json:
            # Note: Not all providers support response_format strictly,
            # but LiteLLM handles translating this to the provider format.
            call_kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(1, max_retries + 1):
            try:
                response = litellm.completion(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                content = response.choices[0].message.content or ""
                tokens = getattr(response, "usage", None)
                token_count = getattr(tokens, "total_tokens", 0) if tokens else 0
                logger.info("LLM call succeeded → model=%s, tokens_used=%d",
                           self._model, token_count)
                return content

            except Exception as exc:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, max_retries, exc, wait,
                )
                if attempt < max_retries:
                    time.sleep(wait)

        raise ConfounderProviderError(
            f"All {max_retries} LLM call attempts failed. "
            f"Provider={self._settings.llm_provider.value}, model={self._model}."
        )

    @property
    def provider_info(self) -> dict[str, str]:
        return {
            "provider": self._settings.llm_provider.value,
            "model": self._model,
        }
