"""Confounder configuration via Pydantic settings."""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    MISTRAL = "mistral"
    TOGETHER = "together"


# ── Model defaults per provider ────────────────────────────────
_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "llama3.1",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.GROQ: "llama-3.1-70b-versatile",
    LLMProvider.MISTRAL: "mistral-large-latest",
    LLMProvider.TOGETHER: "meta-llama/Llama-3-70b-chat-hf",
}


class ConfounderSettings(BaseSettings):
    """Application settings loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="CONFOUNDER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="LLM provider to use for confounder generation",
    )
    llm_model: str = Field(
        default="",
        description="Model name override. Empty = use provider default",
    )
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=1, le=10)

    # ── Statistics ──────────────────────────────────────────────
    alpha: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Statistical significance level",
    )
    bias_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Minimum relative bias effect (10%) to be considered problematic",
    )

    # ── General ─────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Derived helpers ─────────────────────────────────────────
    @property
    def resolved_model(self) -> str:
        """Return the LiteLLM model string."""
        base = self.llm_model or _DEFAULT_MODELS.get(self.llm_provider, "llama3.1")
        if self.llm_provider == LLMProvider.OLLAMA and "/" not in base:
            return f"ollama/{base}"
        return base


def configure_logging(level: str = "INFO") -> None:
    """Set up structured logging."""
    fmt = "%(asctime)s │ %(name)s │ %(levelname)-7s │ %(message)s"
    logging.basicConfig(format=fmt, datefmt="%H:%M:%S", level=getattr(logging, level.upper()))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_settings() -> ConfounderSettings:
    """Singleton settings accessor."""
    return ConfounderSettings()
