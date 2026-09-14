"""Application configuration, loaded once from environment variables.

Every secret and tunable value enters the program through this module.
Nothing else in the codebase should read `os.environ` directly - that way
there is exactly one place to look when something is misconfigured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# config.py lives at <root>/app/utils/config.py, so the project root is
# three levels up. Building the path this way means the program works no
# matter which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Read .env and copy its values into os.environ.
load_dotenv(ENV_FILE)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    """Return an environment variable, or fail with an actionable message."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}\n"
            f"Add it to {ENV_FILE} - see .env.example for the format."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the app's configuration."""

    llm_provider: str
    model: str
    api_key: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and validate the settings. Cached, so .env is parsed only once."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "gemini":
        return Settings(
            llm_provider=provider,
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
            api_key=_require("GEMINI_API_KEY"),
        )

    if provider == "anthropic":
        return Settings(
            llm_provider=provider,
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-5").strip(),
            api_key=_require("ANTHROPIC_API_KEY"),
        )

    raise ConfigError(
        f"Unsupported LLM_PROVIDER: {provider!r}. Expected 'gemini' or 'anthropic'."
    )
