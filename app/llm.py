"""Provider-agnostic access to the language model.

Three entry points:

  ask()      - one-shot question in, text out. No tools, no memory.
  generate() - lower level. Accepts conversation history and tool schemas,
               and can return a request to call a tool.
  embed()    - turns text into vectors (lists of numbers) for RAG search.

The rest of the application goes through this module and never imports a
vendor SDK directly, so switching providers is a change to .env plus a new
branch here - not a rewrite of every agent.
"""

from __future__ import annotations

import re
import time 
from functools import lru_cache
from typing import Callable, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.utils.config import Settings, get_settings

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

# How many times to retry a request that failed for a temporary reason.
MAX_ATTEMPTS = 4

# Free tiers reject bursts. Rather than firing requests as fast as possible
# and dealing with the 429 afterwards, we space them out so most 429s never
# happen at all. Retry is the safety net; this is the seatbelt.
MIN_SECONDS_BETWEEN_CALLS = 4.0

# Embedding settings. 768 dimensions is plenty for document search and uses
# a quarter of the storage of the model's full 3072.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
EMBED_BATCH_SIZE = 50

# Gemini puts the wait time in the error text: "Please retry in 12.2s"
_RETRY_DELAY_PATTERN = re.compile(r"retry in ([\d.]+)s")

_last_call_at = 0.0


class LLMError(RuntimeError):
    """Raised when the model could not be reached or refused to answer."""


@lru_cache(maxsize=1)
def _gemini_client() -> genai.Client:
    """Create the Gemini client once and reuse it (it holds a connection pool)."""
    return genai.Client(api_key=get_settings().api_key)


def _throttle() -> None:
    """Block until MIN_SECONDS_BETWEEN_CALLS has passed since the last call."""
    global _last_call_at
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _is_retryable(error: genai_errors.APIError) -> bool:
    """429 means slow down; 5xx means the server is busy. Both pass with time."""
    return error.code == 429 or error.code >= 500


def _retry_delay(error: genai_errors.APIError, attempt: int) -> float:
    """Prefer the server's suggested wait; otherwise back off exponentially."""
    match = _RETRY_DELAY_PATTERN.search(str(error.message or ""))
    if match:
        return float(match.group(1)) + 1.0
    return min(5.0 * (2 ** (attempt - 1)), 60.0)


def _describe(error: genai_errors.APIError, model: str) -> str:
    """Turn a raw API error into something a human can act on."""
    if error.code == 429:
        return (
            f"Rate limit reached for {model}. The Gemini free tier allows only "
            "a few requests per minute. Wait a minute and try again."
        )
    if error.code >= 500:
        return f"{model} is temporarily unavailable ({error.code}). Try again shortly."
    if error.code == 403:
        return "API key rejected (403). Check GEMINI_API_KEY in .env."
    if error.code == 404:
        return f"Model {model!r} does not exist (404). Check the model name."
    return f"Request rejected ({error.code}): {error.message}"


def _with_retry(request: Callable[[], T], model: str) -> T:
    """Run an API request with pacing and automatic retry.

    Every call to the provider goes through here, so rate limiting and error
    handling are written once and apply to chat and embeddings alike.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _throttle()
            return request()
        except genai_errors.APIError as error:
            if not _is_retryable(error) or attempt == MAX_ATTEMPTS:
                raise LLMError(_describe(error, model)) from error

            delay = _retry_delay(error, attempt)
            print(
                f"    [llm] {error.code} - waiting {delay:.0f}s "
                f"(attempt {attempt}/{MAX_ATTEMPTS})"
            )
            time.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise LLMError("Exhausted all retry attempts.")


def _require_gemini() -> Settings:
    """Return settings, or fail clearly if another provider is selected."""
    settings = get_settings()
    if settings.llm_provider != "gemini":
        raise NotImplementedError(
            f"Provider {settings.llm_provider!r} is configured but not implemented yet."
        )
    return settings


def generate(
    contents: str | list[types.Content],
    system: str | None = None,
    tools: list[dict] | None = None,
) -> types.GenerateContentResponse:
    """Send a request to the model and return the raw provider response.

    Args:
        contents: Either a plain prompt string, or the full conversation
            history as a list of Content objects.
        system: Optional persona / rules that shape how the model responds.
        tools: Optional list of JSON-Schema tool definitions. When supplied,
            the model may reply with a function call instead of text.

    Returns:
        The provider's response object. The caller must inspect it - it may
        contain text, one or more function calls, or both.

    Raises:
        LLMError: On any API failure, with a readable message.
    """
    settings = _require_gemini()

    config = types.GenerateContentConfig(
        system_instruction=system,
        # We drive the tool loop ourselves so we can log and validate each
        # call, so the SDK's automatic function calling stays off.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tools=[types.Tool(function_declarations=tools)] if tools else None,
    )

    return _with_retry(
        lambda: _gemini_client().models.generate_content(
            model=settings.model,
            contents=contents,
            config=config,
        ),
        settings.model,
    )


def embed(texts: list[str], for_query: bool = False) -> list[list[float]]:
    """Convert texts into embedding vectors.

    Args:
        texts: The strings to embed.
        for_query: True when embedding a search question, False when
            embedding document chunks. The model encodes the two slightly
            differently so a short question lands near the passage that
            answers it, not near passages that merely share its words.

    Returns:
        One vector (a list of EMBEDDING_DIMENSIONS floats) per input text,
        in the same order.
    """
    _require_gemini()

    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY" if for_query else "RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )

    vectors: list[list[float]] = []
    # Send in batches: one request per text would burn the rate limit.
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = _with_retry(
            lambda: _gemini_client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=config,
            ),
            EMBEDDING_MODEL,
        )
        vectors.extend(list(item.values) for item in response.embeddings)

    return vectors


def extract_text(response: types.GenerateContentResponse) -> str:
    """Pull the plain-text parts out of a response, ignoring function calls.

    Reading `response.text` directly raises when the response also contains
    function calls, so we walk the parts instead.
    """
    if not response.candidates:
        return ""

    parts = response.candidates[0].content.parts or []
    return "".join(part.text for part in parts if part.text).strip()


def ask(prompt: str, system: str | None = None) -> str:
    """Send a single prompt to the configured model and return its text reply."""
    return extract_text(generate(prompt, system=system))


def ask_json(prompt: str, schema: type[M], system: str | None = None) -> M:
    """Ask the model for output shaped like `schema`, and return a parsed instance.

    Uses Gemini's structured-output mode: the model is constrained to emit
    JSON matching the schema. response.parsed already gives a validated
    instance in the common case; we fall back to parsing response.text by
    hand for the rare case where the SDK could not auto-parse it.
    """
    settings = _require_gemini()

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
    )

    response = _with_retry(
        lambda: _gemini_client().models.generate_content(
            model=settings.model,
            contents=prompt,
            config=config,
        ),
        settings.model,
    )

    if response.parsed is not None:
        return response.parsed

    try:
        return schema.model_validate_json(response.text)
    except ValidationError as error:
        raise LLMError(
            f"Model returned JSON that does not match {schema.__name__}: {error}"
        ) from error
