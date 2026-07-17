"""Anthropic answer generator.

What it does
------------
Sends the versioned system prompt + (question, context) to an Anthropic model at
temperature 0 and returns a cited :class:`Answer`.

Why temperature 0
-----------------
Reproducibility. Experiments must be comparable across runs, so generation is
deterministic-ish by fixing temperature to 0 (set in :class:`Settings`).

No silent fallback
------------------
If there is no context to ground the answer, we raise :class:`GenerationError`
rather than letting the model answer from general knowledge. The prompt also
instructs the model to reply ``UNANSWERABLE`` when the context is insufficient.

Assumptions
-----------
- ``anthropic`` is installed and ``ANTHROPIC_API_KEY`` is set. Otherwise a clear
  error is raised at construction / call time.
"""

from __future__ import annotations

import anthropic

from ..errors import GenerationError
from ..logging_utils import get_logger
from ..models import AssembledContext, Answer
from .base import Generator, extract_citations, generator_registry
from .prompts import DEFAULT_PROMPT_VERSION, get_prompt

logger = get_logger(__name__)

_UNANSWERABLE_MARKER = "UNANSWERABLE"


class AnthropicGenerator(Generator):
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        if not api_key:
            raise GenerationError(
                "ANTHROPIC_API_KEY is not set. Set it in the environment or .env to "
                "enable answer generation (retrieval/context still work without it)."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(
        self, question: str, context: AssembledContext, prompt_version: str | None = None
    ) -> Answer:
        if not context.text.strip() or not context.selected:
            # Hard stop: never answer without grounding context.
            raise GenerationError(
                "no retrieval context available; refusing to answer from general "
                "model knowledge"
            )

        version = prompt_version or DEFAULT_PROMPT_VERSION
        prompt = get_prompt(version)
        user = prompt.render_user(question=question, context=context.text)

        logger.info("generating answer with %s (prompt %s)", self._model, version)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=prompt.system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text").strip()

        is_unanswerable = text.startswith(_UNANSWERABLE_MARKER)
        citations = [] if is_unanswerable else extract_citations(text, context)
        return Answer(
            question=question,
            text=text,
            citations=citations,
            is_unanswerable=is_unanswerable,
            generation_model=self._model,
            prompt_version=version,
        )


@generator_registry.register("anthropic")
def _make_anthropic(
    api_key: str | None = None,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Generator:
    return AnthropicGenerator(api_key=api_key, model=model, max_tokens=max_tokens, temperature=temperature)
