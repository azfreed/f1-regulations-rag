"""Context assembly: choose which candidates become the prompt context.

What it does
------------
Given ranked candidates and a token budget, greedily selects chunks (best-first),
deduplicates near-duplicates, formats each with a citation header, and records
which chunks were discarded and why. Produces an :class:`AssembledContext`.

Why it exists
-------------
The generator can only see a bounded context. *Which* chunks we include - and in
what form - strongly affects answer quality and citation accuracy. Making this a
visible, swappable stage lets us experiment with selection strategies and see
exactly what the model was shown.

Deduplication
-------------
Overlapping chunks (especially from the fixed-window chunker) can repeat text. We
drop a candidate whose text is a near-duplicate (normalized prefix match) of one
already selected, recording the reason.

Context selection
-----------------
Greedy-by-score under a token budget is the simplest defensible policy: include the
highest-ranked chunks until the budget is exhausted; everything else is discarded
with reason ``budget_exhausted``.

Replaceable alternatives
-------------------------
- Maximal-marginal-relevance selection, per-article grouping, or neighbour
  expansion could replace :class:`GreedyContextAssembler` via the registry.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..chunking.base import estimate_tokens
from ..models import AssembledContext, DiscardedChunk, RetrievedCandidate
from ..registry import Registry

assembler_registry: Registry["ContextAssembler"] = Registry("context")


@runtime_checkable
class ContextAssembler(Protocol):
    name: str

    def assemble(
        self, candidates: list[RetrievedCandidate], max_tokens: int
    ) -> AssembledContext:
        ...


def _dedup_key(text: str) -> str:
    return " ".join(text.lower().split())[:160]


@assembler_registry.register("greedy")
class GreedyContextAssembler(ContextAssembler):
    name = "greedy"

    def assemble(
        self, candidates: list[RetrievedCandidate], max_tokens: int
    ) -> AssembledContext:
        selected: list[RetrievedCandidate] = []
        discarded: list[DiscardedChunk] = []
        seen_keys: set[str] = set()
        blocks: list[str] = []
        used_tokens = 0

        for cand in candidates:
            chunk = cand.chunk
            key = _dedup_key(chunk.text)
            if key in seen_keys:
                discarded.append(
                    DiscardedChunk(
                        chunk_id=chunk.chunk_id,
                        citation=chunk.citation_label(),
                        reason="duplicate_of_selected",
                    )
                )
                continue

            chunk_tokens = chunk.token_estimate or estimate_tokens(chunk.text)
            if used_tokens + chunk_tokens > max_tokens and selected:
                discarded.append(
                    DiscardedChunk(
                        chunk_id=chunk.chunk_id,
                        citation=chunk.citation_label(),
                        reason="budget_exhausted",
                    )
                )
                continue

            # Each block is prefixed with its citation so the model can attribute claims.
            header = f"[{chunk.citation_label()}]"
            blocks.append(f"{header}\n{chunk.text}")
            seen_keys.add(key)
            used_tokens += chunk_tokens
            selected.append(cand)

        return AssembledContext(
            text="\n\n".join(blocks),
            selected=selected,
            discarded=discarded,
            token_estimate=used_tokens,
        )
