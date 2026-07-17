"""Versioned evaluation dataset.

Each :class:`EvalCase` states a question and what a good system should do:
expected articles (for retrieval metrics), expected answer terms (for a light
answer check), whether the question is answerable from the regulations, and
optional notes. The dataset carries a ``version`` so results are comparable only
against the same cases.

The seed dataset below uses real 2026 regulation references so the test bed is
useful out of the box; extend it in ``data/evaluations/<version>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    question: str
    expected_articles: list[str] = Field(default_factory=list)  # e.g. ["D3", "D4.1"]
    expected_answer_terms: list[str] = Field(default_factory=list)
    answerable: bool = True
    notes: str | None = None


class EvalDataset(BaseModel):
    version: str
    cases: list[EvalCase]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def load_dataset(path: str | Path) -> EvalDataset:
    return EvalDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def seed_dataset() -> EvalDataset:
    """A small starter dataset grounded in the 2026 FIA F1 Regulations.

    Article numbers are best-effort references to the relevant sections; adjust
    them against the parsed corpus as the test bed matures.
    """

    cases = [
        EvalCase(
            id="points-system",
            question="How are World Championship points awarded in a Grand Prix?",
            expected_articles=["A2.2"],
            expected_answer_terms=["points", "first", "tenth"],
            answerable=True,
            notes="Section A championship points system.",
        ),
        EvalCase(
            id="factory-shutdown",
            question="What restrictions apply during a factory shutdown period?",
            expected_articles=["B"],
            expected_answer_terms=["shutdown", "period"],
            answerable=True,
            notes="Restricted/shutdown period rules (Sporting).",
        ),
        EvalCase(
            id="cost-cap",
            question="What is included in the cost cap for F1 teams?",
            expected_articles=["D"],
            expected_answer_terms=["cost", "cap"],
            answerable=True,
            notes="Section D financial regulations.",
        ),
        EvalCase(
            id="fillet-radius",
            question="How is a Fillet Radius defined in the technical regulations?",
            expected_articles=["C3.2.6"],
            expected_answer_terms=["fillet", "internal corner", "concave"],
            answerable=True,
            notes="Section C technical definitions.",
        ),
        EvalCase(
            id="pressure-tappings",
            question="What are the rules for pressure tappings on the car?",
            expected_articles=["C3.2.7"],
            expected_answer_terms=["pressure", "2mm", "flush"],
            answerable=True,
            notes="Section C pressure measuring apertures.",
        ),
        EvalCase(
            id="entry-applications",
            question="What is required for an F1 team entry application?",
            expected_articles=["A3.1"],
            expected_answer_terms=["entry", "application"],
            answerable=True,
        ),
        EvalCase(
            id="anti-doping",
            question="What does the anti-doping provision require?",
            expected_articles=["A4.2"],
            expected_answer_terms=["anti-doping"],
            answerable=True,
        ),
        EvalCase(
            id="ticket-prices",
            question="How much does a general admission ticket to the Monaco Grand Prix cost?",
            expected_articles=[],
            expected_answer_terms=[],
            answerable=False,
            notes="Not covered by the regulations; system should decline.",
        ),
        EvalCase(
            id="driver-salary",
            question="What is the maximum salary a driver can be paid?",
            expected_articles=[],
            expected_answer_terms=[],
            answerable=False,
            notes="Driver salaries are excluded from the cost cap; no maximum is set.",
        ),
    ]
    return EvalDataset(version="2026-seed-v1", cases=cases)
