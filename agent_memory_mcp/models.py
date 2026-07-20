"""Pydantic v2 data models and small loaders (DESIGN.md §8.5)."""

from __future__ import annotations

import json
from pathlib import Path as _FsPath
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    id: str
    type: str
    name: str
    attrs: dict = Field(default_factory=dict)


class Relation(BaseModel):
    src: str
    rel: str
    dst: str


class Statement(BaseModel):
    text: str
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class Fact(BaseModel):
    fact_id: str
    text: str
    src: Optional[str] = None
    rel: Optional[str] = None
    dst: Optional[str] = None


class Hit(BaseModel):
    fact_id: str
    text: str
    score: float


class Path(BaseModel):
    steps: list[str] = Field(default_factory=list)  # ["Dana --MANAGED_BY--> Evan", ...]


class RememberResult(BaseModel):
    learned: list[Relation] = Field(default_factory=list)
    entities_added: int = 0
    relations_added: int = 0
    stored_raw: int = 0
    note: str = ""


class RecallResult(BaseModel):
    answer: Optional[str] = None
    method: Literal["graph", "vector", "none"] = "none"
    support: list[str] = Field(default_factory=list)
    path: list[str] = Field(default_factory=list)


class Question(BaseModel):
    id: str
    text: str
    category: Literal["single_hop", "multi_hop", "aggregation"]
    gold_answer: str


class GradedResult(BaseModel):
    question_id: str
    method: str
    category: str
    predicted: Optional[str]
    gold: str
    correct: bool


class Scorecard(BaseModel):
    by_method_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    overall: dict[str, float] = Field(default_factory=dict)
    n: int = 0
    results: list[GradedResult] = Field(default_factory=list)


def load_questions(path: str | _FsPath) -> list[Question]:
    """Load the labeled eval questions from a JSON file (DESIGN.md §8.4)."""
    data = json.loads(_FsPath(path).read_text(encoding="utf-8"))
    return [Question(**item) for item in data]


def read_corpus_lines(path: str | _FsPath) -> list[str]:
    """Read the seed corpus: one natural-language statement per non-empty line."""
    lines = _FsPath(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
