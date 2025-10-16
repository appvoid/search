from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchResult:
    """Represents a single search result pulled from a search engine."""

    title: str
    link: str
    description: Optional[str] = None
    content: Optional[str] = None

    def to_dict(self) -> dict:
        data = {"title": self.title, "link": self.link}
        if self.description:
            data["description"] = self.description
        if self.content:
            data["content"] = self.content
        return data


@dataclass
class EvaluationResult:
    """Represents the LLM evaluation of an answer."""

    satisfactory: bool
    reason: str = ""


@dataclass
class WorkflowResult:
    """Outcome of a complete search workflow iteration."""

    answer: str
    evaluation: EvaluationResult
    attempts: int
    answers_history: List[str] = field(default_factory=list)
