"""Konfigurationsmodell für die Sequenzanalyse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class SequenzAnalyseConfig:
    """Bündelt Parameter für Modell- und Analyseausführung."""
    model: str = "gpt-5-nano"
    max_output_tokens: Optional[int] = None
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "medium"
    reasoning_summary: Optional[str] = None
    temperature: float = 1.0
    tool_choice: str = "none"
    store: bool = False
