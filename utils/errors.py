"""Structured error types and aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


class ProcessingError(Exception):
    """Base class for processing errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ValidationError(ProcessingError):
    """Validation error with details."""


class ResourceError(ProcessingError):
    """Resource allocation or limit error."""


@dataclass
class AggregatedError:
    time: datetime
    error: ProcessingError
    context: Dict[str, Any]


class ErrorAggregator:
    """Collect multiple processing errors for later inspection."""

    def __init__(self) -> None:
        self.errors: List[AggregatedError] = []

    def add_error(self, error: ProcessingError) -> None:
        self.errors.append(
            AggregatedError(time=datetime.now(), error=error, context=error.details)
        )
