from datetime import datetime
from typing import Dict, Optional, List, Any
import logging


class ProcessingError(Exception):
    """Base error class with context."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "message": str(self),
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "type": self.__class__.__name__,
        }


class ValidationError(ProcessingError):
    """Validation errors."""

    pass


class ResourceError(ProcessingError):
    """Resource-related errors."""

    pass


class SynergyDatabaseError(ProcessingError):
    """Database connection failures."""

    pass


class ErrorAggregator:
    """Collect and manage errors."""

    def __init__(self) -> None:
        self.errors: List[Dict[str, Any]] = []

    def add_error(self, error: ProcessingError) -> None:
        """Add an error with context."""
        self.errors.append(error.to_dict())
        logging.error(f"{error.__class__.__name__}: {error}")

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total": len(self.errors),
            "types": {
                error["type"]: len(
                    [e for e in self.errors if e["type"] == error["type"]]
                )
                for error in self.errors
            },
        }
