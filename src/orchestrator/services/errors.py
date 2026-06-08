"""Domain exception hierarchy for the service layer.

Replaces FastAPI ``HTTPException`` raises with typed exceptions that the
TUI catches directly.  No status codes — callers pattern-match on the
exception type.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for all service-layer errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class NotFoundError(ServiceError):
    """Resource does not exist (was HTTP 404)."""


class ValidationError(ServiceError):
    """Input validation failure (was HTTP 400/422)."""


class PolicyViolation(ServiceError):
    """Action denied by scope/risk policy (was HTTP 403)."""


class PermanentFailure(ServiceError):
    """Infrastructure unavailable, e.g. WeasyPrint missing (was HTTP 503)."""


class CriticVeto(ServiceError):
    """PreExploitCritic vetoed the action (was HTTP 409)."""


class ApprovalRequired(ServiceError):
    """Action requires human approval before proceeding (was HTTP 202).

    ``approval_id`` is populated when an ApprovalRow was created so the
    caller can reference it.
    """

    def __init__(self, message: str, approval_id: str | None = None) -> None:
        self.approval_id = approval_id
        super().__init__(message)
