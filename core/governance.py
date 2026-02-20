"""
Deterministic PII redaction via regex. No AI used for governance.
"""

import re
from datetime import datetime
from typing import Dict, List, NamedTuple, Tuple

from core.config import PII_PATTERNS


class RedactionEntry(NamedTuple):
    """Single redaction event for audit trail."""

    rule_id: str
    position: int
    timestamp: str
    original_length: int
    replacement: str


class DeterministicGovernance:
    """
    Deterministic governance for PII redaction using regex only.
    All redactions are logged for Truth-Link audit compliance.
    """

    def __init__(self) -> None:
        """Initialize compiled regex patterns."""
        self._patterns: Dict[str, re.Pattern] = {}
        try:
            for rule_id, pattern_str in PII_PATTERNS.items():
                self._patterns[rule_id] = re.compile(pattern_str)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in config: {e}") from e

    def redact_pii(self, text: str) -> Tuple[str, List[RedactionEntry]]:
        """
        Redact PII from text using deterministic regex patterns.

        Args:
            text: Raw text that may contain PII.

        Returns:
            Tuple of (redacted_text, audit_trail).
            audit_trail contains RedactionEntry for each match.
        """
        if not text:
            return "", []

        redacted = text
        audit_trail: List[RedactionEntry] = []

        try:
            for rule_id, pattern in self._patterns.items():
                matches = list(pattern.finditer(redacted))
                for match in reversed(matches):
                    position = match.start()
                    original = match.group()
                    replacement = self._get_replacement(rule_id, len(original))
                    redacted = redacted[:position] + replacement + redacted[position + len(original) :]
                    audit_trail.append(
                        RedactionEntry(
                            rule_id=rule_id,
                            position=position,
                            timestamp=datetime.utcnow().isoformat() + "Z",
                            original_length=len(original),
                            replacement=replacement,
                        )
                    )
        except Exception as e:
            raise RuntimeError(f"Redaction failed: {e}") from e

        return redacted, audit_trail

    def _get_replacement(self, rule_id: str, length: int) -> str:
        """
        Generate deterministic replacement string based on rule and length.

        Args:
            rule_id: Identifier of the PII rule (ssn, phone, email, credit_card).
            length: Length of the original matched string.

        Returns:
            Placeholder string (e.g., [REDACTED-SSN]).
        """
        placeholders = {
            "ssn": "[REDACTED-SSN]",
            "phone": "[REDACTED-PHONE]",
            "email": "[REDACTED-EMAIL]",
            "credit_card": "[REDACTED-CARD]",
        }
        return placeholders.get(rule_id, "[REDACTED]")
