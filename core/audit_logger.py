"""
Truth-Link immutable append-only audit logger (JSONL).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class TruthLinkLogger:
    """
    Append-only JSONL audit logger for compliance and provenance.
    Logs are immutable; entries are never modified or deleted.
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        """
        Initialize the audit logger.

        Args:
            log_dir: Directory for log files. Defaults to logs/ relative to project.
        """
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent.parent / "logs"
        self._log_dir = Path(log_dir)
        self._redaction_log = self._log_dir / "redactions.jsonl"
        self._query_log = self._log_dir / "queries.jsonl"

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Cannot create log directory {self._log_dir}: {e}") from e

    def log_redaction(self, entry: Dict[str, Any]) -> None:
        """
        Append a redaction event to the JSONL audit log.

        Args:
            entry: Dict with rule_id, position, timestamp, original_length, replacement, etc.
        """
        try:
            with open(self._redaction_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            raise RuntimeError(f"Failed to append redaction log: {e}") from e

    def log_query(
        self,
        user_id: str,
        query: str,
        sources: List[str],
        response_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a query event to the JSONL audit log.

        Args:
            user_id: Identifier for the user/session.
            query: The user's query text.
            sources: Document IDs or snippets used as context.
            response_hash: SHA-256 hash of the LLM response for integrity.
            metadata: Optional additional fields.
        """
        entry: Dict[str, Any] = {
            "event": "query",
            "user_id": user_id,
            "query": query,
            "sources": sources,
            "response_hash": response_hash,
            "metadata": metadata or {},
        }
        try:
            from datetime import datetime

            entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        except Exception:
            pass
        try:
            with open(self._query_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            raise RuntimeError(f"Failed to append query log: {e}") from e

    def get_audit_trail(
        self,
        log_type: str = "all",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries.

        Args:
            log_type: "redactions", "queries", or "all".
            limit: Optional max number of entries to return (most recent).

        Returns:
            List of log entries as dicts.
        """
        entries: List[Dict[str, Any]] = []
        files: List[Path] = []

        if log_type in ("redactions", "all"):
            files.append(self._redaction_log)
        if log_type in ("queries", "all"):
            files.append(self._query_log)

        try:
            for path in files:
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if limit is not None and limit > 0:
                entries = entries[-limit:]
        except OSError as e:
            raise RuntimeError(f"Failed to read audit log: {e}") from e

        return entries
