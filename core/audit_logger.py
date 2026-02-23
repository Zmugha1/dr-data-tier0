"""
Truth-Link immutable append-only audit logger (JSONL).
Pipeline audit logger (steps 17-19) for deterministic hash chain.
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


class AuditLogger:
    """Steps 17-19: Hash chain and version pinning for pipeline batches."""

    def __init__(self) -> None:
        self.log_dir = Path(__file__).resolve().parent.parent / "data" / "audit_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self.previous_hash = "0" * 64

        last_log = sorted(self.log_dir.glob("batch_*.json"))
        if last_log:
            try:
                with open(last_log[-1], encoding="utf-8") as f:
                    data = json.load(f)
                    self.previous_hash = data.get("batch_hash", self.previous_hash)
            except Exception:
                pass

    def log_ingestion(self, filename: str, content_hash: str, chunks: list) -> None:
        """Log each document ingestion."""
        from datetime import datetime

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "DOCUMENT_INGESTED",
            "filename": filename,
            "content_hash": content_hash,
            "chunk_count": len(chunks),
            "previous_hash": self.previous_hash,
        }
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["entry_hash"] = entry_hash
        self.previous_hash = hashlib.sha256(
            (self.previous_hash + entry_hash).encode()
        ).hexdigest()
        self.entries.append(entry)

    def finalize_batch(
        self,
        doc_count: int = 0,
        total_chunks: int = 0,
        total_entities: int = 0,
        total_relations: int = 0,
    ) -> Dict[str, Any]:
        """Create batch manifest with full hash chain."""
        from datetime import datetime

        manifest: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "batch_hash": self.previous_hash,
            "entry_count": len(self.entries),
            "entries": self.entries,
            "doc_count": doc_count,
            "total_chunks": total_chunks,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "versions": {
                "embedding_model": "nomic-embed-text",
                "extraction_model": "phi4:latest",
                "chunk_size": 512,
                "code_commit": "unknown",
            },
        }
        filename = f"batch_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
        with open(self.log_dir / filename, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(self.log_dir / "latest_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest
