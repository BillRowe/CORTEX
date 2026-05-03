"""
cortex2.infra_audit  —  Entorhinal Cortex · audit log

Architecture reference:  CORTEX-ARCH-002 §7.2 Updates to existing infrastructure.

Every significant cognitive event generates an AuditEvent with a SHA-256
checksum for tamper detection. Events are stored in a thread-safe in-memory
deque (configurable max 100,000 events) with optional JSON-lines file
persistence for compliance archiving.

v1 event types preserved verbatim. v2 adds:
    PREDICTION_ERROR        large prediction errors that engaged System 2
    OPERATOR_BELIEF_UPDATE  R10 model revisions
    CONSOLIDATION_CYCLE     R6 glymphatic phase summaries
    ROUTING_DECISION        System 1/2 split log

Status: MODIFIED from v1 (was infra_audit in LCIF / cortex v1).
"""

from __future__ import annotations
import hashlib
import json
import threading
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from .data_contracts import AuditEvent
from .enums import AuditEventType, RegionId, ClassificationLevel


class AuditLogger:
    """Tamper-resistant cognitive event log.

    SHA-256 checksum is computed over the canonical JSON serialisation of
    the event payload at write time. Tampering with a persisted event can
    be detected by recomputing the checksum.

    For production compliance deployments, the JSON-lines audit file should
    be written to an immutable storage backend (S3 Object Lock, WORM-enabled
    NAS, or a blockchain-anchored log).
    """

    def __init__(self,
                 max_in_memory: int = 100_000,
                 jsonl_path: Optional[str] = None):
        self.max_in_memory = max_in_memory
        self.jsonl_path = jsonl_path
        self._buffer: Deque[AuditEvent] = deque(maxlen=max_in_memory)
        self._lock = threading.Lock()

    def log(self,
            event_type: AuditEventType,
            region: Optional[RegionId] = None,
            session_id: Optional[str] = None,
            payload: Optional[Dict[str, Any]] = None,
            classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED
            ) -> AuditEvent:
        """Append a new audit event with SHA-256 checksum."""
        raise NotImplementedError("v1-inherited stub")

    def compute_checksum(self, event: AuditEvent) -> str:
        """Canonical JSON + SHA-256."""
        raise NotImplementedError("v1-inherited stub")

    def verify_checksum(self, event: AuditEvent) -> bool:
        """Recompute and compare against stored checksum."""
        raise NotImplementedError("v1-inherited stub")

    def get_events(self,
                   event_types: Optional[List[AuditEventType]] = None,
                   session_id: Optional[str] = None,
                   region: Optional[RegionId] = None,
                   since: Optional[datetime] = None,
                   limit: int = 100) -> List[AuditEvent]:
        """Filter events from the in-memory buffer."""
        raise NotImplementedError("v1-inherited stub")

    def export_jsonl(self, path: str) -> int:
        """Export the entire buffer to a JSON-lines file. Returns count
        written."""
        raise NotImplementedError("v1-inherited stub")
