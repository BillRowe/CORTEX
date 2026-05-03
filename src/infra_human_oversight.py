"""
cortex2.infra_human_oversight  —  Dorsomedial PFC · HITL escalation

Architecture reference:  CORTEX-ARCH-002 §7.2 Updates to existing infrastructure.

Single point of human escalation. When R7 (ACC) cannot auto-resolve a
decision (low confidence, competence-boundary breach, irreversible action,
goal conflict, hard-constraint near-miss), it produces an EscalationRequest
that flows here.

The component is interface-only: the actual UI / Slack-bot / pager
integration is deployment-specific. v1 ships an in-process queue + console
prompt as the default reference implementation; v2 adds optional
ToM-awareness so the escalation message is framed for the specific operator
who will receive it (per their OperatorBeliefState).

Status: UNCHANGED structurally; v2 adds one new framing method.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from queue import Queue
import uuid

from .data_contracts import OperatorBeliefState
from .r7_acc import EscalationRequest


@dataclass
class EscalationResolution:
    """Operator's response to an EscalationRequest."""
    request_id:    str
    decision:      str                # "approve" | "deny" | "modify"
    rationale:     str = ""
    modified_action: Optional[Any] = None
    operator_id:   Optional[str] = None
    resolved_at:   datetime = field(default_factory=datetime.utcnow)


class HumanOversightInterface:
    """The HITL escalation queue.

    Default implementation: in-process Queue + console prompt callback.
    Production deployments should swap the prompt callback for a real
    integration (Slack bot, web UI, pager).
    """

    def __init__(self,
                 prompt_callback: Optional[Callable[[EscalationRequest], EscalationResolution]] = None,
                 default_timeout_seconds: int = 600):
        self.prompt_callback = prompt_callback
        self.default_timeout_seconds = default_timeout_seconds
        self._pending: "Queue[EscalationRequest]" = Queue()
        self._resolved: Dict[str, EscalationResolution] = {}

    def submit(self, request: EscalationRequest) -> EscalationResolution:
        """Block until resolution. Honours request.timeout_seconds.

        If request.blocking is False, returns a synthesised "deny" resolution
        immediately and queues for offline review.
        """
        raise NotImplementedError("v1-inherited stub")

    def submit_async(self, request: EscalationRequest) -> str:
        """Non-blocking submission. Returns request_id; caller polls
        get_resolution()."""
        raise NotImplementedError("v1-inherited stub")

    def get_resolution(self, request_id: str) -> Optional[EscalationResolution]:
        return self._resolved.get(request_id)

    def frame_for_operator(self,
                            request: EscalationRequest,
                            belief_state: Optional[OperatorBeliefState]
                            ) -> str:
        """v2 NEW. Render the escalation message in the operator's frame.

        Uses the operator's role, vocabulary preference, and known
        knowledge gaps to produce framing that minimises misunderstanding.
        Falls back to a generic message if no belief_state is available.
        """
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.3)")
