"""
cortex2.r10_theory_of_mind  —  R10 Theory of Mind Network · TPJ + Medial PFC

Architecture reference:  CORTEX-ARCH-002 §4.3 R10 Theory of Mind Network
Roadmap reference:       CORTEX-ROAD-001 §4 Phase 2 (M6–M9) — Social & Agentic
Hero advance:            #3 Theory of Mind

R10 maintains an explicit, structured model of the operator: their beliefs,
knowledge state, goals, frame, and mental model of the system. Every output
the system produces is shaped by this model. The same recommendation is
rendered differently for a CFO than for an engineering lead, not because
the content differs but because R10 knows the operator's frame and adapts
the framing.

Crucially, R10 tracks the difference between what the operator believes
and what is actually true (the false-belief space) — this is the same
Sally-Anne capability that current LLMs systematically fail.

Benchmark targets (M9 Gate G2):
    FANToM   ≥ 85%   (current GPT-4 class: 60–70%)
    BigToM   ≥ 85%   (current GPT-4 class: 65–75%)

Status: v2 NEW — no LCIF / v1 predecessor.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .data_contracts import (
    OperatorBeliefState, CognitiveRepresentation, ActionRequest,
    GoalState, UncertaintyEstimate,
)
from .enums import RegionId


# ─── Internal records ───────────────────────────────────────────────────────
@dataclass
class Belief:
    """A single attributed belief held by an operator.

    R10 distinguishes between (a) what the operator currently believes,
    (b) what the operator knows (subset of beliefs that are also true),
    and (c) what the operator does not know but the system does (the
    knowledge-gap space — important for output adaptation).
    """
    claim:        str
    confidence:   float                       # operator's confidence in the claim
    is_true:      Optional[bool] = None       # ground truth, if known
    last_seen:    datetime = field(default_factory=datetime.utcnow)
    source:       Optional[str] = None        # how R10 came to attribute this belief


@dataclass
class FrameMismatch:
    """A detected mismatch between operator's frame and system's frame.

    Used by R8 Motor Cortex to address frame mismatches in outputs ("you
    may be thinking of X but this is actually Y") rather than letting them
    propagate silently.
    """
    operator_frame:  str
    system_frame:    str
    severity:        float                  # [0,1] how consequential the mismatch is
    context:         Dict[str, Any] = field(default_factory=dict)


# ─── OperatorModel ──────────────────────────────────────────────────────────
class OperatorModel:
    """Per-operator belief-state store. Persisted across sessions.

    In production this is backed by Postgres (operator-model store, see
    §8.1 external interfaces). Per-user separation is enforced at this
    layer; cross-operator inference is never permitted.
    """

    def __init__(self, operator_id: str, role: Optional[str] = None):
        self.operator_id = operator_id
        self.role = role
        self.beliefs: Dict[str, Belief] = {}
        self.knowledge_gaps: List[str] = []
        self.frame_mismatches: List[FrameMismatch] = []
        self.history_observations = 0

    def attribute_belief(self, claim: str, confidence: float,
                         source: Optional[str] = None) -> None:
        """Attribute a belief to the operator based on observed behaviour."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def update_from_observation(self, observation: CognitiveRepresentation) -> None:
        """Revise the operator model based on a single observed turn.

        Each observation may add new beliefs, revise confidence on existing
        ones, or flag a frame mismatch. The history_observations counter
        increments so callers can gauge model maturity.
        """
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def to_belief_state(self) -> OperatorBeliefState:
        """Snapshot the current operator model into a serialisable
        OperatorBeliefState dataclass for downstream consumers.
        """
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def confidence(self) -> float:
        """How confident is R10 in this operator model? Drives whether
        downstream consumers trust the model for output adaptation."""
        raise NotImplementedError("v2 stub")


# ─── BeliefState + FalseBeliefTracker ──────────────────────────────────────
class BeliefState:
    """Lightweight read-only view over an OperatorModel for downstream
    consumers (R4, R7, R8). Implements the public OperatorBeliefState
    semantics without exposing mutable internals.
    """

    def __init__(self, model: OperatorModel):
        self._model = model

    def believes(self, claim: str) -> Optional[Belief]:
        """Returns the operator's attributed belief about `claim`, or None."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def knows(self, claim: str) -> bool:
        """True iff the operator both believes the claim AND it's true."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def has_gap_about(self, topic: str) -> bool:
        """True iff the operator does not know something the system does."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")


class FalseBeliefTracker:
    """Tracks the false-belief space — claims the operator believes that are
    actually false, or that the operator does not believe but are actually
    true.

    This is the architectural locus of the Sally-Anne capability. The tracker
    surfaces false beliefs to R8 so outputs can address them explicitly,
    and to R7 so escalations can be framed accordingly.
    """

    def __init__(self):
        self._false_beliefs: List[Tuple[str, str, Belief]] = []

    def detect(self, model: OperatorModel,
               ground_truth: Dict[str, bool]) -> List[Belief]:
        """Identify operator beliefs that contradict the ground-truth map.
        ground_truth maps `claim → is_true`."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")


# ─── OutputAdapter ──────────────────────────────────────────────────────────
class OutputAdapter:
    """The output-adaptation pipeline. Every user-facing text from R8
    consults R10 via this adapter before generation.

    Adaptation dimensions:
        - vocabulary_preference  (technical / commercial / board / casual)
        - decision_horizon       (this quarter / next sprint / strategic)
        - knowledge_frame        (what the operator already knows)
        - typical_skepticism     (what the operator will push back on)
        - false_belief_addressal (any false beliefs to address explicitly)
    """

    def __init__(self):
        pass

    def adapt(self,
              content: Any,
              belief_state: BeliefState,
              context: Optional[Dict[str, Any]] = None) -> Any:
        """Re-frame `content` according to the operator belief state.

        Production implementation: invokes a small fast LM (Phi-3-mini or
        Llama-3.2-1B per §9 tech stack) with a structured prompt encoding
        the operator's frame. Returns adapted content of the same type.
        """
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.3)")


# ─── MentalisingEngine ──────────────────────────────────────────────────────
class MentalisingEngine:
    """The core inference engine for theory-of-mind reasoning.

    Given an observation of operator behaviour and the prior operator model,
    update the model. The implementation runs a small LM with a structured
    mentalising prompt (the same structure as the FANToM and BigToM
    benchmark prompts), then parses the output into Belief and FrameMismatch
    updates.
    """

    def __init__(self, lm_backbone: str = "phi-3-mini"):
        self.lm_backbone = lm_backbone

    def infer_beliefs(self,
                      operator_turn: CognitiveRepresentation,
                      prior_model: OperatorModel) -> List[Belief]:
        """Infer the operator's beliefs from a single turn."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def infer_goals(self,
                    operator_turn: CognitiveRepresentation,
                    prior_model: OperatorModel) -> List[GoalState]:
        """Infer the operator's currently-active goals from a turn."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def detect_frame_mismatch(self,
                              operator_frame_evidence: str,
                              system_frame: str) -> Optional[FrameMismatch]:
        """Detect when the operator's frame diverges from the system's."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")


# ─── TheoryOfMindNetwork  (the region itself) ──────────────────────────────
class TheoryOfMindNetwork:
    """R10 — operator modelling, mentalising, output adaptation.

    Public interface:
        - model_operator(operator_id, ...)      build/retrieve a model
        - update_operator_belief(...)           revise based on observation
        - get_operator_state(operator_id)       snapshot for callers
        - adapt_output(content, operator_id)    used by R8 before output
    """
    region_id = RegionId.R10_TOM

    def __init__(self,
                 mentaliser: Optional[MentalisingEngine] = None,
                 adapter: Optional[OutputAdapter] = None,
                 false_belief_tracker: Optional[FalseBeliefTracker] = None):
        self.mentaliser = mentaliser or MentalisingEngine()
        self.adapter = adapter or OutputAdapter()
        self.false_belief_tracker = false_belief_tracker or FalseBeliefTracker()
        self._models: Dict[str, OperatorModel] = {}

    def model_operator(self,
                       operator_id: str,
                       traits: Optional[Dict[str, Any]] = None,
                       role: Optional[str] = None) -> OperatorBeliefState:
        """Construct or retrieve an OperatorModel and return its public
        belief-state snapshot."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def update_operator_belief(self,
                               operator_id: str,
                               observation: CognitiveRepresentation) -> OperatorBeliefState:
        """Update the operator model based on a new observation; return the
        revised belief-state snapshot."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def get_operator_state(self, operator_id: str) -> Optional[OperatorBeliefState]:
        """Public accessor; returns None if no model exists for the operator."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.1)")

    def adapt_output(self, content: Any, operator_id: str) -> Any:
        """Re-frame the content for the specified operator. Falls back to
        the unadapted content if no operator model exists."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.3)")
