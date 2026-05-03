"""
cortex2.r7_acc  —  R7 Anterior Cingulate Cortex · alignment + routing

Architecture reference:  CORTEX-ARCH-002 §5.4 R7 ACC
Roadmap reference:       v1 alignment preserved + Phase 1 (System 1/2 routing)
                                                + Phase 2 (active inference goals)
Hero advances served:    #2 Dual-Process (routing), #4 Active Inference (goals)

R7 retains all v1 alignment capabilities (five hardcoded constitutional
constraints, six-stage authorisation pipeline, HITL escalation).

v2 adds two responsibilities:

    SystemRouter         Decides whether each query goes to R5 (System 1
                          Caudate) or R4 (System 2 PFC) based on prediction
                          error magnitude (from R9 / Locus Coeruleus),
                          stakes/reversibility, and operator-belief
                          uncertainty (from R10).

    ActiveInferenceGoals Goals are reformulated as preferred world-states.
                          Goal selection minimises expected free energy.
                          Curiosity emerges as derived behaviour.

Status: MODIFIED from v1 (was layer6_alignment in LCIF / cortex v1).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from .data_contracts import (
    ActionRequest, GoalState, ReasoningChain, OperatorBeliefState,
    PolicyEvaluation, PredictionError, AuditEvent,
)
from .enums import (
    GoalLevel, RoutingDecision, ClassificationLevel, RegionId, AuditEventType,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constitutional constraints  (v1 inherited, hardcoded)
# ═══════════════════════════════════════════════════════════════════════════

class AlignmentViolationError(Exception):
    """Raised when a hard constraint is violated. Aborts the action and
    creates a CONSTRAINT_VIOLATED audit event."""


@dataclass
class Constraint:
    """A single constitutional or policy constraint. Each carries a check
    function returning True if the constraint is satisfied."""
    constraint_id:  str
    name:           str
    description:    str
    is_hard:        bool                       # True for CONST-001..005
    check:          Callable[[ActionRequest], bool]


class ConstitutionalConstraintEngine:
    """v1 inherited. Hardcoded five-constraint engine + operator-defined
    policy constraint layer.

    The five hard constraints (CONST-001 through CONST-005) are not
    learnable, not configurable, and not modifiable at runtime.
    """

    CONST_IDS = ["CONST-001", "CONST-002", "CONST-003", "CONST-004", "CONST-005"]

    def __init__(self):
        self._constraints: List[Constraint] = []
        self._init_hardcoded()

    def _init_hardcoded(self) -> None:
        """Wire the five non-learnable constraints. Implementations are
        independent of any LLM or external config.

        v1-inherited stub: the production implementation registers five
        Constraint objects with id-strings CONST-001 through CONST-005,
        each carrying a check function that runs without an LLM and
        without external configuration. The stub is a deliberate no-op
        so the constructor can succeed for type-checking and import-time
        validation; calling evaluate() on this engine before the real
        implementation lands will raise NotImplementedError there."""
        # No-op stub; production wiring deferred to Phase 0 (D0.7).
        return

    def add_policy_constraint(self, constraint: Constraint) -> None:
        """Add an operator-defined soft constraint."""
        if constraint.is_hard:
            raise ValueError("Policy constraints must not be hard")
        self._constraints.append(constraint)

    def evaluate(self, action: ActionRequest) -> List[Constraint]:
        """Returns the list of constraints that the action violates.
        Empty list means the action passes."""
        raise NotImplementedError("v1-inherited stub")


# ═══════════════════════════════════════════════════════════════════════════
# Goal management  (v1 hierarchy preserved + v2 active-inference extension)
# ═══════════════════════════════════════════════════════════════════════════

class GoalHierarchyManager:
    """v1 inherited. Manages the four-level goal hierarchy.

    v2 extends with the EPISTEMIC level for active-inference information-
    seeking sub-goals. Conflict detection unchanged: priority inversions
    and maintenance-instrumental blocks are auto-resolved or escalated.
    """

    def __init__(self):
        self._goals: Dict[str, GoalState] = {}

    def add_goal(self, description: str, level: GoalLevel,
                 priority: float, parent_id: Optional[str] = None) -> str:
        raise NotImplementedError("v1-inherited stub")

    def complete_goal(self, goal_id: str) -> None:
        raise NotImplementedError("v1-inherited stub")

    def detect_conflicts(self) -> List[Tuple[GoalState, GoalState]]:
        raise NotImplementedError("v1-inherited stub")

    def active_goals(self) -> List[GoalState]:
        return [g for g in self._goals.values() if not g.completed]


class ActiveInferenceGoalFormulator:
    """v2 NEW (hero #4). Reformulates goals as preferred world-states and
    selects actions that minimise expected free energy.

    The PFC's ActiveInferencePlanner does the per-action EFE evaluation;
    this class manages the goal-state representation and translates
    user-supplied descriptive goals ("reduce churn") into the preferred-
    world-state form active inference operates on.
    """

    def __init__(self, planner: Any = None):
        # `planner` is an R4.ActiveInferencePlanner reference, kept loose
        # to avoid cross-region import in stubs.
        self._planner = planner

    def goal_to_preferred_state(self, goal: GoalState) -> Dict[str, Any]:
        """Translate a descriptive GoalState into a preferred-world-state
        the planner can compute EFE against."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.5)")

    def add_epistemic_subgoal(self, parent_goal_id: str,
                              uncertainty_target: str) -> str:
        """Add an EPISTEMIC sub-goal whose purpose is to reduce uncertainty
        about a specific variable. This is how active-inference curiosity
        manifests in the goal hierarchy."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.5)")


# ═══════════════════════════════════════════════════════════════════════════
# Metacognition  (v1 inherited)
# ═══════════════════════════════════════════════════════════════════════════

class MetacognitiveMonitor:
    """v1 inherited. Tracks predicted-vs-actual accuracy via Expected
    Calibration Error (ECE). Alerts when ECE > 0.15.

    v2 reuse: the same monitor consumes prediction errors from the
    predictive hierarchy (PredictionError dataclass) — this is what the
    Insular Cortex now derives uncertainty from rather than only from
    ensemble disagreement.
    """

    def __init__(self, n_bins: int = 10, ece_alert_threshold: float = 0.15):
        self.n_bins = n_bins
        self.ece_alert_threshold = ece_alert_threshold
        self._predictions: List[Tuple[float, bool]] = []

    def record(self, predicted_confidence: float, was_correct: bool) -> None:
        raise NotImplementedError("v1-inherited stub")

    def expected_calibration_error(self) -> float:
        raise NotImplementedError("v1-inherited stub")

    def assess_reasoning_quality(self, chain: ReasoningChain) -> float:
        """Used by the authorisation pipeline (check 4 of 6)."""
        raise NotImplementedError("v1-inherited stub")


# ═══════════════════════════════════════════════════════════════════════════
# v2 NEW  —  System 1/2 router
# ═══════════════════════════════════════════════════════════════════════════

class SystemRouter:
    """v2 NEW. Decides whether a query goes to R5 (System 1) or R4 (System 2).

    Three inputs determine the decision:
        - prediction-error magnitude     (from R9 / Locus Coeruleus)
        - action stakes / reversibility  (from R7 itself)
        - operator-belief uncertainty    (from R10)

    Thresholds are learned per-domain via reinforcement from outcome
    feedback. Conservative default: route to System 2 whenever any input
    exceeds its threshold.
    """

    def __init__(self,
                 prediction_error_threshold: float = 0.30,
                 stakes_threshold: float = 0.50,
                 operator_uncertainty_threshold: float = 0.40):
        self.prediction_error_threshold = prediction_error_threshold
        self.stakes_threshold = stakes_threshold
        self.operator_uncertainty_threshold = operator_uncertainty_threshold
        self._decision_history: List[Tuple[Dict[str, float], RoutingDecision, bool]] = []

    def decide(self,
               prediction_error: Optional[PredictionError],
               action_stakes: float,
               operator_belief_state: Optional[OperatorBeliefState]
               ) -> RoutingDecision:
        """Make the routing decision.

        Returns:
            RoutingDecision.SYSTEM_1 if all inputs are below threshold.
            RoutingDecision.SYSTEM_2 if any input exceeds its threshold.
            RoutingDecision.HYBRID if System 1 should attempt first.
        """
        raise NotImplementedError("v2 stub — implement in Phase 1 (D1.2)")

    def record_outcome(self,
                        signals: Dict[str, float],
                        decision: RoutingDecision,
                        was_correct: bool) -> None:
        """Append a decision-outcome triple to the history; used by
        learn_thresholds()."""
        self._decision_history.append((signals, decision, was_correct))

    def learn_thresholds(self) -> None:
        """Learn the decision thresholds from accumulated history. Run
        offline (during glymphatic phase) to avoid live thrash."""
        raise NotImplementedError("v2 stub — implement in Phase 1 enhancement")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Per-route counts + average outcome-correctness. Used by the
        CORTEX facade method get_routing_stats()."""
        raise NotImplementedError("v2 stub")


# ═══════════════════════════════════════════════════════════════════════════
# Authorisation pipeline  (v1 inherited, structure preserved)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EscalationRequest:
    """Sent to the Dorsomedial PFC human-oversight interface when the
    pipeline cannot auto-resolve."""
    id:                 str = field(default_factory=lambda: str(uuid.uuid4()))
    reason:             str = ""
    proposed_action:    Optional[ActionRequest] = None
    blocking:           bool = True
    timeout_seconds:    int = 600
    operator_id:        Optional[str] = None      # v2 — for ToM-aware framing


# ═══════════════════════════════════════════════════════════════════════════
# AlignmentLayer  (the region as a whole)
# ═══════════════════════════════════════════════════════════════════════════

class AlignmentLayer:
    """R7 — alignment, routing, goal management.

    Public interface used by the CORTEX facade. Every proposed action passes
    through check_and_authorize() before R8 may execute it.
    """
    region_id = RegionId.R7_ACC

    def __init__(self,
                 constraints: Optional[ConstitutionalConstraintEngine] = None,
                 goals: Optional[GoalHierarchyManager] = None,
                 metacog: Optional[MetacognitiveMonitor] = None,
                 router: Optional[SystemRouter] = None,
                 ai_goals: Optional[ActiveInferenceGoalFormulator] = None,
                 low_confidence_threshold: float = 0.40,
                 competence_threshold: float = 0.45):
        self.constraints  = constraints or ConstitutionalConstraintEngine()
        self.goals        = goals or GoalHierarchyManager()
        self.metacog      = metacog or MetacognitiveMonitor()
        self.router       = router or SystemRouter()
        self.ai_goals     = ai_goals or ActiveInferenceGoalFormulator()
        self.low_confidence_threshold = low_confidence_threshold
        self.competence_threshold = competence_threshold
        self._competence_domains: List[str] = []

    # ── v1 inherited ──────────────────────────────────────────────────────
    def check_and_authorize(self, action: ActionRequest,
                            chain: Optional[ReasoningChain] = None) -> ActionRequest:
        """Six-check authorisation pipeline (v1 verbatim).

        Order:
            1. Constitutional + policy constraints (hard violations → raise)
            2. Confidence threshold
            3. Competence boundary
            4. Reasoning-quality check (via MetacognitiveMonitor)
            5. Goal-conflict check
            6. Irreversible-action check (always escalates)

        Sets `alignment_approved=True` on the returned ActionRequest if all
        checks pass; otherwise raises AlignmentViolationError or returns an
        action whose alignment_approved remains False (with an
        EscalationRequest queued).
        """
        raise NotImplementedError("v1-inherited stub")

    def add_goal(self, description: str, level: GoalLevel = GoalLevel.INSTRUMENTAL,
                 priority: float = 0.5, parent_id: Optional[str] = None) -> str:
        """Public entry point for the CORTEX facade."""
        return self.goals.add_goal(description, level, priority, parent_id)

    def update_active_goals(self) -> List[GoalState]:
        """Returns the currently active goals; signals R2 to recompute
        salience based on the new goal set."""
        return self.goals.active_goals()

    # ── v2 NEW ────────────────────────────────────────────────────────────
    def route(self,
              prediction_error: Optional[PredictionError],
              action_stakes: float,
              operator_belief_state: Optional[OperatorBeliefState]
              ) -> RoutingDecision:
        """Public routing entry point. Delegates to SystemRouter.

        Called by the CORTEX facade for every reasoning request; the
        return value selects between R5 (Caudate) and R4 (PFC).
        """
        return self.router.decide(prediction_error, action_stakes, operator_belief_state)

    def get_routing_stats(self) -> Dict[str, Any]:
        """Public accessor for the CORTEX facade method get_routing_stats()."""
        return self.router.get_routing_stats()
