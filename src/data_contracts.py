"""
cortex2.data_contracts  —  the dataclasses crossing every region boundary.

Architecture reference:  CORTEX-ARCH-002 §8.3 Key data contracts.

v1 contracts are preserved verbatim — v1 client code that deserialises a v2
CognitiveRepresentation continues to work because the new fields are optional.
v2 introduces five new dataclasses for the seven hero advances:

    DiscoveredGraph       (#5 — Causal Discovery)
    OperatorBeliefState   (#3 — Theory of Mind)
    PolicyEvaluation      (#4 — Active Inference)
    PredictionError       (#1 — Predictive Hierarchy)
    ConsolidationCycle    (#7 — Glymphatic Consolidation)

Every dataclass is intentionally pure-Python, dependency-free, and serialisable
to JSON. The only typing imports are from the standard library.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, FrozenSet
import uuid

from .enums import (
    ModalityType, ClassificationLevel, SourceReliability,
    InformationCredibility, ReasoningMode, ActionType,
    GoalLevel, RoutingDecision, AuditEventType, RegionId,
)


# ═══════════════════════════════════════════════════════════════════════════
# v1 contracts (schemas unchanged in v2)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SourceProvenance:
    """Per-source attribution attached to a CognitiveRepresentation.

    Inherited from v1. Used by infra_uncertainty.UncertaintyPropagator to
    compute calibrated confidence at ingestion time.
    """
    source_id:    str
    source_type:  str                              # "human" | "sensor" | "document" | "api" | ...
    reliability:  Optional[SourceReliability] = None
    credibility:  Optional[InformationCredibility] = None
    timestamp:    datetime = field(default_factory=datetime.utcnow)
    metadata:     Dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyEstimate:
    """Calibrated uncertainty attached to any cognitive output.

    Inherited from v1. v2 extends usage: the predictive hierarchy in R9
    populates `epistemic` from prediction-error variance rather than only
    from ensemble disagreement.
    """
    confidence:   float                       # [0,1] — calibrated probability
    epistemic:    Optional[float] = None      # uncertainty due to limited knowledge
    aleatoric:    Optional[float] = None      # uncertainty due to inherent noise
    method:       str = "uncalibrated"        # "fisher_z" | "ensemble" | "logprob" | "predictive_variance" (v2)
    notes:        Optional[str] = None


@dataclass
class CognitiveRepresentation:
    """Universal perception output. Every R1 ingest path produces one of these.

    Schema unchanged from v1; v2 expands the modality space (ModalityType
    enum) without changing the dataclass shape. Cross-modal fused outputs
    populate `metadata['fusion_sources']` with the contributing modalities.
    """
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    content:             Any = None
    modality:            ModalityType = ModalityType.TEXT
    embedding:           Optional[List[float]] = None    # 768-dim default
    grounding_evidence:  Dict[str, Any] = field(default_factory=dict)
    provenance:          Optional[SourceProvenance] = None
    uncertainty:         Optional[UncertaintyEstimate] = None
    classification:      ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    metadata:            Dict[str, Any] = field(default_factory=dict)
    created_at:          datetime = field(default_factory=datetime.utcnow)


@dataclass
class MemoryRecord:
    """Universal memory output across the three R3 stores (semantic, episodic,
    procedural). Inherited from v1.
    """
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type:         str = "semantic"      # "semantic" | "episodic" | "procedural"
    content:             Any = None
    embedding:           Optional[List[float]] = None
    similarity_score:    Optional[float] = None
    classification:      ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    uncertainty:         Optional[UncertaintyEstimate] = None
    metadata:            Dict[str, Any] = field(default_factory=dict)
    created_at:          datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReasoningStep:
    """One step in a ReasoningChain. v1 schema."""
    premise:             str
    rule:                str
    conclusion:          str
    uncertainty:         Optional[UncertaintyEstimate] = None
    symbolic_node_ids:   List[str] = field(default_factory=list)


@dataclass
class ReasoningChain:
    """Output of R4 PFC. v1 schema preserved; v2 adds optional graph_provenance
    field populated when the chain was produced by reasoning over a graph
    discovered by CausalDiscovery rather than supplied by an expert.
    """
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    mode:                ReasoningMode = ReasoningMode.DEDUCTIVE
    query:               str = ""
    steps:               List[ReasoningStep] = field(default_factory=list)
    conclusion:          str = ""
    uncertainty:         Optional[UncertaintyEstimate] = None
    premises_sufficient: bool = True
    identified_gaps:     List[str] = field(default_factory=list)
    graph_provenance:    Optional[str] = None        # v2 — id of DiscoveredGraph if applicable
    metadata:            Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRequest:
    """Authorisation pipeline payload. Every proposed action passes through R7
    check_and_authorize() before R8 may execute it. Inherited from v1.
    """
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type:         ActionType = ActionType.LANGUAGE_OUTPUT
    parameters:          Dict[str, Any] = field(default_factory=dict)
    reversible:          bool = True
    estimated_effects:   List[str] = field(default_factory=list)
    alignment_approved:  bool = False              # MUST be True before R8 executes
    uncertainty:         Optional[UncertaintyEstimate] = None
    classification:      ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    operator_id:         Optional[str] = None      # v2 — who is the authorising operator
    metadata:            Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result returned by R8 after action execution. Inherited from v1."""
    action_id:           str
    success:             bool
    output:              Any = None
    elapsed_ms:          float = 0.0
    rollback_data:       Optional[Dict[str, Any]] = None
    error:               Optional[str] = None
    classification:      ClassificationLevel = ClassificationLevel.UNCLASSIFIED


@dataclass
class GoalState:
    """Goal hierarchy entry. v1 schema. v2 adds support for EPISTEMIC level
    goals via the GoalLevel enum extension."""
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    description:         str = ""
    level:               GoalLevel = GoalLevel.INSTRUMENTAL
    priority:            float = 0.5
    parent_id:           Optional[str] = None
    embedding:           Optional[List[float]] = None
    completed:           bool = False
    metadata:            Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Tamper-resistant audit log entry. SHA-256 checksum populated by the
    Entorhinal Cortex on write. Inherited from v1; v2 adds new event types
    via AuditEventType enum extension.
    """
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type:          AuditEventType = AuditEventType.SESSION_START
    region:              Optional[RegionId] = None
    session_id:          Optional[str] = None
    timestamp:           datetime = field(default_factory=datetime.utcnow)
    classification:      ClassificationLevel = ClassificationLevel.UNCLASSIFIED
    payload:             Dict[str, Any] = field(default_factory=dict)
    checksum:            Optional[str] = None      # SHA-256 of canonical content


# ═══════════════════════════════════════════════════════════════════════════
# v2 NEW contracts
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveredGraph:
    """Output of R4.CausalDiscovery (hero #5).

    Captures everything a downstream consumer needs: the recovered graph,
    the conditional-independence test trace (auditable), the exogenous
    priors used, and a recovery-confidence score that downstream callers
    can use to decide whether to trust the graph for interventional
    reasoning or to require human review.
    """
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    variables:           List[str] = field(default_factory=list)
    skeleton:            List[FrozenSet[str]] = field(default_factory=list)   # undirected edges
    directed_edges:      List[Tuple[str, str]] = field(default_factory=list)  # (parent, child)
    sepsets:             Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # frozen-set key serialised
    ci_test_log:         List[Dict[str, Any]] = field(default_factory=list)
    exogenous_priors:    List[str] = field(default_factory=list)
    algorithm:           str = "pc"               # "pc" | "notears" | "lingam"
    alpha:               float = 0.01
    n_samples:           int = 0
    recovery_confidence: float = 0.0
    metadata:            Dict[str, Any] = field(default_factory=dict)
    created_at:          datetime = field(default_factory=datetime.utcnow)


@dataclass
class OperatorBeliefState:
    """Output of R10 Theory of Mind (hero #3).

    Structured representation of what the system thinks the operator
    believes, knows, wants, and prefers. Used by R8 for output adaptation
    and by R7 for escalation framing.
    """
    operator_id:           str
    role:                  Optional[str] = None
    knowledge_frame:       str = ""
    decision_horizon:      str = ""
    vocabulary_preference: str = ""
    typical_skepticism:    str = ""
    beliefs:               Dict[str, float] = field(default_factory=dict)   # claim → confidence operator holds
    knowledge_gaps:        List[str] = field(default_factory=list)
    frame_mismatches:      List[str] = field(default_factory=list)
    confidence:            float = 0.5            # how confident R10 is in this model
    last_updated:          datetime = field(default_factory=datetime.utcnow)
    history_observations:  int = 0                # number of operator turns observed
    metadata:              Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvaluation:
    """Output of R4.ActiveInferencePlanner (hero #4).

    Evaluation of a single candidate action under the active-inference
    objective. Lower expected_free_energy = preferred action. Pragmatic and
    epistemic value are reported separately so the operator can see whether
    a given recommendation is goal-pursuing or information-gaining.
    """
    candidate_action:      str
    expected_free_energy:  float                  # to MINIMISE
    pragmatic_value:       float                  # goal-attainment component
    epistemic_value:       float                  # information-gain component
    posterior_outcomes:    Dict[str, float] = field(default_factory=dict)  # outcome → P(outcome | action)
    rationale:             str = ""
    metadata:              Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionError:
    """Bottom-up message in the predictive hierarchy (hero #1).

    Emitted by lower regions (R1, R2) when an input does not match the
    top-down prediction from R9 (DMN). Magnitude gates whether R7 routes
    to System 2 and whether the Locus Coeruleus boosts learning rate.
    """
    id:                  str = field(default_factory=lambda: str(uuid.uuid4()))
    source_region:       RegionId = RegionId.R1_THALAMUS_PULVINAR
    target_region:       RegionId = RegionId.R9_DMN
    magnitude:           float = 0.0              # L2 distance between prediction and observation
    signed_residual:     Optional[List[float]] = None
    context_embedding:   Optional[List[float]] = None
    novelty_score:       float = 0.0              # gated by Locus Coeruleus
    timestamp:           datetime = field(default_factory=datetime.utcnow)
    metadata:            Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationCycle:
    """Output of R6.GlymphaticConsolidation (hero #7).

    Audit record of a single offline consolidation phase. Used both for
    transparency (operators can see what the system "learned during sleep")
    and for the regression-test gate that prevents harmful consolidations
    from being committed (the Phase 3 G3 success criterion).
    """
    id:                       str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at:               datetime = field(default_factory=datetime.utcnow)
    finished_at:              Optional[datetime] = None
    episodes_replayed:        int = 0
    schemas_extracted:        int = 0
    memories_pruned:          int = 0
    lora_steps:               int = 0
    pre_test_accuracy:        Optional[float] = None
    post_test_accuracy:       Optional[float] = None
    regression_detected:      bool = False
    committed:                bool = False        # False if regression gate vetoed
    summary:                  str = ""
    metadata:                 Dict[str, Any] = field(default_factory=dict)
