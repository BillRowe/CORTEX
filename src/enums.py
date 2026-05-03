"""
cortex2.enums  —  shared enumerations used across all regions and infrastructure.

Architecture reference:  CORTEX-ARCH-002 §3 (architecture overview), §5.1 (R1 modalities)

These enums are the type-level vocabulary of CORTEX-2. They are intentionally
exhaustive: any region that produces or consumes one of these values uses
the canonical enum rather than free-form strings, so audit logs, routing
decisions, and constitutional-constraint evaluations can pattern-match
unambiguously.

v1 enums are preserved verbatim. v2 additions are explicitly tagged.
"""

from __future__ import annotations
from enum import Enum, auto


# ─── Modality (R1 input space) ──────────────────────────────────────────────
class ModalityType(Enum):
    """Input modality of a CognitiveRepresentation.

    Encoded by R1 (Thalamus + Pulvinar). v2 expands the v1 set with five new
    modalities corresponding to the architecture's multi-modal upgrades.
    """
    TEXT             = "text"             # v1 — sentence transformer
    IMAGE            = "image"            # v2 NEW — native VLM (was CLIP in v1)
    VIDEO            = "video"            # v2 NEW — TimeSformer / VideoMAE
    AUDIO_SPEECH     = "audio_speech"     # v1 — Whisper + sentence transformer
    AUDIO_NONSPEECH  = "audio_nonspeech"  # v2 NEW — AudioMAE / CLAP
    STRUCTURED       = "structured"       # v1 — JSON / CSV / SQL
    TIMESERIES       = "timeseries"       # v2 NEW — PatchTST temporal
    MULTIDOC         = "multidoc"         # v2 NEW — cross-document fusion
    SENSOR_STREAM    = "sensor_stream"    # v1
    PHYSICAL_FEEDBACK = "physical_feedback"  # v1 — ROS2 / OPC-UA
    MULTIMODAL_FUSED = "multimodal_fused" # v2 NEW — output of cross-modal fusion


# ─── Region identifiers ─────────────────────────────────────────────────────
class RegionId(Enum):
    """Stable identifiers for the ten cortical regions and five cross-cutting
    infrastructure components. Used by R7 (ACC) for routing decisions and by
    the audit log for region-of-origin attribution.
    """
    # Cortical regions
    R1_THALAMUS_PULVINAR = "r1"
    R2_DLPFC             = "r2"
    R3_HIPPOCAMPUS       = "r3"
    R4_PFC               = "r4"
    R5_CAUDATE           = "r5"   # v2 NEW
    R6_CEREBELLUM        = "r6"
    R7_ACC               = "r7"
    R8_MOTOR_CORTEX      = "r8"
    R9_DMN               = "r9"   # v2 NEW
    R10_TOM              = "r10"  # v2 NEW
    # Cross-cutting infrastructure
    INSULAR              = "insular"
    ENTORHINAL           = "entorhinal"
    CALLOSUM             = "callosum"
    DMPFC                = "dmpfc"
    LOCUS_COERULEUS      = "locus_coeruleus"  # v2 NEW


# ─── Classification (security/handling level) ──────────────────────────────
class ClassificationLevel(Enum):
    """Classification levels propagated through every CognitiveRepresentation,
    MemoryRecord, ActionRequest, and AuditEvent. Output classification is
    bounded above by the highest classification of contributing inputs
    (CONST-004 enforces this in R7).
    """
    UNCLASSIFIED = 0
    FOUO         = 1
    CUI          = 2
    SECRET       = 3
    TOP_SECRET   = 4
    TS_SCI       = 5

    def __ge__(self, other: "ClassificationLevel") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "ClassificationLevel") -> bool:
        return self.value > other.value


# ─── Source provenance ratings (IC-style A–F × 1–6) ────────────────────────
class SourceReliability(Enum):
    """IC-rating scale for source reliability. Used by R1 when ingesting
    intelligence reports. Multiplies into the calibrated confidence on the
    UncertaintyEstimate attached to the CognitiveRepresentation.
    """
    A = "completely_reliable"
    B = "usually_reliable"
    C = "fairly_reliable"
    D = "not_usually_reliable"
    E = "unreliable"
    F = "cannot_be_judged"


class InformationCredibility(Enum):
    """IC-rating scale for information credibility (1–6). Used alongside
    SourceReliability in the UncertaintyPropagator.attach_from_source method.
    """
    CONFIRMED      = 1
    PROBABLY_TRUE  = 2
    POSSIBLY_TRUE  = 3
    DOUBTFUL       = 4
    IMPROBABLE     = 5
    CANNOT_JUDGE   = 6


# ─── Reasoning modes (R4 PFC) ───────────────────────────────────────────────
class ReasoningMode(Enum):
    """Top-level reasoning mode selected by R4 PFC. v2 adds CAUSAL_DISCOVERY
    (learning a graph from data) and ACTIVE_INFERENCE (planning to minimise
    expected free energy). System 1 cached responses bypass this enum and are
    handled by R5 Caudate directly.
    """
    DEDUCTIVE              = "deductive"
    CAUSAL_DIAGNOSTIC      = "causal_diagnostic"
    CAUSAL_PREDICTIVE      = "causal_predictive"
    CAUSAL_COUNTERFACTUAL  = "causal_counterfactual"
    PLANNING               = "planning"
    CAUSAL_DISCOVERY       = "causal_discovery"        # v2 NEW
    ACTIVE_INFERENCE       = "active_inference"        # v2 NEW


# ─── Action types (R8 Motor Cortex) ─────────────────────────────────────────
class ActionType(Enum):
    """Categories of authorised action that R8 can execute. Inherited from v1."""
    LANGUAGE_OUTPUT      = "language_output"
    STRUCTURED_DOCUMENT  = "structured_document"
    API_CALL             = "api_call"
    CODE_EXECUTION       = "code_execution"
    DATABASE_INTERACTION = "database_interaction"
    PHYSICAL_ACTUATION   = "physical_actuation"


# ─── Goal hierarchy levels (R7 ACC) ─────────────────────────────────────────
class GoalLevel(Enum):
    """Four-level goal hierarchy. v1 semantics preserved; v2 adds an
    EPISTEMIC level for active-inference information-seeking sub-goals."""
    TERMINAL     = "terminal"
    INSTRUMENTAL = "instrumental"
    PROCEDURAL   = "procedural"
    MAINTENANCE  = "maintenance"
    EPISTEMIC    = "epistemic"           # v2 NEW — active inference info-gain goals


# ─── Routing decisions (R7 ACC + Pulvinar) ──────────────────────────────────
class RoutingDecision(Enum):
    """R7's decision about which pathway handles a query.
    System 1: R5 Caudate fast path, milliseconds.
    System 2: R4 PFC deliberative path, hundreds of ms to seconds.
    Hybrid: System 1 first, escalate if confidence below threshold.
    """
    SYSTEM_1 = "system_1"
    SYSTEM_2 = "system_2"
    HYBRID   = "hybrid"


# ─── Audit event types (Entorhinal Cortex) ──────────────────────────────────
class AuditEventType(Enum):
    """Tamper-resistant audit log event types. v1 set preserved; v2 adds
    three new event types corresponding to the predictive-coding hierarchy,
    operator modelling, and glymphatic consolidation."""
    SESSION_START          = "session_start"
    SESSION_END            = "session_end"
    PERCEPTION_INPUT       = "perception_input"
    MEMORY_READ            = "memory_read"
    MEMORY_WRITE           = "memory_write"
    MEMORY_CONSOLIDATION   = "memory_consolidation"
    REASONING_START        = "reasoning_start"
    REASONING_COMPLETE     = "reasoning_complete"
    GOAL_ACTIVATED         = "goal_activated"
    GOAL_COMPLETED         = "goal_completed"
    GOAL_CONFLICT          = "goal_conflict"
    CONSTRAINT_EVALUATED   = "constraint_evaluated"
    CONSTRAINT_VIOLATED    = "constraint_violated"
    ESCALATION_TRIGGERED   = "escalation_triggered"
    ESCALATION_RESOLVED    = "escalation_resolved"
    ACTION_PROPOSED        = "action_proposed"
    ACTION_EXECUTED        = "action_executed"
    ACTION_FAILED          = "action_failed"
    LEARNING_TRIGGERED     = "learning_triggered"
    PREDICTION_ERROR       = "prediction_error"          # v2 NEW
    OPERATOR_BELIEF_UPDATE = "operator_belief_update"    # v2 NEW
    CONSOLIDATION_CYCLE    = "consolidation_cycle"       # v2 NEW
    ROUTING_DECISION       = "routing_decision"          # v2 NEW (System 1/2 split log)
