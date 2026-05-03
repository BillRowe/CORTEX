"""
cortex2 — CORTEX-2: The Predictive Brain

Architecture reference:  CORTEX-ARCH-002
Roadmap reference:       CORTEX-ROAD-001

A 10-region cortical architecture with 5 cross-cutting infrastructure
components, organised around seven hero advances grounded in 2023–2026
research:

    #1 Predictive Cortical Hierarchy        (R9 DMN)
    #2 Dual-Process Reasoning               (R5 Caudate + R4 PFC)
    #3 Theory of Mind                       (R10 ToM Network)
    #4 Active Inference                     (R7 ACC + R4 PFC)
    #5 Causal Discovery                     (R4 PFC submodule)
    #6 Sparse Thalamic Routing              (R1 Pulvinar)
    #7 Glymphatic Consolidation             (R6 Cerebellum)

This module exposes the CORTEX facade — the primary high-level API used by
application code. Direct region access via `brain.r1` … `brain.r10` and
the cross-cutting infrastructure handles is also available for advanced
integration.

All v1 methods are preserved verbatim. v2 adds multi-modal ingestion
methods and hero-advance methods as described in CORTEX-ARCH-002 §8.2.
"""

from __future__ import annotations

# ─── Public version ─────────────────────────────────────────────────────────
__version__ = "2.0.0-skeleton"
__architecture_doc__ = "CORTEX-ARCH-002"
__roadmap_doc__ = "CORTEX-ROAD-001"

# ─── Re-export enums and data contracts at package level ────────────────────
from .enums import (
    ModalityType, RegionId, ClassificationLevel,
    SourceReliability, InformationCredibility,
    ReasoningMode, ActionType, GoalLevel, RoutingDecision, AuditEventType,
)
from .data_contracts import (
    # v1 contracts
    SourceProvenance, UncertaintyEstimate, CognitiveRepresentation,
    MemoryRecord, ReasoningStep, ReasoningChain,
    ActionRequest, ActionResult, GoalState, AuditEvent,
    # v2 NEW contracts
    DiscoveredGraph, OperatorBeliefState, PolicyEvaluation,
    PredictionError, ConsolidationCycle,
)

# ─── Region imports ─────────────────────────────────────────────────────────
from .r1_thalamus_pulvinar import (
    MultimodalPerceptionLayer, Pulvinar, GatingDecision, CrossModalFuser,
)
from .r2_dlpfc           import WorkingMemoryLayer, SessionState
from .r3_hippocampus     import LongTermMemoryLayer
from .r4_pfc             import (
    ReasoningEngine, CausalGraph, CausalEdge,
    DeductiveReasoner, CausalReasoner, CausalDiscovery,
    ActiveInferencePlanner, HTNPlanner, Plan, PlanStep,
)
from .r5_caudate         import (
    CaudateNucleus, CaudateOutcome, CachedResponse,
    FastPatternMatcher, ResponseCache,
)
from .r6_cerebellum      import (
    LearningAdaptationEngine, GlymphaticConsolidation,
    SchemaAbstractor, MemoryValuation, LoRAConfig,
)
from .r7_acc             import (
    AlignmentLayer, ConstitutionalConstraintEngine,
    GoalHierarchyManager, ActiveInferenceGoalFormulator,
    SystemRouter, MetacognitiveMonitor, EscalationRequest,
    AlignmentViolationError,
)
from .r8_motor_cortex    import (
    ActionOutputLayer, PreExecutionVerifier,
    LanguageOutputModule, StructuredDocumentGenerator,
    APICallModule, CodeExecutionModule, DatabaseInteractionModule,
    PhysicalActuationModule,
)
from .r9_default_mode_network import (
    DefaultModeNetwork, GenerativeModel, PredictionGenerator,
    HypothesisIncubator, Prediction, IncubatedHypothesis,
)
from .r10_theory_of_mind import (
    TheoryOfMindNetwork, OperatorModel, BeliefState,
    FalseBeliefTracker, OutputAdapter, MentalisingEngine,
    Belief, FrameMismatch,
)

# ─── Cross-cutting infrastructure imports ───────────────────────────────────
from .infra_uncertainty       import UncertaintyPropagator, CalibrationMonitor
from .infra_audit             import AuditLogger
from .infra_gnn_bridge        import GNNBridge, SymbolGroundingMap, SymbolBinding
from .infra_human_oversight   import (
    HumanOversightInterface, EscalationResolution,
)
from .infra_locus_coeruleus   import (
    LocusCoeruleus, NoveltyMonitor, NoveltySignal,
    SystemEngagementGate, LearningRateModulator, AttentionGain,
)


# ─── Stdlib for the facade ──────────────────────────────────────────────────
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# CORTEX  —  the public facade
# ═══════════════════════════════════════════════════════════════════════════
class CORTEX:
    """The CORTEX-2 facade. Primary high-level API for application code.

    Construction:
        brain = CORTEX()                        # all defaults
        brain = CORTEX(llm_callable=my_llm)     # inject an LLM callable
        # advanced: pass pre-constructed regions
        brain = CORTEX(r4=custom_pfc, r10=custom_tom)

    Direct region access:
        brain.r1, brain.r2, ..., brain.r10
        brain.insular, brain.entorhinal, brain.callosum,
        brain.dmpfc, brain.locus_coeruleus

    All v1 methods preserved verbatim. v2 additions cover multi-modal
    ingestion (process_image / process_video / process_audio /
    process_timeseries / process_documents / process_multimodal /
    stream_ingest) and the seven hero-advance methods.
    """

    # ─── Construction & wiring ─────────────────────────────────────────────
    def __init__(self,
                 # cortical regions (all optional, sensible defaults)
                 r1: Optional[MultimodalPerceptionLayer] = None,
                 r2: Optional[WorkingMemoryLayer]        = None,
                 r3: Optional[LongTermMemoryLayer]       = None,
                 r4: Optional[ReasoningEngine]           = None,
                 r5: Optional[CaudateNucleus]            = None,
                 r6: Optional[LearningAdaptationEngine]  = None,
                 r7: Optional[AlignmentLayer]            = None,
                 r8: Optional[ActionOutputLayer]         = None,
                 r9: Optional[DefaultModeNetwork]        = None,
                 r10: Optional[TheoryOfMindNetwork]      = None,
                 # cross-cutting infrastructure
                 insular:         Optional[UncertaintyPropagator]    = None,
                 entorhinal:      Optional[AuditLogger]              = None,
                 callosum:        Optional[GNNBridge]                = None,
                 dmpfc:           Optional[HumanOversightInterface]  = None,
                 locus_coeruleus: Optional[LocusCoeruleus]           = None,
                 # cross-region wiring helpers
                 llm_callable: Optional[Callable] = None,
                 ):
        """Instantiate the brain.

        Wiring order matters: regions consume references to other regions
        and to cross-cutting infrastructure during construction. The
        defaults below produce a fully-wired but unimplemented stub
        (every method raises NotImplementedError until Phase 0 ships).
        """
        # Cross-cutting infrastructure first (regions depend on these)
        self.insular         = insular         or UncertaintyPropagator()
        self.entorhinal      = entorhinal      or AuditLogger()
        self.callosum        = callosum        or GNNBridge()
        self.dmpfc           = dmpfc           or HumanOversightInterface()
        self.locus_coeruleus = locus_coeruleus or LocusCoeruleus()

        # R3 first (R6 depends on it)
        self.r3 = r3 or LongTermMemoryLayer()

        # R10 (R8 hooks into it for output adaptation)
        self.r10 = r10 or TheoryOfMindNetwork()

        # R6 (depends on R3)
        self.r6 = r6 or LearningAdaptationEngine(hippocampus=self.r3)

        # R4 (depends on llm_callable)
        self.r4 = r4 or ReasoningEngine(llm_callable=llm_callable)

        # R5 (System 1 cache)
        self.r5 = r5 or CaudateNucleus()

        # R7 (alignment + routing) — depends on nothing external at construction
        self.r7 = r7 or AlignmentLayer()

        # R8 (depends on R10 for output adaptation)
        self.r8 = r8 or ActionOutputLayer(
            tom_adapter=self.r10.adapt_output,
        )

        # R2 (working memory)
        self.r2 = r2 or WorkingMemoryLayer()

        # R1 (multimodal perception — last because it has the most dependencies)
        self.r1 = r1 or MultimodalPerceptionLayer()

        # R9 DMN (predictive coding hub)
        self.r9 = r9 or DefaultModeNetwork()

        # Convenience: the LLM callable for callers that want to plug their own
        self._llm = llm_callable

    # ═══════════════════════════════════════════════════════════════════════
    # v1-inherited methods  (signatures preserved verbatim)
    # ═══════════════════════════════════════════════════════════════════════

    def start_session(self,
                      session_id: str,
                      user_id: Optional[str] = None,
                      task: Optional[str] = None,
                      restore_from: Optional[str] = None) -> SessionState:
        """Start or restore a cognitive session.

        Returns the SessionState. If restore_from is provided and the
        session file is found, the buffer is rehydrated.
        """
        raise NotImplementedError("v1-inherited stub — wire to r2.start_session")

    def end_session(self,
                    session_id: str,
                    open_questions: Optional[List[str]] = None,
                    task_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """End a session. Triggers R6 consolidation; returns summary dict
        with session_state and learning_summary."""
        raise NotImplementedError("v1-inherited stub")

    def process_text(self,
                     text: str,
                     session_id: str,
                     provenance: Optional[SourceProvenance] = None,
                     retrieve_memory: bool = True,
                     top_k_memory: int = 5) -> CognitiveRepresentation:
        """v1-inherited primary text-ingestion path.

        Wiring:
            R1.perceive_text  →  insular.attach_from_source  →  R2.add_to_context
            then optional R3.retrieve  →  R2.integrate_memory_retrieval.
        """
        raise NotImplementedError("v1-inherited stub")

    def reason(self,
               query: str,
               session_id: str,
               mode: ReasoningMode = ReasoningMode.DEDUCTIVE,
               causal_graph: Optional[CausalGraph] = None,
               **kwargs: Any) -> ReasoningChain:
        """v1-inherited reasoning entry point.

        v2 routing: if R7 routes to System 1, the query is handled by R5
        Caudate; only on cache miss / low confidence does it escalate to
        R4 PFC. For explicitly System 2 modes (causal counterfactual,
        active inference), the router is bypassed.
        """
        raise NotImplementedError("v1+v2 stub")

    def respond(self,
                content: Any,
                session_id: str,
                action_type: ActionType = ActionType.LANGUAGE_OUTPUT,
                classification: ClassificationLevel = ClassificationLevel.UNCLASSIFIED,
                uncertainty: Optional[UncertaintyEstimate] = None,
                operator_id: Optional[str] = None,
                **kwargs: Any) -> ActionResult:
        """v1-inherited response emission. Builds an ActionRequest, runs it
        through R7 authorisation, then dispatches to R8.

        v2: if operator_id is set, R8 adapts the output via R10.
        """
        raise NotImplementedError("v1+v2 stub")

    def add_goal(self,
                 description: str,
                 session_id: str,
                 level: GoalLevel = GoalLevel.INSTRUMENTAL,
                 priority: float = 0.5,
                 parent_id: Optional[str] = None) -> str:
        """v1-inherited. Add a goal; returns goal_id."""
        return self.r7.add_goal(description, level, priority, parent_id)

    def status(self) -> Dict[str, Any]:
        """v1-inherited. Return high-level system status (version,
        working_memory, learning, action stats)."""
        raise NotImplementedError("v1-inherited stub")

    # ═══════════════════════════════════════════════════════════════════════
    # v2 NEW — multi-modal ingestion
    # ═══════════════════════════════════════════════════════════════════════

    def process_image(self,
                      image_bytes: bytes,
                      session_id: str,
                      prompt: Optional[str] = None,
                      provenance: Optional[SourceProvenance] = None
                      ) -> CognitiveRepresentation:
        """Ingest a still image via R1's native VLM. Returns a
        CognitiveRepresentation with modality=ModalityType.IMAGE."""
        raise NotImplementedError("v2 stub — wire to r1.perceive_image")

    def process_video(self,
                      video_bytes: bytes,
                      session_id: str,
                      segment_seconds: Optional[float] = None,
                      provenance: Optional[SourceProvenance] = None
                      ) -> CognitiveRepresentation:
        """Ingest a video via R1's TimeSformer/VideoMAE encoder. Returns
        a CognitiveRepresentation with per-segment embeddings + temporal
        trajectory in metadata."""
        raise NotImplementedError("v2 stub — wire to r1.perceive_video")

    def process_audio(self,
                      audio_bytes: bytes,
                      session_id: str,
                      mode: str = "asr",
                      provenance: Optional[SourceProvenance] = None
                      ) -> CognitiveRepresentation:
        """Ingest audio. mode='asr' speech-to-text; 'features' non-speech;
        'both' returns the merged representation."""
        raise NotImplementedError("v2 stub — wire to r1.perceive_audio")

    def process_timeseries(self,
                           values: List[float],
                           timestamps: Optional[List[float]] = None,
                           session_id: str = "default",
                           schema: Optional[Dict[str, Any]] = None,
                           provenance: Optional[SourceProvenance] = None
                           ) -> CognitiveRepresentation:
        """Ingest a time-series via R1's PatchTST encoder."""
        raise NotImplementedError("v2 stub — wire to r1.perceive_timeseries")

    def process_documents(self,
                          documents: List[CognitiveRepresentation],
                          session_id: str,
                          fusion: str = "cross_attn",
                          provenance: Optional[SourceProvenance] = None
                          ) -> CognitiveRepresentation:
        """Fuse multiple documents into a single CognitiveRepresentation
        with cross-document attention. fusion: 'concat' | 'cross_attn' |
        'weighted_mean'."""
        raise NotImplementedError("v2 stub — wire to r1.perceive_documents")

    def process_multimodal(self,
                           inputs: Dict[ModalityType, Any],
                           session_id: str,
                           provenance: Optional[SourceProvenance] = None
                           ) -> CognitiveRepresentation:
        """Fuse simultaneous multi-modal inputs into a single
        CognitiveRepresentation. Fusion happens BEFORE Pulvinar gating
        (per CORTEX-ARCH-002 §8.4 key design decision)."""
        raise NotImplementedError("v2 stub — wire to r1.perceive_multimodal")

    async def stream_ingest(self,
                            stream: Any,
                            session_id: str,
                            modality: ModalityType,
                            batch_seconds: float = 10.0
                            ) -> AsyncIterator[CognitiveRepresentation]:
        """Async iterator for high-throughput stream ingestion (gRPC, Kafka)."""
        raise NotImplementedError("v2 stub — wire to r1.stream_ingest")
        # Required for type-checker: this function is an async generator.
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    # ═══════════════════════════════════════════════════════════════════════
    # v2 NEW — hero-advance methods
    # ═══════════════════════════════════════════════════════════════════════

    # ─── Hero #5  —  Causal Discovery ──────────────────────────────────────
    def discover_causal_graph(self,
                              data: Dict[str, List[float]],
                              algorithm: str = "pc",
                              alpha: float = 0.01,
                              exogenous: Optional[List[str]] = None
                              ) -> DiscoveredGraph:
        """Discover a causal DAG from observational data. See
        r4.causal_discovery.discover() for full semantics."""
        return self.r4.causal_discovery.discover(
            data=data, algorithm=algorithm, alpha=alpha, exogenous=exogenous
        )

    def reason_with_discovered_graph(self,
                                     query: str,
                                     graph: DiscoveredGraph,
                                     treatment: Optional[str] = None,
                                     outcome: Optional[str] = None,
                                     mode: ReasoningMode = ReasoningMode.CAUSAL_PREDICTIVE,
                                     **kwargs: Any) -> ReasoningChain:
        """Run interventional reasoning over a previously-discovered graph.
        The returned ReasoningChain has graph_provenance set to graph.id."""
        raise NotImplementedError("v2 stub — wire CausalDiscovery → CausalReasoner")

    # ─── Hero #3  —  Theory of Mind ────────────────────────────────────────
    def model_operator(self,
                       operator_id: str,
                       traits: Optional[Dict[str, Any]] = None,
                       role: Optional[str] = None) -> OperatorBeliefState:
        """Build or retrieve an operator model. Returns the public belief-state."""
        return self.r10.model_operator(operator_id, traits, role)

    def update_operator_belief(self,
                               operator_id: str,
                               observation: CognitiveRepresentation
                               ) -> OperatorBeliefState:
        """Revise the operator model based on a single observation."""
        return self.r10.update_operator_belief(operator_id, observation)

    def get_operator_state(self, operator_id: str) -> Optional[OperatorBeliefState]:
        """Snapshot the current operator model. None if no model exists."""
        return self.r10.get_operator_state(operator_id)

    def adapt_output(self, content: Any, operator_id: str) -> Any:
        """Re-frame content for a specific operator. Falls back to the
        unadapted content if no operator model exists."""
        return self.r10.adapt_output(content, operator_id)

    # ─── Hero #4  —  Active Inference ──────────────────────────────────────
    def plan_active_inference(self,
                              goal_state: Dict[str, Any],
                              candidate_actions: List[str],
                              prior_beliefs: Optional[Dict[str, float]] = None
                              ) -> List[PolicyEvaluation]:
        """Evaluate candidate actions under expected free energy. Returned
        list is sorted by ascending EFE (best first)."""
        return self.r4.active_inference.plan(
            goal_state=goal_state,
            candidate_actions=candidate_actions,
            prior_beliefs=prior_beliefs,
        )

    # ─── Hero #1  —  Predictive Hierarchy ──────────────────────────────────
    def get_prediction(self, region: RegionId) -> Optional[Prediction]:
        """Return R9's most recent top-down prediction for a target region."""
        return self.r9.get_prediction(region)

    def get_prediction_error(self,
                             region: RegionId) -> Optional[NoveltySignal]:
        """Return the most recent prediction-error magnitude for a region.
        The NoveltySignal carries both the magnitude and the gating decision
        derived from it (System 1 vs System 2 recommendation)."""
        return self.locus_coeruleus.current_signal(region)

    def incubate_hypotheses(self,
                            seconds: float,
                            context: Optional[Dict[str, Any]] = None
                            ) -> List[IncubatedHypothesis]:
        """Run the DMN mind-wandering loop for `seconds`. Hypotheses are
        stored in R3 episodic memory and surface when matching queries
        arrive subsequently."""
        return self.r9.incubate_hypotheses(seconds, context)

    # ─── Hero #7  —  Glymphatic Consolidation ──────────────────────────────
    def trigger_consolidation(self,
                              scope: str = "nightly") -> ConsolidationCycle:
        """Run a glymphatic consolidation cycle. scope: 'nightly' (scheduler-
        initiated) | 'manual' (operator-initiated)."""
        return self.r6.trigger_consolidation(scope)

    def get_consolidation_log(self, last_n: int = 10) -> List[ConsolidationCycle]:
        """Return the last N consolidation cycle records for audit."""
        return self.r6.get_consolidation_log(last_n)

    # ─── Heroes #2 / #6  —  Routing & sparse-expert telemetry ──────────────
    def get_routing_stats(self) -> Dict[str, Any]:
        """Per-route counts (System 1 vs System 2) plus average outcome
        correctness over the recent window. Drives the operator dashboard."""
        return self.r7.get_routing_stats()

    def report_region_load(self) -> Dict[RegionId, float]:
        """Per-region utilisation as gated by the Pulvinar over the recent
        window. Used to tune the gating budget and to detect under-/over-
        engaged regions."""
        return self.r1.pulvinar.report_load()


# ─── __all__  (explicit public API) ────────────────────────────────────────
__all__ = [
    # version + doc refs
    "__version__", "__architecture_doc__", "__roadmap_doc__",
    # facade
    "CORTEX",
    # enums
    "ModalityType", "RegionId", "ClassificationLevel",
    "SourceReliability", "InformationCredibility",
    "ReasoningMode", "ActionType", "GoalLevel", "RoutingDecision",
    "AuditEventType",
    # v1 data contracts
    "SourceProvenance", "UncertaintyEstimate", "CognitiveRepresentation",
    "MemoryRecord", "ReasoningStep", "ReasoningChain",
    "ActionRequest", "ActionResult", "GoalState", "AuditEvent",
    # v2 data contracts
    "DiscoveredGraph", "OperatorBeliefState", "PolicyEvaluation",
    "PredictionError", "ConsolidationCycle",
    # regions
    "MultimodalPerceptionLayer", "Pulvinar", "GatingDecision", "CrossModalFuser",
    "WorkingMemoryLayer", "SessionState",
    "LongTermMemoryLayer",
    "ReasoningEngine", "CausalGraph", "CausalEdge",
    "DeductiveReasoner", "CausalReasoner", "CausalDiscovery",
    "ActiveInferencePlanner", "HTNPlanner", "Plan", "PlanStep",
    "CaudateNucleus", "CaudateOutcome", "CachedResponse",
    "FastPatternMatcher", "ResponseCache",
    "LearningAdaptationEngine", "GlymphaticConsolidation",
    "SchemaAbstractor", "MemoryValuation", "LoRAConfig",
    "AlignmentLayer", "ConstitutionalConstraintEngine",
    "GoalHierarchyManager", "ActiveInferenceGoalFormulator",
    "SystemRouter", "MetacognitiveMonitor", "EscalationRequest",
    "AlignmentViolationError",
    "ActionOutputLayer", "PreExecutionVerifier",
    "LanguageOutputModule", "StructuredDocumentGenerator",
    "APICallModule", "CodeExecutionModule", "DatabaseInteractionModule",
    "PhysicalActuationModule",
    "DefaultModeNetwork", "GenerativeModel", "PredictionGenerator",
    "HypothesisIncubator", "Prediction", "IncubatedHypothesis",
    "TheoryOfMindNetwork", "OperatorModel", "BeliefState",
    "FalseBeliefTracker", "OutputAdapter", "MentalisingEngine",
    "Belief", "FrameMismatch",
    # infrastructure
    "UncertaintyPropagator", "CalibrationMonitor",
    "AuditLogger",
    "GNNBridge", "SymbolGroundingMap", "SymbolBinding",
    "HumanOversightInterface", "EscalationResolution",
    "LocusCoeruleus", "NoveltyMonitor", "NoveltySignal",
    "SystemEngagementGate", "LearningRateModulator", "AttentionGain",
]
