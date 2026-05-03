"""
cortex2.r4_pfc  —  R4 Prefrontal Cortex · System 2 deliberation

Architecture reference:  CORTEX-ARCH-002 §5.2 R4 PFC
Roadmap reference:       v1 reasoning preserved + Phase 2 (active inference)
                                                 + Phase 3 (causal discovery)
Hero advances served:    #4 Active Inference, #5 Causal Discovery,
                          (and as System 2 partner to #2 Dual-Process Reasoning)

PFC retains all v1 reasoning capabilities (deductive Z3 SMT, Pearl
do-calculus over given graphs, HTN planning) but is no longer the default
reasoning pathway — it engages only when R7 routes to it via the
System 1/2 router.

v2 adds two submodules:

    CausalDiscovery         (hero #5)
        Learns causal graphs from observational data via PC algorithm,
        NOTEARS, or LiNGAM. Closes the v1 loop where graphs had to be
        supplied by domain experts.

    ActiveInferencePlanner  (hero #4)
        Plans information-seeking actions to reduce expected free energy.
        Yields principled curiosity and exploration as derived behaviour.

Status: MODIFIED from v1 (was layer4_reasoning in LCIF / cortex v1).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple
import uuid

from .data_contracts import (
    CognitiveRepresentation, ReasoningChain, ReasoningStep,
    UncertaintyEstimate, GoalState, DiscoveredGraph, PolicyEvaluation,
)
from .enums import ReasoningMode, RegionId, GoalLevel


# ═══════════════════════════════════════════════════════════════════════════
# v1-inherited reasoning submodules (signatures preserved)
# ═══════════════════════════════════════════════════════════════════════════

class DeductiveReasoner:
    """v1 inherited. Z3 SMT solver for formal deductive reasoning.

    Falls back to keyword-coverage analysis when z3-solver not installed.
    Reports identified_gaps explicitly when premises are insufficient —
    never confabulates.
    """

    def __init__(self):
        self._solver = None  # lazy-load z3

    def reason(self, premises: List[str], query: str) -> ReasoningChain:
        raise NotImplementedError("v1-inherited stub — implement via z3-solver")


@dataclass
class CausalEdge:
    """An edge in a CausalGraph."""
    parent:    str
    child:     str
    relation:  str = "causes"
    weight:    Optional[float] = None
    uncertainty: Optional[UncertaintyEstimate] = None


@dataclass
class CausalGraph:
    """A directed acyclic graph used by CausalReasoner. v1 schema preserved.

    CausalDiscovery (v2 NEW) outputs DiscoveredGraph instances; those are
    converted to this CausalGraph form before being handed to CausalReasoner.
    """
    nodes:     List[str] = field(default_factory=list)
    edges:     List[CausalEdge] = field(default_factory=list)
    metadata:  Dict[str, Any] = field(default_factory=dict)


class CausalReasoner:
    """v1 inherited. Pearl do-calculus over a given CausalGraph.

    Three causal query types: diagnostic (what causes Y?), predictive
    (P(Y | do(X=x))), counterfactual (had X been x', what would Y be?).
    Backed by DoWhy when installed; falls back to structural-path analysis
    otherwise.
    """

    def __init__(self):
        self._dowhy = None  # lazy-load

    def reason_diagnostic(self, graph: CausalGraph, effect: str) -> ReasoningChain:
        raise NotImplementedError("v1-inherited stub — implement via DoWhy")

    def reason_predictive(self, graph: CausalGraph, treatment: str,
                          outcome: str, value: Any,
                          data: Optional[Any] = None) -> ReasoningChain:
        raise NotImplementedError("v1-inherited stub")

    def reason_counterfactual(self, graph: CausalGraph,
                              factual: Dict[str, Any],
                              counterfactual: Dict[str, Any],
                              outcome: str) -> ReasoningChain:
        raise NotImplementedError("v1-inherited stub")

    def find_backdoor_set(self, graph: CausalGraph,
                          treatment: str, outcome: str) -> List[str]:
        """Identify the backdoor adjustment set for an intervention query."""
        raise NotImplementedError("v1-inherited stub")


# ═══════════════════════════════════════════════════════════════════════════
# v2 NEW  —  CausalDiscovery
# ═══════════════════════════════════════════════════════════════════════════

class CausalDiscovery:
    """v2 NEW (hero #5). Learns a causal DAG from observational data.

    Three algorithms supported:
        - 'pc'       PC algorithm (constraint-based, conditional independence)
        - 'notears'  NOTEARS continuous optimisation (score-based)
        - 'lingam'   LiNGAM (linear non-Gaussian acyclic model)

    All three are exposed via the same interface; the choice depends on
    data characteristics (LiNGAM needs non-Gaussian errors; NOTEARS scales
    better to many variables; PC is the most general).

    Background-knowledge priors (e.g., "this variable is exogenous and
    cannot have parents") are passed via `exogenous` and are the standard
    mechanism for breaking Markov-equivalence-class ties in production.
    """

    def __init__(self, default_alpha: float = 0.01):
        self.default_alpha = default_alpha
        self._causal_learn = None  # lazy-load
        self._lingam = None        # lazy-load

    def discover(self,
                 data: Dict[str, List[float]],
                 algorithm: str = "pc",
                 alpha: Optional[float] = None,
                 exogenous: Optional[List[str]] = None
                 ) -> DiscoveredGraph:
        """Recover a causal DAG from observational data.

        Args:
            data: variable name → list of values (n samples each).
            algorithm: 'pc' | 'notears' | 'lingam'.
            alpha: significance level for CI tests (PC only); default 0.01.
            exogenous: variables that cannot have parents (background prior).

        Returns:
            DiscoveredGraph with skeleton, directed edges, sepsets, full
            CI-test trace for audit, and recovery_confidence score.
        """
        raise NotImplementedError("v2 stub — implement in Phase 3 (D3.1) via causal-learn / lingam")

    def to_causal_graph(self, discovered: DiscoveredGraph) -> CausalGraph:
        """Convert a DiscoveredGraph into a CausalGraph the v1
        CausalReasoner can operate on."""
        raise NotImplementedError("v2 stub — straightforward conversion")


# ═══════════════════════════════════════════════════════════════════════════
# v2 NEW  —  ActiveInferencePlanner
# ═══════════════════════════════════════════════════════════════════════════

class ActiveInferencePlanner:
    """v2 NEW (hero #4). Plans actions by minimising expected free energy.

    Expected free energy decomposes into pragmatic value (goal attainment)
    and epistemic value (information gain). The planner ranks candidate
    actions by EFE; the action minimising EFE is the recommended policy.

    Backed by pymdp when installed. The cleanest demonstration is the
    "should I ask a clarifying question?" decision: when expected
    information gain from asking exceeds the expected cost of asking,
    the planner recommends asking; otherwise it recommends acting.
    """

    def __init__(self):
        self._pymdp = None  # lazy-load

    def evaluate_action(self,
                        action: str,
                        goal_state: Dict[str, Any],
                        prior_beliefs: Dict[str, float]
                        ) -> PolicyEvaluation:
        """Compute EFE for one candidate action.

        Returns a PolicyEvaluation with:
            - expected_free_energy   (to MINIMISE)
            - pragmatic_value        (goal-attainment component)
            - epistemic_value        (information-gain component)
            - posterior_outcomes     (P(outcome | action))
        """
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.4) via pymdp")

    def plan(self,
             goal_state: Dict[str, Any],
             candidate_actions: List[str],
             prior_beliefs: Optional[Dict[str, float]] = None
             ) -> List[PolicyEvaluation]:
        """Evaluate all candidate actions; return them sorted by ascending
        expected free energy (best first)."""
        raise NotImplementedError("v2 stub — implement in Phase 2 (D2.4)")


# ═══════════════════════════════════════════════════════════════════════════
# HTN Planner  (v1 inherited)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    """One step in an HTN plan."""
    action:           str
    preconditions:    List[str] = field(default_factory=list)
    risk_level:       str = "low"           # "low" | "medium" | "high"
    is_reversible:    bool = True
    estimated_effects: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """HTN planner output. Embeds an ActionRequest if the plan calls for
    a system action; the ActionRequest is then routed through R7 for
    authorisation before R8 may execute."""
    id:              str = field(default_factory=lambda: str(uuid.uuid4()))
    steps:           List[PlanStep] = field(default_factory=list)
    requires_human_authorization: bool = False
    constraints_verified: bool = False
    risks_identified: List[str] = field(default_factory=list)


class HTNPlanner:
    """v1 inherited. LangGraph-based hierarchical task network planner.

    Falls back to direct LLM planning when LangGraph not installed.
    """

    def __init__(self, llm_callable: Optional[Callable] = None):
        self.llm_callable = llm_callable

    def plan(self, goal: str, context: List[CognitiveRepresentation],
             constraints: Optional[List[str]] = None) -> Plan:
        raise NotImplementedError("v1-inherited stub")


# ═══════════════════════════════════════════════════════════════════════════
# Reasoning engine  —  the region's public face
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    """R4 — System 2 deliberation. Public interface used by R7 (ACC)
    when routing to System 2 and by the CORTEX facade for explicit
    reasoning calls.
    """
    region_id = RegionId.R4_PFC

    def __init__(self,
                 deductive: Optional[DeductiveReasoner] = None,
                 causal: Optional[CausalReasoner] = None,
                 causal_discovery: Optional[CausalDiscovery] = None,
                 htn: Optional[HTNPlanner] = None,
                 active_inference: Optional[ActiveInferencePlanner] = None,
                 llm_callable: Optional[Callable] = None):
        self.deductive        = deductive or DeductiveReasoner()
        self.causal           = causal or CausalReasoner()
        self.causal_discovery = causal_discovery or CausalDiscovery()
        self.htn              = htn or HTNPlanner(llm_callable)
        self.active_inference = active_inference or ActiveInferencePlanner()
        self.llm_callable     = llm_callable

    def reason(self, query: str, mode: ReasoningMode,
               context: Optional[List[CognitiveRepresentation]] = None,
               causal_graph: Optional[CausalGraph] = None,
               **kwargs: Any) -> ReasoningChain:
        """v1-inherited entry point. Dispatches to the appropriate
        submodule based on the reasoning mode.

        v2 adds two new modes:
            - CAUSAL_DISCOVERY  (kwargs: data, algorithm, alpha, exogenous)
            - ACTIVE_INFERENCE  (kwargs: goal_state, candidate_actions)
        """
        raise NotImplementedError("v1+v2 dispatch stub — implement dispatch table")

    def discover_and_reason(self,
                            data: Dict[str, List[float]],
                            query: str,
                            treatment: str,
                            outcome: str,
                            algorithm: str = "pc",
                            exogenous: Optional[List[str]] = None
                            ) -> Tuple[DiscoveredGraph, ReasoningChain]:
        """v2 convenience method (hero #5).

        End-to-end pipeline: discover the causal graph from data, identify
        the backdoor set, run the interventional reasoning, and return both
        the DiscoveredGraph (for audit) and the ReasoningChain (with
        graph_provenance set to the DiscoveredGraph id).
        """
        raise NotImplementedError("v2 stub — wire CausalDiscovery + CausalReasoner")
