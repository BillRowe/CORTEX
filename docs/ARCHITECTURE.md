# Architecture

This document is the developer's reference for the cognitive architecture implemented in this repository. It describes the concrete software organization, data contracts, and extension points. For the conceptual motivation and theoretical justification, see [the paper][paper-link]. For installation and operational concerns, see [INSTALLATION.md](INSTALLATION.md). For the design constraints that drove these architectural choices, see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md).

---

## Overview

The architecture is a layered cognitive system organized around the canonical functions of cognition. Seven functional layers are stacked vertically, with five cross-cutting infrastructure components that span the full stack. Information flows in two directions simultaneously: upward as evidence accumulation and downward as goal-directed constraint.

```
                              ┌──────────────────────┐
                              │  Authorized outputs  │
                              └──────────▲───────────┘
                                         │
   ┌─────────────────────────────────────┴───────────┐  ◀── Cross-cutting
   │ L7  Action and Output Interface                 │      infrastructure
   │ L6  Goals, Metacognition, Alignment   ◀──── Human Op   (5 components,
   │ L5  Learning and Adaptation                     │      each spanning
   │ L4  Reasoning and Planning  (causal moat)       │      the full stack)
   │ L3  Long-Term Memory  (semantic/episodic/proc)  │
   │ L2  Working Memory and Active Context           │
   │ L1  Multimodal Perception and Grounding         │
   └─────────────────────────▲───────────────────────┘
                             │
                  ┌──────────┴──────────┐
                  │   External inputs   │
                  └─────────────────────┘
```

Each layer is a Python package implementing a defined interface. Layers communicate via versioned data contracts; no layer reaches into another layer's internals. The infrastructure components are accessible to every layer through dependency injection at construction time.

---

## Repository layout

```
src/
├── core/                  # data contracts, types, base classes
│   ├── representations.py # CognitiveRepresentation and friends
│   ├── contracts.py       # inter-layer protocol definitions
│   ├── uncertainty.py     # Confidence, Provenance types
│   └── exceptions.py      # architecture-wide exceptions
│
├── layer1_perception.py   # L1: encoders, fusion, grounding
├── layer2_working.py      # L2: workspace, salience, attention
├── layer3_memory/         # L3: three memory stores
│   ├── semantic.py        #     knowledge graph (Neo4j-backed)
│   ├── episodic.py        #     vector store (Qdrant-backed)
│   └── procedural.py      #     skill template registry
├── layer4_reasoning/      # L4: three reasoners + integration
│   ├── deductive.py       #     SMT (Z3) wrapper
│   ├── causal.py          #     DoWhy + causal-learn integration
│   ├── htn.py             #     hierarchical task network planner
│   └── orchestrator.py    #     System 1/2 routing
├── layer5_learning.py     # L5: LoRA, consolidation, compilation
├── layer6_alignment/      # L6: goals + metacognition + alignment
│   ├── goals.py           #     goal hierarchy and decomposition
│   ├── metacognition.py   #     competence boundary, self-monitoring
│   ├── constitutional.py  #     hardcoded constitutional checks
│   └── escalation.py      #     human-oversight gating
├── layer7_action.py       # L7: output formatting and dispatch
│
└── infra/                 # cross-cutting infrastructure
    ├── audit.py           # tamper-evident event log
    ├── uncertainty.py     # confidence propagation
    ├── gnn_bridge.py      # neural↔symbolic translation
    ├── oversight.py       # human-in-the-loop interface
    └── novelty.py         # locus coeruleus broadcast channel
```

The `core/` package is loaded first; everything else depends on it. Layers are loaded in numerical order. Infrastructure components are constructed once and passed to layers via dependency injection.

---

## Core types and data contracts

The foundation of the architecture is a small set of types defined in `core/representations.py` and `core/contracts.py`. Every inter-layer communication is typed, and the type system enforces structural invariants.

### `CognitiveRepresentation`

The unit of intra-system communication. Every layer produces and consumes `CognitiveRepresentation` instances or specializations thereof.

```python
@dataclass(frozen=True)
class CognitiveRepresentation:
    content: Any                       # payload (text, structured, embedding, etc.)
    modality: Modality                 # TEXT | STRUCTURED | IMAGE | AUDIO | SENSOR
    provenance: Provenance             # where this came from
    confidence: Confidence             # calibrated probability + ECE bound
    timestamp: datetime
    references: tuple[Reference, ...]  # links to memory or other reps
    metadata: Mapping[str, Any]        # layer-specific annotations
```

The class is frozen (immutable). Layers that need to "modify" a representation produce a new one with explicit `derived_from=...` provenance. This makes the audit log trivially correct and prevents whole categories of bugs.

### `Confidence`

```python
@dataclass(frozen=True)
class Confidence:
    p: float                  # probability in [0, 1]
    expected_calibration_error: float  # bound on |P(correct) − p|
    derivation: ConfidenceDerivation   # how this was computed
```

Confidence is never a bare float. The `expected_calibration_error` field forces every component that produces a confidence to also produce a bound on how miscalibrated that confidence might be. The `derivation` records whether the confidence came from model logprobs, ensemble agreement, retrieval similarity, formal proof, or some other mechanism.

### `Provenance`

```python
@dataclass(frozen=True)
class Provenance:
    source: SourceIdentifier        # where the content originated
    derivation_chain: tuple[str, ...]  # ordered list of operations applied
    reliability: float              # source-level reliability score
```

Provenance is propagated through every transformation. A representation in L7 that's about to become an output carries a complete record of how it was constructed, all the way back to the L1 input.

### Inter-layer contracts

Each pair of adjacent layers has a defined contract specifying what flows between them. These are protocols (in the `typing.Protocol` sense) implemented by both sides.

```python
class L1ToL2Contract(Protocol):
    def emit_representation(self, rep: CognitiveRepresentation) -> None: ...
    def emit_grounding(self, grounding: Grounding) -> None: ...

class L2ToL3Contract(Protocol):
    def query_semantic(self, q: SemanticQuery) -> SemanticResult: ...
    def query_episodic(self, q: EpisodicQuery) -> tuple[Episode, ...]: ...
    def query_procedural(self, q: SkillQuery) -> Skill | None: ...

# ...one contract per adjacent pair
```

A layer's only allowed dependencies are: (1) `core`, (2) infrastructure, (3) the contracts of its immediate neighbors. A layer that imports another layer's implementation directly is a contract violation and will fail review.

---

## Layer specifications

### L1 — Multimodal Perception and Grounding

**Module:** `src/layer1_perception.py`

**Responsibility:** ingest raw inputs across modalities, encode them with per-modality encoders, fuse into unified representations, and ground perceived entities to semantic-memory referents.

**Input:** raw bytes/strings via the `Layer1.ingest()` API.

**Output:** `CognitiveRepresentation` with modality-specific content and grounding metadata.

**Key classes:**

```python
class Layer1:
    encoders: dict[Modality, Encoder]   # one per modality
    fusion: FusionModule                 # combines multi-modal representations
    grounder: SymbolGrounder             # binds perceived entities to memory IDs

    def ingest(self, raw: RawInput) -> CognitiveRepresentation: ...
```

**Implementation status:**
- Text encoder: sentence-transformers (working)
- Image encoder: vision transformer (basic implementation)
- Audio encoder: stub (returns placeholder embedding)
- Sensor stream encoder: stub
- Fusion: simple concatenation + learned projection (working)
- Grounding: vector similarity search against L3 semantic store (working)

**Extension points:** new modality encoders implement the `Encoder` protocol and are registered via `Layer1.register_encoder(modality, encoder)`. The fusion module is replaceable.

---

### L2 — Working Memory and Active Context

**Module:** `src/layer2_working.py`

**Responsibility:** maintain a bounded workspace of currently-attended representations, with salience-weighted attention and cross-session context persistence.

**Input:** `CognitiveRepresentation` from L1.

**Output:** access to the current workspace contents for L3+ layers.

**Key classes:**

```python
class WorkingMemory:
    capacity: int                        # default ~50 representations
    salience: SalienceFunction           # computes attention weight
    workspace: list[CognitiveRepresentation]

    def add(self, rep: CognitiveRepresentation) -> None: ...
    def attend(self, query: AttentionQuery) -> tuple[CognitiveRepresentation, ...]: ...
    def consolidate(self) -> None: ...   # called by L5 to move old items to L3
```

**Capacity discipline:** when the workspace exceeds capacity, the least-salient items are consolidated to L3 episodic memory and removed from the workspace. The capacity constant is intentionally not "infinite context" — bounded working memory is a load-bearing architectural commitment (see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md), P2).

**Implementation status:** working. Salience function uses a learned weighting over recency, goal-relevance (signaled by L6), and novelty (signaled by infrastructure novelty channel).

---

### L3 — Long-Term Memory

**Module:** `src/layer3_memory/`

**Responsibility:** three structurally distinct memory stores, each with its own update discipline and query semantics.

**Sub-modules:**

| Store      | Backend     | Update rate | Discipline                                |
| ---------- | ----------- | ----------- | ----------------------------------------- |
| Semantic   | Neo4j       | Slow        | Conservative, requires consolidation pass |
| Episodic   | Qdrant      | Fast        | Append-on-experience                      |
| Procedural | Local       | Medium      | Compiles from successful repeated chains  |

The three-store distinction is not skin-deep — they have different APIs, different query patterns, and different update rules. Conflating them is explicitly an anti-pattern (see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md), P3).

**Key classes:**

```python
class SemanticMemory:
    def add_fact(self, subject, predicate, object, confidence) -> NodeId: ...
    def query(self, q: SemanticQuery) -> SemanticResult: ...
    def consolidate_from_episodic(self, episodes: tuple[Episode, ...]) -> None: ...

class EpisodicMemory:
    def record(self, ep: Episode) -> EpisodeId: ...
    def retrieve(self, q: EpisodicQuery, k: int = 10) -> tuple[Episode, ...]: ...

class ProceduralMemory:
    def compile_skill(self, trace: ReasoningTrace) -> Skill: ...
    def lookup(self, situation: Situation) -> Skill | None: ...
```

**Implementation status:**
- Semantic: skeleton; requires running Neo4j instance
- Episodic: skeleton; requires running Qdrant instance
- Procedural: working in-memory implementation; persistence pending

---

### L4 — Reasoning and Planning

**Module:** `src/layer4_reasoning/`

**Responsibility:** three distinct reasoners (deductive, causal, hierarchical-planning) with an orchestrator that routes queries to the appropriate engine. **This is the architectural moat** — the integration of formal causal inference into a cognitive architecture is the central technical claim of the work.

**Sub-modules:**

#### Deductive reasoner (`deductive.py`)

Wraps Z3 SMT solver. Used for queries that reduce to satisfiability or theorem-proving. Returns proofs (when found) and counterexamples (when refuted) as first-class data.

```python
class DeductiveReasoner:
    def query(self, premises: tuple[Formula, ...], goal: Formula) -> ProofResult: ...
```

#### Causal reasoner (`causal.py`)

The differentiator. Wraps DoWhy and causal-learn for causal-graph recovery, identification, and estimation. Implements Pearl's do-calculus on top of the causal-graph layer.

```python
class CausalReasoner:
    def discover_graph(self, data: pd.DataFrame, config: PCConfig) -> CausalGraph: ...
    def identify(self, graph: CausalGraph, treatment: str, outcome: str) -> Identification: ...
    def estimate(self, identification: Identification, data: pd.DataFrame) -> CausalEstimate: ...
    def sensitivity(self, estimate: CausalEstimate, **kwargs) -> SensitivityReport: ...
```

The causal reasoner is what produces the +2.51 pp answer in the headline demonstration. It is fully working and validated against the synthetic SCM in `examples/causal_demo.py`.

**Reproducibility note:** the demo uses the textbook stratified backdoor estimator. For real-world data with continuous covariates, the codebase falls back to DoWhy's regression-based estimator. The 95% CI is computed via standard error formulas under the no-unmeasured-confounding assumption.

#### HTN planner (`htn.py`)

Decomposes high-level goals into hierarchical task networks. Used by L6 when a goal cannot be satisfied by a single primitive action.

#### Orchestrator (`orchestrator.py`)

Implements System 1 / System 2 routing. A reasoning request is first attempted as a fast cached lookup against procedural memory (System 1); if no skill matches, the request is escalated to one of the three reasoners (System 2). The novelty channel (see infrastructure) gates this escalation: when novelty is high, the orchestrator routes directly to System 2 even for queries that have a System 1 hit.

**Implementation status:**
- Deductive: working
- Causal: working and validated
- HTN: working for textbook decomposition; complex domains are stubs
- Orchestrator: working

---

### L5 — Learning and Adaptation

**Module:** `src/layer5_learning.py`

**Responsibility:** four distinct learning mechanisms, each operating at a different timescale.

| Mechanism                          | Timescale | What it updates                  |
| ---------------------------------- | --------- | -------------------------------- |
| LoRA fine-tuning                   | Slow      | LLM behavior on long-horizon goals |
| Memory consolidation               | Medium    | Semantic memory from episodic    |
| Procedural compilation             | Medium    | Procedural memory from L4 traces |
| Spike-timing plasticity (analogue) | Fast      | Salience weights in L2           |

**Key classes:**

```python
class LoRAFineTuner:
    def fine_tune(self, training_data: tuple[Example, ...]) -> AdapterId: ...

class MemoryConsolidator:
    def consolidate(self, episodes: tuple[Episode, ...]) -> tuple[Fact, ...]: ...

class ProceduralCompiler:
    def compile(self, trace: ReasoningTrace) -> Skill: ...
```

**Implementation status:**
- Memory consolidation: working
- Procedural compilation: working
- LoRA: stub (signature defined; actual fine-tuning pipeline not implemented)
- Spike-timing analogue: not implemented

---

### L6 — Goals, Metacognition, Alignment

**Module:** `src/layer6_alignment/`

**Responsibility:** maintain the goal hierarchy, monitor system competence, evaluate proposed actions against constitutional constraints, and escalate to human oversight when warranted. **This is the safety layer.**

**Sub-modules:**

```
goals.py            # hierarchical goal decomposition and tracking
metacognition.py    # competence boundary monitoring
constitutional.py   # hardcoded constitutional checks
escalation.py       # human-oversight gating
```

**The six-stage authorization pipeline:** every proposed action passes through six checks before reaching L7. Failure at any stage either blocks the action or escalates to a human operator.

```
proposed action
    ↓
  [1] confidence threshold met?
    ↓ yes
  [2] within competence boundary?
    ↓ yes
  [3] consistent with active goal hierarchy?
    ↓ yes
  [4] passes constitutional checks?
    ↓ yes
  [5] reversible (or human pre-authorized)?
    ↓ yes
  [6] audit log writable?
    ↓ yes
authorized → L7
```

The five constitutional checks are intentionally hardcoded as concrete predicates, not learned. They are the architectural commitments the system cannot revise on its own.

```python
CONSTITUTIONAL_CHECKS = (
    NoIrreversibleHarm(),
    NoDeception(),
    HonorOperatorAuthority(),
    PreserveAuditability(),
    RespectScope(),
)
```

**Implementation status:** working. The escalation interface is CLI-based; a web interface is on the roadmap.

---

### L7 — Action and Output Interface

**Module:** `src/layer7_action.py`

**Responsibility:** translate authorized cognitive outputs to real-world effects across six output modalities (language, document, code, API call, structured data, alert).

**Key classes:**

```python
class ActionDispatcher:
    def dispatch(self, action: AuthorizedAction) -> ActionResult: ...

class OutputFormatter:
    def format(self, rep: CognitiveRepresentation, modality: OutputModality) -> Output: ...
```

**Implementation status:**
- Language output: working
- Code execution: working (sandboxed)
- Document generation: basic
- API call: stub
- Structured data: working
- Alert: working

---

## Cross-cutting infrastructure

Five infrastructure components span the full vertical stack and are accessible to every layer.

### Audit log (`infra/audit.py`)

Tamper-evident, SHA-256 chained event log. Every significant cognitive event — input arrival, layer transition, reasoning step, alignment check, action dispatch — emits an `AuditEvent` that is appended to the log. The chain hash makes silent tampering detectable.

```python
class AuditLog:
    def emit(self, event: AuditEvent) -> AuditEntryId: ...
    def verify_chain(self, from_id: AuditEntryId | None = None) -> ChainVerification: ...
    def export(self, format: ExportFormat) -> bytes: ...
```

The log is append-only by construction. It is **not** a debugging convenience — it is part of the safety case (see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md), P7).

### Uncertainty propagation (`infra/uncertainty.py`)

Calibrated confidence propagated through every transformation. Implements the standard uncertainty calculus over the `Confidence` type and emits `CalibrationAlert` events to the audit log when expected calibration error drifts beyond bounds.

### GNN bridge (`infra/gnn_bridge.py`)

Bidirectional translation between neural representations (vectors) and symbolic representations (graph nodes). Used by L3 (semantic memory) for entity-aware retrieval and by L4 (causal reasoner) for graph-conditioned representation learning.

**Implementation status:** GAT (graph attention) encoder requires PyTorch Geometric. The system falls back to heuristic translation if PTG is not installed.

### Human oversight interface (`infra/oversight.py`)

The single point at which a human operator can authorize, block, or override system actions. Every escalation from L6 is routed through this interface. The CLI implementation is working; a web interface is planned.

### Novelty signal (`infra/novelty.py`)

A broadcast channel modeled on the locus coeruleus. When novelty exceeds a threshold (computed as predictive surprise relative to the L3 semantic-memory expectations), the signal fires and is observable by every layer. L4 uses it to gate System 1/2 routing; L6 uses it to lower the confidence threshold for human escalation; L5 uses it to mark episodes for consolidation.

---

## Inter-layer communication

Layers communicate exclusively through their published contracts. The pattern is:

1. Layer N produces a `CognitiveRepresentation` (or specialization).
2. The representation is appended to a typed channel.
3. Layer N+1 reads from that channel through its inbound contract.

There is no shared mutable state between layers. There is no method on Layer N that Layer N+2 can call directly. This is enforced by import discipline at the package level.

```python
# layer4_reasoning/orchestrator.py
from src.core.contracts import L3ToL4Contract, L4ToL5Contract  # ✓ allowed
from src.layer3_memory import SemanticMemory                   # ✗ violation
```

Bypassing this discipline with adapter wrappers, dependency injection that smuggles concrete types, or duck-typed access patterns is also a violation. The typing system catches some of these; code review catches the rest.

---

## Reasoning subsystem detail

The L4 reasoning subsystem is the architectural differentiator. The core data flow:

```
                   ┌──────────────────────┐
                   │     Reasoning request │
                   │  (from L2 + L6 goal)  │
                   └────────────┬─────────┘
                                ↓
                       ┌────────┴─────────┐
                       │   Orchestrator   │
                       │  (System 1 or 2) │
                       └────────┬─────────┘
                                ↓
              ┌──── novelty? ──┴── procedural hit? ────┐
              ↓                                          ↓
        ┌─────┴─────┐                              ┌─────┴─────┐
        │ System 2  │                              │ System 1  │
        │           │                              │ (cached   │
        │ ┌───────┐ │                              │  skill)   │
        │ │Deduct.│ │                              └───────────┘
        │ ├───────┤ │
        │ │Causal │ │   ←── Pearl do-calculus on
        │ ├───────┤ │       discovered/specified graph
        │ │ HTN   │ │
        │ └───────┘ │
        └─────┬─────┘
              ↓
     ┌────────┴─────────┐
     │ Reasoning result │  →  to L5 (for consolidation)
     │  + audit trace   │  →  to L6 (for alignment review)
     └──────────────────┘
```

The causal reasoner specifically supports four operations:

1. **Discover** — recover a causal graph from observational data (PC algorithm by default; FCI for handling latent confounders is on the roadmap).
2. **Identify** — given a graph and a treatment/outcome pair, determine whether the causal effect is identifiable and produce the estimand.
3. **Estimate** — compute the estimand from data, with a confidence interval.
4. **Sensitivity** — analyze robustness to violations of the no-unmeasured-confounding assumption.

The headline demonstration in `examples/causal_demo.py` exercises operations 2 and 3 (the graph is given, since the SCM is synthetic). A real-world deployment would exercise all four.

---

## Memory subsystem detail

The three-store discipline (P3 in [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md)) is enforced at the API level: there is no top-level `Memory.query()` method. Code that wants to query memory must specify which store. This is friction by design — conflating semantic, episodic, and procedural memory is a well-known failure mode.

Update flow:

- **Episodic** — append-on-experience. Every significant interaction produces an episode written to Qdrant.
- **Procedural** — compiled from L4 reasoning traces by L5 when a successful chain is observed N times.
- **Semantic** — updated by L5's memory consolidation pass, which periodically scans recent episodes for facts that have stabilized across multiple episodes and admits them to the knowledge graph with appropriate provenance.

The asymmetry — episodic updates fast, semantic updates slowly — is a deliberate cognitive analogue. It also has a practical benefit: rapid contamination of the semantic store from a single bad experience is impossible by construction.

---

## Alignment pipeline detail

The six-stage authorization pipeline is in `layer6_alignment/`. Each stage emits an audit event:

```python
def authorize(self, action: ProposedAction) -> AuthorizationResult:
    self.audit.emit(StageEntered("confidence_check", action))
    if not self.confidence_meets_threshold(action):
        self.audit.emit(Blocked("confidence_below_threshold", action))
        return AuthorizationResult.blocked("confidence")

    self.audit.emit(StageEntered("competence_check", action))
    if not self.within_competence_boundary(action):
        return self.escalate(action, reason="outside_competence")

    # ... stages 3-6 follow the same pattern
```

The escalation interface presents the operator with the proposed action, the audit trace leading to it, the system's confidence and reasoning, and any relevant memory retrievals. The operator can authorize, block, or modify-and-authorize. All decisions are written to the audit log.

---

## Extension points

The architecture is designed to be extensible at well-defined seams:

- **New modality encoders:** implement the `Encoder` protocol; register with `Layer1.register_encoder`.
- **New reasoners:** implement the `Reasoner` protocol; register with the orchestrator.
- **New constitutional checks:** add to the `CONSTITUTIONAL_CHECKS` tuple; checks are evaluated in order, short-circuit on failure.
- **New memory backends:** implement the relevant `Memory` protocol; the existing Neo4j/Qdrant choices are not load-bearing.
- **New output modalities:** register a new formatter with `Layer7`.

Extension points that are deliberately closed:

- The seven layers and their numbering are architectural commitments, not configuration. Adding "Layer 8" or splitting Layer 4 should require a real design discussion, not a code change.
- The audit log format. Tampering-resistance requires stability.
- The five constitutional checks. These are the alignment foundation; revision is a deliberate, auditable decision.

---

## Glossary

- **CognitiveRepresentation** — the unit of intra-system communication; a frozen dataclass carrying content, modality, provenance, confidence, and references.
- **Confidence** — a calibrated probability with an explicit ECE bound and derivation history.
- **Provenance** — the chain of operations that produced a representation, traceable back to source.
- **Constitutional check** — a hardcoded predicate evaluated in the L6 alignment pipeline; cannot be revised by the system itself.
- **Competence boundary** — the set of query types the system has been validated on; queries outside this set are escalated.
- **Novelty signal** — a system-wide broadcast that fires when predictive surprise exceeds threshold.
- **Audit event** — a structured record of a significant cognitive event; appended to the SHA-256-chained audit log.
- **Backdoor adjustment** — Pearl's procedure for computing P(Y | do(X)) by stratifying over a sufficient adjustment set; the operation behind the +2.51 pp answer in the headline demo.
- **Identifiability** — whether a causal effect can be computed from observational data given a causal graph; not all queries are identifiable.

---

For the conceptual case for *why* this is architected this way, see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) and [the paper][paper-link].

[paper-link]: https://arxiv.org/abs/XXXX.XXXXX
