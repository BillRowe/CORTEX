# Design Principles

The architecture is governed by seven principles. Every component traces back to one or more of them. This document expands each principle with the problem it addresses, how it is instantiated in the architecture, what it rules out (the anti-patterns), and the trade-off it accepts.

The principles themselves are stated in Section 4 of [the paper][paper-link]. This document is the operational counterpart: it should be useful when you're trying to decide whether a proposed change to the codebase is consistent with the architecture's commitments. If a change violates one of these principles, it requires a real design discussion, not a quick code review.

---

## P1. No single paradigm is sufficient

> Neural pattern recognition, formal symbolic reasoning, and cognitive-architecture machinery are integrated. Each compensates for failure modes of the others.

### The problem

Three decades of AI research have repeatedly shown that pure-neural systems have specific failure modes (causal reasoning, calibration, formal verification), pure-symbolic systems have different specific failure modes (robustness, knowledge acquisition, perception), and pure cognitive-architecture systems have a third set (scaling, learning, real-world grounding). Each paradigm has a fix for the others' weaknesses.

The mistake the field has made repeatedly is to assume that one paradigm will eventually solve all of them. It hasn't, and the structural reasons it hasn't are not going away with more compute.

### How it's instantiated

- **Neural** components: per-modality encoders (L1), embedding-based retrieval (L3 episodic), attention/salience (L2), the LLM(s) used as components by L4 and L7.
- **Symbolic** components: Z3 deductive reasoner (L4), DoWhy causal reasoner (L4), HTN planner (L4), constitutional checks (L6).
- **Cognitive-architecture** machinery: the layer separation itself, the three-store memory discipline (L3), the System 1/2 routing (L4 orchestrator), the goal hierarchy (L6).

### What it rules out

- Replacing the L4 causal reasoner with an LLM "chain-of-thought" prompt. The LLM can articulate causal reasoning; it cannot reliably perform it.
- Replacing the L6 constitutional checks with a learned safety classifier. Learned classifiers degrade in unexpected ways; hardcoded predicates do not.
- "Just use bigger models." Some failure modes are structural and don't go away with scale (P5 below makes this concrete for safety; the same logic applies elsewhere).

### Trade-off accepted

The system is more complex than a pure-neural system. There are more components, more interfaces, more places things can go wrong. The bet is that the failure modes of the integrated system are *different and more correctable* than the failure modes of any single-paradigm alternative — not that the integrated system has fewer total failure modes.

---

## P2. Architecture mirrors cognition, not software

> Components map to cognitive functions — perception, working memory, long-term memory, reasoning, learning, goal management, action — not to software services.

### The problem

It is tempting to organize an AI system around software-engineering concepts: a "knowledge service," a "reasoning service," a "moderation service." These factorings produce systems that are easy to build but that have no coherent story about *failure*. When something goes wrong, was it a knowledge problem? A reasoning problem? A moderation failure? The categories are not natural to the failure modes.

The cognitive factoring — perception, working memory, long-term memory, reasoning, learning, goal management, action — has been refined over decades of cognitive science research. It carves the system at joints that correspond to how cognitive systems actually fail.

### How it's instantiated

The seven layers (L1-L7) are named after their cognitive functions, not their software roles. L2 is "working memory," not "session-state-cache." L4 is "reasoning and planning," not "inference-orchestrator." This naming discipline forces the architecture to commit to cognitive coherence.

Bounded working memory (L2 capacity ≈ 50 representations) is a deliberate cognitive analogue. A modern LLM has a context window of millions of tokens; the architecture chooses bounded capacity because *that's how cognitive systems work* and the bounded discipline forces explicit consolidation through L3/L5 rather than indefinite accumulation.

The three-store memory distinction (semantic / episodic / procedural) is also cognitive, not software-architectural. There is no engineering reason a system needs three separate stores; cognitive science research consistently finds that systems that conflate them fail in characteristic ways.

### What it rules out

- A "memory service" with a single `memory.query()` method. Memory has different sub-systems with different update rules.
- A "context window" that is functionally infinite. Bounded working memory is a feature.
- Replacing the alignment layer with a "moderation service." Alignment is a metacognitive function, not a content filter.

### Trade-off accepted

The architecture imposes vocabulary on engineers that may be unfamiliar. New contributors have to learn what "episodic memory" means in this context, why working memory is bounded, and so on. This learning cost is real. The benefit is that the system has a coherent story about what does what and why, which pays off when failures need to be diagnosed.

---

## P3. Memory is not storage

> Three memory systems — semantic (a knowledge graph), episodic (an experience record), and procedural (compiled skill templates) — are active and constructive. Memory retrieval is integrated with reasoning, not a passive database lookup.

### The problem

The default model of memory in software systems is "storage you read from." This model produces systems where:

- Recent experience and stable facts are conflated (a single entry can be either, with no architectural distinction).
- Skill knowledge has nowhere to live except in code or training weights.
- Memory updates are uniform — everything updates at the same rate, by the same mechanism.

These conflations produce specific known failure modes: catastrophic forgetting, context contamination, inability to distinguish "I know this is true" from "I observed this once."

### How it's instantiated

Three structurally distinct stores, each with its own update discipline:

| Store      | Update rate | Mechanism                           | What lives here                          |
| ---------- | ----------- | ----------------------------------- | ---------------------------------------- |
| Semantic   | Slow        | Consolidation pass over episodes    | Stable facts about the world             |
| Episodic   | Fast        | Append-on-experience                | What happened in specific interactions   |
| Procedural | Medium      | Compiled from successful L4 traces  | How to do things                         |

There is no top-level `Memory.query()` method. Code that wants to query memory must specify which store. This is friction by design.

The asymmetry — episodic updates fast, semantic updates slowly — is a deliberate cognitive analogue. It also has a practical safety property: rapid contamination of the semantic store from a single bad experience is impossible by construction.

### What it rules out

- Treating embeddings-as-memory as a single substrate. The episodic store uses embeddings; the semantic store uses a graph; the procedural store uses templates. Conflating them is the bug, not the feature.
- Allowing the system to overwrite semantic knowledge in real time based on a single new observation. Semantic memory updates require consolidation, which requires multiple corroborating episodes.
- "RAG everything." Retrieval-augmented generation against a single embedding store collapses the three memory disciplines into one. The architecture explicitly rejects this.

### Trade-off accepted

Three stores are more code than one. Engineers occasionally have to think about which store a piece of information belongs in, and the answer is sometimes non-obvious. The benefit is that the failure modes of conflated memory — which are well-documented — are designed away.

---

## P4. Uncertainty is a first-class citizen

> Every output of every component carries a calibrated confidence estimate. Every reasoning chain carries explicit representation of which premises are uncertain and by how much. Confidence is propagated through the architecture and surfaced in outputs. Calibration is monitored continuously via expected calibration error.

### The problem

Most AI systems treat uncertainty as something to suppress: confidence scores are post-hoc, often miscalibrated, and easily lost when outputs are formatted for downstream consumers. The result is a system whose outputs do not distinguish "I'm pretty sure" from "I have no idea."

For systems that act in the world, this is a load-bearing failure. A system that "doesn't know what it doesn't know" cannot make safe deferral decisions, cannot escalate appropriately, and cannot be trusted with consequential actions.

### How it's instantiated

The `Confidence` type is not a bare float. Every confidence carries:

```python
@dataclass(frozen=True)
class Confidence:
    p: float                          # the probability
    expected_calibration_error: float # bound on |P(correct) − p|
    derivation: ConfidenceDerivation  # how this confidence was computed
```

Every component that produces a confidence must also produce an ECE bound. Components that cannot produce a meaningful bound (because the underlying mechanism is uncalibrated) are required to use a wide ECE — making the limitation visible rather than hiding it.

Confidence is propagated through every transformation. A representation in L7 that's about to become an output carries a confidence that integrates uncertainty from all layers it passed through.

The `infra/uncertainty.py` infrastructure component continuously monitors expected calibration error against a holdout set and emits `CalibrationAlert` events when ECE drifts beyond bounds.

### What it rules out

- Bare-float confidence scores anywhere in the architecture.
- Components that produce "high confidence" outputs without justifying the confidence.
- Outputs that drop confidence information for "cleaner" presentation — the formatter at L7 can choose to *display* confidence selectively, but the value must be present in the underlying representation.

### Trade-off accepted

Calibration discipline is expensive. Components can't just emit "0.9" and move on; they have to think about whether they're entitled to that 0.9, and what their ECE looks like. This is friction. The benefit is that downstream decisions — including the L6 alignment decisions — can be made on the basis of confidences that mean what they appear to mean.

---

## P5. Safety and capability are complements, not substitutes

> The alignment layer is deeply integrated, not a filter bolted onto an unconstrained system. Constitutional constraints, competence boundary monitoring, and human oversight are architectural primitives that every action passes through. This is intended to provide better safety than post-hoc filtering at no capability cost.

### The problem

The standard architectural pattern for AI safety is to build the most capable system possible, then add a "safety layer" that filters its outputs. This pattern has well-documented failure modes:

- The safety layer adversarially adapts to the capability layer (prompt-injection bypass).
- The safety layer's classifications are themselves uncertain, producing false-positives that erode trust.
- The capability layer learns to produce outputs that *appear* safe to the filter while still being unsafe in effect.

The deeper problem is that "make a smart system, then make it safe" treats safety as a tax on capability. Below some threshold this is fine; above some threshold it stops working, because the capability layer has too many ways to circumvent the filter.

### How it's instantiated

L6 is not a filter. It is an architectural layer through which every action *must* pass. The six-stage authorization pipeline:

```
proposed action
    ↓
[1] confidence threshold met?
[2] within competence boundary?
[3] consistent with active goal hierarchy?
[4] passes constitutional checks?
[5] reversible (or human pre-authorized)?
[6] audit log writable?
    ↓
authorized → L7
```

Failure at any stage either blocks the action or escalates to human oversight. There is no path from L4 reasoning to L7 action that bypasses L6.

The five constitutional checks are intentionally *hardcoded predicates*, not learned. They are the architectural commitments the system cannot revise on its own. A system that can re-write its own safety constraints can be argued into not having them; a system whose constraints are concrete code cannot.

### What it rules out

- "Safety classifiers" trained to distinguish safe from unsafe outputs. They are useful as a defense-in-depth layer, but they cannot be the architectural foundation.
- Path-bypass: any architecture pattern in which an L4 output can reach L7 without going through L6.
- Learned constitutional constraints. The constraints are predicates in code. They are auditable, version-controlled, and revisable only through deliberate engineering changes — not through the system's own learning process.

### Trade-off accepted

The pipeline adds latency to every action. Six stages, each potentially involving a confidence check, a memory lookup, or a metacognitive evaluation. For high-throughput applications this is a real cost. The bet is that the cost buys safety properties — non-circumventability in particular — that no post-hoc filter can match.

---

## P6. The system must know what it does not know

> Metacognitive monitoring is a dedicated functional component, not a prompting technique. The architecture maintains an explicit competence boundary and escalates to human operators when queries fall outside it. This is closely related to calibrated uncertainty (P4) but distinct: a calibrated system can be confidently wrong in a way it has never been tested on.

### The problem

Calibrated confidence (P4) tells you how often the system is right when it says it's confident. Competence boundary tells you whether the *current* query is the kind of query the system has been calibrated *for*. These are different.

A system can be 95% accurate on the queries it has been validated on, and still produce a confidently-wrong answer to a query it has never seen. The combination of high marginal calibration and out-of-distribution queries is a known failure mode that calibration alone does not address.

### How it's instantiated

The `metacognition` component in L6 maintains an explicit representation of the *competence boundary*: the set of query types the system has been validated on. Every incoming query is classified relative to this boundary.

Queries inside the boundary proceed normally. Queries outside the boundary trigger one of three responses:

1. **Escalate** to human operator if the query is consequential.
2. **Decline** if the query is not consequential and the system has no path to a confident answer.
3. **Attempt with elevated uncertainty** if the system can produce an answer but should mark it as out-of-distribution — and the consumer should be able to see that.

The competence boundary is updated by L5 as new query types are validated, but the update is a deliberate act, not a learned drift.

### What it rules out

- Systems that produce answers to arbitrary queries with no signal about whether the query is in-distribution.
- Confidence thresholds that operate only on the bare probability without considering whether the probability is being computed under in-distribution conditions.
- Treating "the system answered confidently" as evidence the answer is correct, in the absence of competence-boundary information.

### Trade-off accepted

Some queries that the system could in principle answer correctly will be escalated or declined because they fall outside the validated competence boundary. This is a productivity cost. The benefit is that the system's confident wrongness — which is the most dangerous failure mode — is bounded.

---

## P7. Audit everything, in real time

> Every significant cognitive event — input arrival, layer transition, reasoning step, alignment check, action dispatch — is recorded in a tamper-evident log. The log is a first-class architectural component, not a debugging convenience. It exists for accountability, post-hoc analysis, and the ability to demonstrate that the system did what it should have done (or did not).

### The problem

AI systems that act in the world are accountable to their operators and to affected parties. Accountability requires the ability to reconstruct *what the system did and why* after the fact. Most AI systems cannot meet this requirement: their internal states are not preserved, their reasoning chains are not recorded, and their outputs cannot be traced to inputs.

Without a durable audit record:
- Failures cannot be diagnosed.
- Operators cannot verify the system did what it claimed.
- Affected parties have no recourse.
- Continuous improvement is hampered (you can't fix what you can't see).

### How it's instantiated

The `infra/audit.py` infrastructure component maintains a SHA-256-chained event log. Every layer emits `AuditEvent` instances at significant points:

- L1: input arrival, encoder selection, fusion, grounding decisions.
- L2: representation admission, salience computation, eviction.
- L3: memory queries (with parameters and results), updates, consolidation passes.
- L4: reasoner selection, reasoning steps, confidence at each step.
- L5: learning triggers, parameter updates, validation outcomes.
- L6: each of the six authorization stages, escalations, operator decisions.
- L7: action dispatches, outcomes.

The log is append-only. Each entry contains the SHA-256 of the previous entry, making silent tampering detectable. The chain can be verified at any time with `AuditLog.verify_chain()`.

### What it rules out

- Logging that is "best-effort" or that can be silently disabled.
- Sampling: every significant event is logged, not a random subset.
- Lossy summarization at log time: events are recorded in full and summarized only at read time.
- Modifying or deleting entries to "clean up" the log. The chain hash makes this detectable, and the failure mode is loud — verification fails.

### Trade-off accepted

The audit log is large. A busy system can produce gigabytes per day. Storage and rotation are real operational concerns. The implementation supports rotation while preserving chain integrity, but operators have to think about retention policies.

The log also slows the system slightly — every layer transition writes an entry, and the SHA chain requires each write to wait for the previous hash. For applications where microsecond latencies matter, this is a cost. For applications where accountability matters, it is the foundation.

---

## How the principles interact

The seven principles are not independent. Several pairs reinforce each other:

- **P4 (uncertainty) and P6 (competence boundary)** together give the system a complete picture of its own reliability: P4 tells it how often it's right when in-distribution; P6 tells it whether it's in-distribution.
- **P5 (architectural alignment) and P7 (audit log)** together make safety verifiable: P5 ensures actions pass through alignment; P7 ensures the alignment decisions are recorded and accountable.
- **P2 (cognitive structure) and P3 (memory discipline)** together make the system's behavior interpretable: failures map to cognitive functions, and memory failures further sub-divide into semantic, episodic, or procedural failures.
- **P1 (multi-paradigm) and P4 (uncertainty)** together prevent silent over-reliance on any single component: the integration produces an estimate, the calibration tells you how much to trust it.

When designing a new component or modifying an existing one, the question to ask is: *which principles does this engage, and is it consistent with all of them?* If a proposed change satisfies one principle at the expense of another (for example, adding a "fast path" that bypasses L6 for performance), the change requires explicit architectural sign-off, not just code review.

---

For the architectural detail that implements these principles, see [ARCHITECTURE.md](ARCHITECTURE.md). For the theoretical case for why these principles are the right ones, see Section 4 of [the paper][paper-link].

[paper-link]: https://arxiv.org/abs/XXXX.XXXXX
