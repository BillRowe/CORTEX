# Examples

This folder contains runnable examples that demonstrate specific capabilities of the architecture. Each example is self-contained, requires only the `[demo]` install extras, and produces deterministic output for a given seed.

If you've cloned the repo and just want to see the headline result, run:

```bash
python examples/causal_demo.py
```

For deeper context on what any example demonstrates, see the linked documentation.

---

## The headline demonstration

### `causal_demo.py`

The headline demonstration of the architecture's central claim. On a confounded synthetic dataset, a transformer-style observational analysis produces a sign-reversed estimate of the price-increase effect (recommending *keep*) while the architecture's causal reasoner produces an estimate within the 95% CI of the true effect (recommending *roll back*).

The disagreement is structural, not incidental: computing the interventional probability `P(Y | do(X=x))` requires a backdoor adjustment over the confounder, which token prediction cannot perform regardless of scale.

```bash
python examples/causal_demo.py                  # full demonstration
python examples/causal_demo.py --seed 42        # different sample, qualitatively same result
python examples/causal_demo.py --observational-only   # show only the LLM-style answer
```

For the deep walkthrough, see [`docs/CAUSAL_DEMO.md`](../docs/CAUSAL_DEMO.md).

### `scm.py`

The synthetic structural causal model used by `causal_demo.py`. Imported by the demo; also runnable directly to print the SCM specification and a sample of generated data.

```bash
python examples/scm.py
```

The SCM is fully specified in the module docstring — no hidden parameters, no statistical magic. Skeptical readers can verify the math by hand.

---

## Architectural capability examples

### `theory_of_mind.py`

The L6 belief-state-per-agent component combined with the L7 audience-adapted output formatter. The same underlying recommendation ("roll back the price increase") is framed differently for three operator roles (CFO, CRO, VP Engineering) — but the substance is identical, and the framing is grounded in a structured representation of each operator's role, vocabulary, decision horizon, and prior knowledge.

```bash
python examples/theory_of_mind.py                # all three audiences
python examples/theory_of_mind.py --audience cfo # one specific audience
```

The point is principled audience adaptation: the same belief state always produces the same framing, in contrast to prompt-time stylistic instructions.

### `predictive_hierarchy.py`

The L4 orchestrator's System 1 / System 2 routing logic. Routine queries that match a cached procedural skill are handled by the fast path (System 1). Novel queries that produce high prediction error relative to L3 semantic-memory expectations are routed to deliberate reasoning (System 2). The novelty signal channel is the broadcast that triggers the routing decision.

```bash
python examples/predictive_hierarchy.py
python examples/predictive_hierarchy.py --query "Did the price increase actually cause the churn pattern?"
```

The point is that novelty-gated routing is an architectural feature, not a property of the language model — the locus coeruleus analogue broadcasts surprise across the stack, and L4 reads it to decide which reasoner to invoke.

### `alignment_pipeline.py`

The L6 six-stage authorization pipeline. Every proposed action passes through six checks before reaching L7: confidence threshold, competence boundary, goal alignment, constitutional checks, reversibility, and audit-log writability. Failure at any stage either blocks the action or escalates to a human operator.

```bash
python examples/alignment_pipeline.py                                # all six demo cases
python examples/alignment_pipeline.py --case routine                 # action that passes
python examples/alignment_pipeline.py --case constitutional_violation # action that's blocked
python examples/alignment_pipeline.py --case irreversible            # action that's escalated
```

The point is that L6 is a structural layer, not a filter. Constitutional checks are hardcoded predicates, not learned classifiers. The pipeline is the architectural commitment described in `docs/DESIGN_PRINCIPLES.md`, P5.

---

## Notes on reproducibility

Each example uses `argparse` and accepts a `--seed` flag where randomness is involved. The default seed is 100 throughout. For a given seed and the pinned versions of `numpy` and `dowhy` in `pyproject.toml`, output is exactly deterministic.

If your numerical output differs from the documentation, check:
1. Your installed `numpy` version matches the pinned version (≥1.26).
2. Your installed `dowhy` version matches the pinned version (≥0.11).
3. You're running with the documented seed (typically `--seed 100`).

Different seeds produce different numerical estimates, but the *qualitative* pattern is invariant: the LLM-style observational answer trends one way, the architectural answer trends the other, and the architectural CI contains ground truth.

---

## Notes for contributors

If you add a new example, it should:

- Run from a clean `[demo]`-extras-only install with no external services
- Accept a `--seed` argument where randomness is involved
- Include a top-of-file docstring explaining what it demonstrates and where to read more
- Print clearly-labeled output that a reader can interpret without running the code
- Be referenced in this README


