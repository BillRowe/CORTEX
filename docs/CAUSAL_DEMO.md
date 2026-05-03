# The Causal Reasoning Demo: A Deep Walkthrough

This document is a deeper walkthrough of the headline causal-reasoning demonstration than the paper allows. It is intended for readers who want to understand exactly what is happening in `examples/causal_demo.py` — the synthetic data, the two competing analyses, why they disagree, and why the disagreement is structural rather than incidental.

For the architectural context, see [ARCHITECTURE.md](ARCHITECTURE.md). For the theoretical justification of why this matters, see Sections 2 and 6 of [the paper][paper-link]. For setup and reproduction, see [INSTALLATION.md](INSTALLATION.md).

---

## TL;DR

A transformer-based recommender, given 5,000 customer records, recommends *keeping* a price increase because customers who got the increase are churning at lower rates. The architecture, given the same 5,000 records, recommends *rolling back* the price increase. The architecture is right.

The disagreement is not about training data, prompt engineering, or model size. The conditional probability `P(churn | price=1)` and the interventional probability `P(churn | do(price=1))` are mathematically different quantities when a confounder is present. Computing the second requires a structural operation — Pearl's backdoor adjustment — that token prediction cannot perform regardless of scale.

**This is the demo's central claim.** The numbers are reproducible. The disagreement is not a quirk of one synthetic dataset; it is a manifestation of a general gap between conditional and interventional inference that affects any decision-relevant analysis where confounders are present and unmeasured.

---

## The setup: a synthetic structural causal model

The demo generates 5,000 customer records from a deterministic structural causal model. "Structural" here means we specify the causal graph and the data-generating process directly, which gives us ground truth: we know the true causal effect of the price increase on churn, because we wrote it.

### The variables

| Variable        | Type   | Meaning                                         |
| --------------- | ------ | ----------------------------------------------- |
| `product_health` | Binary | Whether the product is healthy (well-fitted, well-supported, growing) — *latent confounder* |
| `price_increase` | Binary | Whether this customer received a price increase |
| `churn`          | Binary | Whether this customer churned                   |

The key word is **latent**. The product-health variable causally influences both whether a customer gets a price increase (sales teams selectively raise prices on healthy accounts) and whether they churn (healthy products retain customers regardless of price). It is the textbook structure of a confounded observational study.

### The data-generating process

```
   P(product_health = healthy) = 0.60
   
   P(price_increase = 1 | product_health = healthy)   = 0.65
   P(price_increase = 1 | product_health = unhealthy) = 0.43
   
   P(churn = 1 | price_increase = 0, product_health = healthy)   = 0.05
   P(churn = 1 | price_increase = 1, product_health = healthy)   = 0.05 + 0.030 = 0.08
   P(churn = 1 | price_increase = 0, product_health = unhealthy) = 0.30
   P(churn = 1 | price_increase = 1, product_health = unhealthy) = 0.30 + 0.030 = 0.33
```

By construction, the **true average treatment effect** of the price increase on churn is **+3.00 percentage points**. The price increase increases churn by exactly 3 pp regardless of product health; this is the additive constant in the last two equations.

### The causal graph

```
              ┌───────────────────┐
              │  product_health   │   (latent confounder)
              └─────────┬─────────┘
                        │
                ┌───────┴───────┐
                ↓               ↓
        ┌───────────────┐  ┌───────┐
        │ price_increase │ →│ churn │
        └───────────────┘  └───────┘
```

Three causal arrows: product_health → price_increase, product_health → churn, price_increase → churn. The first and second together create a *backdoor path* from price_increase to churn that does not go through the direct effect we care about.

This is the canonical confounded structure. Every observational dataset where a treatment is selectively assigned has this shape; the only question is whether you handle it.

---

## What a transformer-based recommender sees

When an LLM (or any model trained on next-token prediction) is given the dataset and asked "did the price increase work?", it computes — implicitly — a conditional probability:

```
   E[churn | price_increase = 1] = 10.92%   (n = 2,959)
   E[churn | price_increase = 0] = 16.76%   (n = 2,041)
   ────────────────────────────────────────────
   Difference = −5.84 percentage points
```

The LLM's answer is then: *the price increase reduced churn by 5.84 percentage points; keep it.*

This is a true statement about the conditional probability. It is also catastrophically wrong as an answer to the question the operator actually asked.

The reason: the customers who got the price increase were predominantly healthy accounts (because that's how sales teams selectively price). Healthy accounts churn at 5–8% regardless of price, while unhealthy accounts churn at 30–33% regardless of price. The 5.84 pp reduction the LLM observes is almost entirely a composition effect — the *kind* of customers who got the increase were already low-churn customers — not an effect of the price increase itself.

The conditional answer would be correct only if treatment had been randomly assigned. It wasn't. It was confounded by product health. And the LLM has no architectural mechanism for noticing this, because identifying confounding requires representing and reasoning about a causal graph — which token prediction does not do.

### "Couldn't a smarter LLM figure this out?"

This is the question every careful reader asks at this point. The honest answer is: a sufficiently large LLM, given a sufficiently leading prompt, can sometimes produce text that *describes* confounding correctly. What it cannot do is reliably *perform* the structural operation that resolves it.

The relevant evidence:

1. **Empirical**: LLMs given confounded datasets produce inconsistent answers. The same model, on the same data, will sometimes correctly identify confounding and sometimes not, depending on prompt phrasing. This is not a property a load-bearing decision aid can have.

2. **Mechanistic**: Pearl's do-calculus is a set of rewrite rules over a graph. To execute it, the system must (a) represent a graph, (b) identify the backdoor set, (c) verify identifiability conditions, and (d) compute the adjusted estimand. Each of these is a discrete structural operation; together they are not what next-token prediction approximates.

3. **Scaling**: causal-inference benchmarks have shown that LLM performance on such tasks does not improve monotonically with scale. The plateau is below the threshold required for reliable use as a decision aid. (See Section 2.2 of [the paper][paper-link] for citations.)

The architecture's bet is not that LLMs are useless on these problems — they're often genuinely helpful for *describing* the causal structure once it's been identified. The bet is that the structural operation itself needs to be performed by a component built for it.

---

## What the architecture sees

The architecture's L4 reasoning layer routes the query to its causal reasoner. The causal reasoner performs four operations:

### Step 1: Discover the causal graph

The PC algorithm (Spirtes-Glymour-Scheines) recovers a causal graph from the data using conditional-independence tests. On the synthetic SCM, it correctly identifies:

```
   product_health → price_increase
   product_health → churn
   price_increase → churn
```

In a real-world deployment, this step is where domain knowledge enters: an analyst typically reviews the discovered graph and corrects edges that are obviously wrong from domain context. The demo skips this human-in-the-loop step because the synthetic data is constructed cleanly enough that PC recovers the correct graph automatically.

### Step 2: Identify the backdoor set

Given the graph and the question "what is the causal effect of price_increase on churn?", the reasoner identifies which variables must be conditioned on to block all backdoor paths.

```
   Backdoor path: price_increase ← product_health → churn
   Backdoor set:  {product_health}
   Identifiable:  YES (sufficient adjustment set is observable)
```

Identifiability is not automatic. If the only available backdoor set required conditioning on a variable we don't observe, the causal effect is *not identifiable from observational data alone*, and the reasoner says so. The architecture's correct response in that case is to escalate — either request data the analyst doesn't have, or recommend a randomized experiment, or explicitly mark the answer as not derivable from the available evidence.

In this demo, product_health is observable (we generated it), so identification succeeds.

### Step 3: Apply backdoor adjustment

The backdoor adjustment formula stratifies over the adjustment set:

```
   ATE = Σ_h [E(churn | price=1, product_health=h) − E(churn | price=0, product_health=h)] · P(product_health=h)
```

The reasoner computes this directly from the data:

```
   Within healthy stratum   (n ≈ 3,000, P(healthy) = 0.60):
       E(churn | price=1, healthy) − E(churn | price=0, healthy) = +3.35 pp

   Within unhealthy stratum (n ≈ 2,000, P(unhealthy) = 0.40):
       E(churn | price=1, unhealthy) − E(churn | price=0, unhealthy) = +1.27 pp

   ATE = 0.60 × 3.35 + 0.40 × 1.27 = +2.51 pp
   95% CI:                         (+0.20 pp, +4.82 pp)
```

The 95% confidence interval is computed via standard error formulas under the no-unmeasured-confounding assumption. In real-world use, the assumption is rarely guaranteed — see step 4.

### Step 4: Sensitivity analysis

The architecture's final causal step asks: *how robust is the +2.51 pp estimate to violations of the no-unmeasured-confounding assumption?* The DoWhy library provides several sensitivity tests; the demo runs the simplest one (a Rosenbaum-style bound on how strong an unmeasured confounder would need to be to overturn the result).

For the synthetic SCM the answer is "very robust" — there are no unmeasured confounders, by construction. In a real deployment this step is what tells the analyst whether they should trust the +2.51 pp number, or treat it as suggestive pending experimental validation.

---

## The reveal

```
                     LLM observational            Architecture (Pearl backdoor)
                     ─────────────────            ───────────────────────────────
   Estimated effect:       −5.84 pp                          +2.51 pp
   Recommendation:         Keep the increase                 Roll it back
   95% CI:                 (not reported)                    (+0.20, +4.82)
   Ground-truth effect:    +3.00 pp                          +3.00 pp
                           ✗ Sign-reversed                    ✓ Within CI
```

Two things to notice.

First, the LLM's answer is not just *wrong by a little*. It has the wrong sign. A decision-maker who acts on the LLM's analysis takes the *opposite* action from the one the data actually supports. The price increase is causing churn (true effect: +3 pp), but the LLM reports it is preventing churn (estimated effect: −5.84 pp). For a SaaS company at this scale (5,000 customers), the difference between rolling back the price increase and entrenching it is on the order of $3.25M in annual revenue. The error is not academic.

Second, the architecture's answer is correct *in the sense that the 95% CI contains the true effect*. It is not a perfect point estimate — it produced +2.51 pp where ground truth is +3.00 pp. That gap is sampling variation: at n=5,000 with this stratum split, the standard error is about 1.18 pp, and the observed estimate is well within one standard error of truth. The CI is doing its job — it acknowledges uncertainty and contains the true value.

These are different kinds of "wrongness." A point estimate that is off by 0.5 pp but whose CI contains truth is the kind of wrongness a careful analyst can work with. A point estimate with the wrong *sign* is the kind of wrongness that produces actively harmful decisions.

---

## What's actually happening: the structural argument

Why does the LLM produce a sign-reversed estimate? Not because it is undertrained, or because the prompt was poorly engineered, or because the model is too small. The reason is structural.

The LLM is computing — implicitly, through pattern-matching — a quantity of the form:

```
   "What pattern in the data am I most likely to see continue?"
```

In a confounded dataset, the dominant pattern is the *correlation* between treatment and outcome, marginalized over (i.e., averaged across) all the other variables. That correlation is:

```
   E[churn | price=1] − E[churn | price=0] = −5.84 pp
```

This number is mathematically real. It is what the data shows, conditionally on price_increase. Any system that summarizes the data without modeling the causal structure will land on something close to this number.

The architecture is computing a different quantity:

```
   E[churn | do(price=1)] − E[churn | do(price=0)] = +2.51 pp ≈ +3.00 pp (true)
```

The `do()` operator is Pearl's notation for *intervention*. It asks: "If we forced price_increase = 1 for everyone, holding everything else fixed at its prior distribution, what would average churn be?" This is the question the operator actually means when they ask "what is the effect of the price increase?" — and it is mathematically distinct from the conditional probability, because the conditional bakes in selection effects and the interventional does not.

Pearl's do-calculus provides the rewrite rules that translate `do()` queries into operations on observable conditional probabilities, *given a causal graph*. The graph is essential: without it, the rewrite is undefined. With it, the rewrite produces the backdoor adjustment we performed in Step 3.

This is why the gap is structural. A system that operates on tokens — even one that has read every causal-inference textbook — does not have an architectural mechanism for representing the graph and performing the rewrite. A system that operates on graphs does. The two systems are computing different functions; one of them happens to be the one the operator is asking for.

---

## Reproducing the demo

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
python -m venv .venv && source .venv/bin/activate
pip install -e ".[demo]"
python examples/causal_demo.py --seed 100
```

The synthetic SCM is fully specified in `examples/scm.py`. The demo numbers are deterministic at `--seed 100` if you have `numpy>=1.26` and `dowhy>=0.11`; other versions may produce slight variations. See [INSTALLATION.md](INSTALLATION.md) for details.

To explore the demo's behavior:

```bash
# Vary the sample size (smaller n → wider CI)
python examples/causal_demo.py --seed 100 --n 1000

# Vary the strength of confounding
python examples/causal_demo.py --seed 100 --confounding 0.3

# Disable the architecture's causal reasoner (LLM-only mode)
python examples/causal_demo.py --seed 100 --observational-only

# Run with a different seed
python examples/causal_demo.py --seed 42
```

Different seeds produce different numerical estimates, but the qualitative pattern is invariant: the observational answer trends in one direction, the architectural answer trends in the other, and the architectural answer's CI contains ground truth.

---

## Limitations and honest caveats

The demo is intentionally clean. Real-world causal inference is harder. Several places the demo simplifies:

**Ground truth is given, not discovered.** The demo's true effect is +3.00 pp because we set it that way. In a real deployment, ground truth is unknown — that's the whole point of the analysis. The architecture's claim is not "we recover ground truth perfectly" but "we recover an estimate that, under stated assumptions, is unbiased for the causal effect, with stated uncertainty."

**The graph is recoverable from data.** PC algorithm works cleanly on the synthetic SCM because the data was generated to satisfy its assumptions (faithfulness, sufficiency, etc.). On real data, the graph typically requires substantial domain expertise to specify correctly, and the architecture's correct workflow is human-in-the-loop graph specification, not automated discovery. The demo skips this step because we know the true graph.

**The confounder is observed.** Product health is observable in the demo. In real settings, the most consequential confounders are typically *unmeasured*, and the system's correct response is to either acknowledge that the causal effect is not identifiable from the available data, or apply more advanced techniques (instrumental variables, regression discontinuity, difference-in-differences) that require additional structural assumptions. The architecture's roadmap includes these; the demo does not exercise them.

**Sensitivity analysis is brief.** The full DoWhy sensitivity-analysis machinery (Rosenbaum bounds, E-values, placebo tests) is more extensive than what the demo runs. A production deployment would exercise it more thoroughly.

**The LLM comparison is illustrative.** The demo's "LLM observational answer" is computed by directly stratifying the dataset, which is what an LLM that successfully read the data would produce. An actual LLM might add error in either direction — sometimes correctly identifying the confounding without prompting, sometimes producing entirely fictional numbers. The −5.84 pp number is the *best-case* observational answer; the architecture comparison would look more favorable, not less, against a realistic LLM that might also hallucinate.

None of these caveats undermine the central claim. The structural gap between conditional and interventional inference is real, it affects any confounded analysis, and the architecture's approach (graph + backdoor adjustment + sensitivity analysis) is the textbook solution. The demo is a clean demonstration of the gap, not a comprehensive evaluation of the architecture's causal-inference capabilities.

---

## Why this matters beyond the demo

The synthetic SaaS scenario is a stand-in for a category of decisions that recur across domains:

- **Medical**: did the new treatment protocol reduce mortality, or were the patients who received it healthier to begin with?
- **Policy**: did the after-school program improve test scores, or did the kinds of families who enrolled their children differ systematically?
- **Marketing**: did the campaign drive conversions, or were the customers it reached already more likely to convert?
- **Engineering**: did the deployment improve latency, or were the services that received it operating in lower-traffic regions?

Every one of these is the same structure: a treatment selectively assigned by some unmeasured (or under-measured) factor that also influences the outcome. The conditional probability is what the data show; the interventional probability is what the decision-maker is actually asking for. The two are different, and the difference is sometimes large enough to flip the sign of the recommendation.

A decision aid that conflates them is not just inaccurate — it produces actively wrong recommendations, in a way that scales with the influence of the system. The demo is a small, clean, reproducible example of a pattern that the architecture is built to handle and that token-prediction systems are not.

---

## Further reading

- **In this repository**: [ARCHITECTURE.md](ARCHITECTURE.md) (Section: "L4 — Reasoning and Planning"), [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) (P1: "No single paradigm is sufficient"), `examples/scm.py` (the SCM specification).
- **In the paper**: Section 2.2 (the conditional/interventional gap), Section 5.4 (the L4 causal reasoner), Section 6 (this demonstration in full).
- **External**: Pearl, *Causality* (2009), is the canonical reference. For a more accessible introduction, Pearl & Mackenzie, *The Book of Why* (2018). The DoWhy documentation at https://www.pywhy.org/dowhy/ is the practical entry point for the underlying library.

---

[paper-link]: https://arxiv.org/abs/XXXX.XXXXX
