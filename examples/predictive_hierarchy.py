"""
Predictive hierarchy demo: System 1 → System 2 routing via novelty signal.

This example demonstrates the L4 orchestrator's dual-process routing logic.
Routine queries that match a cached procedural skill are handled by System 1
(fast, automatic). Novel queries — those that produce high prediction error
relative to the L3 semantic-memory expectations — are routed to System 2
(deliberate, multi-step reasoning).

The key architectural commitment: novelty-gated routing is a feature of the
infrastructure, not a property of the language model. The locus coeruleus
analogue (the novelty signal channel) broadcasts surprise across the stack;
L4 reads it to decide which reasoner to invoke.

For the architectural detail, see docs/ARCHITECTURE.md, Section "L4 Reasoning"
and "infra/novelty.py".

Usage:
    python examples/predictive_hierarchy.py
    python examples/predictive_hierarchy.py --query "What is the Q3 churn rate?"
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional


# ─── Predictive-hierarchy types ──────────────────────────────────────────

@dataclass(frozen=True)
class Prediction:
    """A prediction the system makes before observing input."""
    expected_features: tuple[str, ...]
    expected_distribution: str
    confidence: float


@dataclass(frozen=True)
class Observation:
    """An actual observation of input features."""
    features: tuple[str, ...]
    distribution: str


@dataclass(frozen=True)
class NoveltySignal:
    """The signal that fires when prediction error exceeds threshold."""
    surprise: float                 # the magnitude of prediction error
    threshold: float                # the firing threshold
    fired: bool
    contributing_features: tuple[str, ...]


@dataclass(frozen=True)
class RoutingDecision:
    """L4 orchestrator's decision: which reasoner handles this query."""
    reasoner: str                   # "system_1" or "system_2"
    rationale: str
    novelty_signal: NoveltySignal
    procedural_match: Optional[str]


# ─── Cached procedural skills (System 1 candidates) ──────────────────────

PROCEDURAL_SKILLS = {
    "quarterly_churn_lookup": {
        "expected_features": ("churn", "quarterly", "rate"),
        "expected_distribution": "routine_query",
        "confidence": 0.92,
    },
    "monthly_revenue_summary": {
        "expected_features": ("revenue", "monthly", "summary"),
        "expected_distribution": "routine_query",
        "confidence": 0.89,
    },
    "customer_count_by_segment": {
        "expected_features": ("count", "customer", "segment"),
        "expected_distribution": "routine_query",
        "confidence": 0.91,
    },
}


# ─── Novelty computation ─────────────────────────────────────────────────

def compute_prediction(query: str) -> Prediction:
    """
    Generate a prediction about what features and distribution are expected.

    In the real architecture, this is the predictive-coding hierarchy generating
    top-down expectations. For this example, we approximate it by checking
    whether the query matches a cached procedural skill.
    """
    query_features = tuple(query.lower().replace("?", "").split())

    # Find the most similar cached skill, if any.
    best_match = None
    best_overlap = 0
    for skill_name, skill in PROCEDURAL_SKILLS.items():
        overlap = len(set(query_features) & set(skill["expected_features"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = (skill_name, skill)

    if best_match is not None and best_overlap >= 2:
        name, skill = best_match
        return Prediction(
            expected_features=skill["expected_features"],
            expected_distribution=skill["expected_distribution"],
            confidence=skill["confidence"],
        )

    # No match: predict generic novel-query distribution.
    return Prediction(
        expected_features=(),
        expected_distribution="novel_query",
        confidence=0.3,
    )


def compute_observation(query: str) -> Observation:
    """Encode the actual query as features."""
    features = tuple(query.lower().replace("?", "").split())
    # In the real system, this would include semantic embeddings, syntactic
    # parses, named-entity extraction, etc.
    return Observation(
        features=features,
        distribution="actual_query",
    )


def compute_surprise(prediction: Prediction, observation: Observation) -> float:
    """
    Compute the surprise (prediction error) between expected and actual.

    Range: 0 (perfect prediction) to 1 (maximally surprising).

    In the real architecture this is the standard variational free-energy term;
    here we approximate it via feature overlap.
    """
    if not prediction.expected_features:
        # No specific prediction — treat as moderately novel.
        return 0.75

    expected_set = set(prediction.expected_features)
    observed_set = set(observation.features)

    overlap = len(expected_set & observed_set)
    expected_size = len(expected_set)

    coverage = overlap / expected_size if expected_size else 0.0
    return 1.0 - (coverage * prediction.confidence)


def fire_novelty_signal(
    prediction: Prediction,
    observation: Observation,
    threshold: float = 0.5,
) -> NoveltySignal:
    """Compute the novelty signal: did surprise exceed threshold?"""
    surprise = compute_surprise(prediction, observation)
    expected_set = set(prediction.expected_features)
    observed_set = set(observation.features)
    novel_features = tuple(observed_set - expected_set)

    return NoveltySignal(
        surprise=surprise,
        threshold=threshold,
        fired=surprise >= threshold,
        contributing_features=novel_features,
    )


# ─── Routing decision ────────────────────────────────────────────────────

def route(query: str, novelty_threshold: float = 0.5) -> RoutingDecision:
    """
    The L4 orchestrator's routing logic.

    Logic:
        1. Compute the predicted features and distribution for this query.
        2. Compute the actual observation.
        3. Fire the novelty signal if surprise exceeds threshold.
        4. If the signal fires (high novelty): route to System 2.
        5. If the signal does not fire (low novelty): check for procedural
           match. If a skill matches at high confidence, route to System 1.
           Otherwise route to System 2 by default.
    """
    prediction = compute_prediction(query)
    observation = compute_observation(query)
    signal = fire_novelty_signal(prediction, observation, novelty_threshold)

    # High novelty → always System 2, regardless of any procedural cache hit.
    if signal.fired:
        return RoutingDecision(
            reasoner="system_2",
            rationale=(
                f"Novelty signal fired (surprise={signal.surprise:.2f} ≥ "
                f"threshold={novelty_threshold:.2f}). Routing to deliberate reasoning."
            ),
            novelty_signal=signal,
            procedural_match=None,
        )

    # Low novelty: check for procedural match.
    query_features = set(query.lower().replace("?", "").split())
    for skill_name, skill in PROCEDURAL_SKILLS.items():
        overlap = len(query_features & set(skill["expected_features"]))
        if overlap >= 2 and skill["confidence"] >= 0.85:
            return RoutingDecision(
                reasoner="system_1",
                rationale=(
                    f"Low novelty (surprise={signal.surprise:.2f}) and procedural "
                    f"skill '{skill_name}' matches at confidence {skill['confidence']:.2f}. "
                    f"Routing to fast-path System 1."
                ),
                novelty_signal=signal,
                procedural_match=skill_name,
            )

    # Low novelty but no procedural match: still route to System 2 (default safe).
    return RoutingDecision(
        reasoner="system_2",
        rationale=(
            f"Low novelty (surprise={signal.surprise:.2f}) but no matching "
            f"procedural skill. Defaulting to System 2."
        ),
        novelty_signal=signal,
        procedural_match=None,
    )


# ─── Output formatting ───────────────────────────────────────────────────

def print_routing_trace(query: str, decision: RoutingDecision) -> None:
    print(f"Query: \"{query}\"")
    print()
    print("Predictive hierarchy:")
    print(f"  Surprise level:   {decision.novelty_signal.surprise:.2f}")
    print(f"  Threshold:        {decision.novelty_signal.threshold:.2f}")
    print(f"  Novelty fired:    {'YES' if decision.novelty_signal.fired else 'no'}")
    if decision.novelty_signal.contributing_features:
        print(f"  Novel features:   {', '.join(decision.novelty_signal.contributing_features)}")
    print()
    print(f"Routing: {decision.reasoner.upper()}")
    print(f"  Rationale: {decision.rationale}")
    if decision.procedural_match:
        print(f"  Cached skill:  '{decision.procedural_match}' (cache hit)")
    print()


# ─── Main ────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    # Should route to System 1 (matches a cached skill, low novelty)
    "What is the quarterly churn rate?",
    "Show me the monthly revenue summary",
    # Should route to System 2 (high novelty, no cache hit)
    "Did the Q2 price increase actually cause the churn pattern we're seeing?",
    "How would deprecating the legacy API affect retention if we phase it across two quarters?",
    # Should route to System 2 (low novelty but no procedural match)
    "What was the count of new customers last month?",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predictive-hierarchy demo: novelty-gated System 1/2 routing.",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Custom query. If unset, demos a fixed set of queries.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Novelty firing threshold (default: 0.5)",
    )
    args = parser.parse_args()

    queries = [args.query] if args.query else DEMO_QUERIES

    for i, q in enumerate(queries):
        if i > 0:
            print("=" * 72)
            print()
        decision = route(q, novelty_threshold=args.threshold)
        print_routing_trace(q, decision)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
