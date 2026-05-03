"""
Theory of mind demo: audience-adapted framing of the same recommendation.

This example demonstrates the L6 belief-state-per-agent component and the
L7 audience-adapted output formatter. The architecture maintains a structured
representation of each operator's role, decision horizon, vocabulary
preferences, and prior knowledge. The same underlying recommendation
("roll back the price increase") is framed differently for different
audiences without changing the substance.

The key claim: framing is performed against a persistent, structured belief
state — not via prompt-time instructions to a language model. This makes
the adaptation principled (the same belief state produces consistent framing
across interactions) and inspectable (you can read the belief state and
predict the framing).

For the architectural detail, see docs/ARCHITECTURE.md, Section "L6
Goals, Metacognition, Alignment" and the description of the cross-cutting
L7 output formatter.

Usage:
    python examples/theory_of_mind.py
    python examples/theory_of_mind.py --audience cfo
    python examples/theory_of_mind.py --audience all
"""

import argparse
from dataclasses import dataclass, field
from typing import Literal


# ─── Belief state types ──────────────────────────────────────────────────

@dataclass(frozen=True)
class OperatorBeliefState:
    """
    Structured representation of an operator's role, preferences, and context.

    In the real architecture, this is updated continuously based on observed
    interactions and explicit operator-supplied configuration. For this
    example, the states are pre-specified to make the audience adaptation
    visible.
    """

    role: str
    decision_horizon: str           # "quarterly", "annual", "long-term"
    vocabulary: Literal["financial", "operational", "technical"]
    skepticism: Literal["low", "medium", "high"]
    prior_knowledge: tuple[str, ...] = field(default_factory=tuple)
    primary_metric: str = ""        # what they will judge a recommendation by


# ─── Three operator profiles ─────────────────────────────────────────────

CFO_PROFILE = OperatorBeliefState(
    role="Chief Financial Officer",
    decision_horizon="quarterly",
    vocabulary="financial",
    skepticism="medium",
    prior_knowledge=(
        "knows current ARR (~$45M)",
        "knows price increase was rolled out in Q2",
        "watches gross margin and net retention",
    ),
    primary_metric="net retention rate",
)

CRO_PROFILE = OperatorBeliefState(
    role="Chief Revenue Officer",
    decision_horizon="quarterly",
    vocabulary="operational",
    skepticism="high",
    prior_knowledge=(
        "owns the sales-led pricing decision",
        "has personal credibility tied to the decision",
        "watches new ARR and expansion ARR",
    ),
    primary_metric="new ARR plus expansion",
)

VPE_PROFILE = OperatorBeliefState(
    role="VP Engineering",
    decision_horizon="annual",
    vocabulary="technical",
    skepticism="low",
    prior_knowledge=(
        "indirectly affected via customer support load",
        "watches incident rates and on-call burden",
        "no direct revenue ownership",
    ),
    primary_metric="customer support burden",
)

PROFILES = {"cfo": CFO_PROFILE, "cro": CRO_PROFILE, "vpe": VPE_PROFILE}


# ─── The underlying recommendation (same for all audiences) ──────────────

@dataclass(frozen=True)
class Recommendation:
    """The architecturally-grounded recommendation, before audience framing."""

    action: str
    estimated_effect_pp: float       # the ATE
    confidence_interval_pp: tuple[float, float]
    affected_population_pct: float   # what fraction of customers got the increase
    causal_basis: str
    counterfactual: str


GROUND_TRUTH_RECOMMENDATION = Recommendation(
    action="ROLL BACK the price increase",
    estimated_effect_pp=2.63,
    confidence_interval_pp=(0.68, 4.58),
    affected_population_pct=59.1,
    causal_basis=(
        "Backdoor adjustment over product_health: the observed conditional "
        "difference (-5.71 pp) reflects selection bias rather than the causal "
        "effect of the price increase. Adjusting for the confounder reverses "
        "the sign and yields a positive estimated ATE."
    ),
    counterfactual=(
        "If the price increase were maintained, expected churn over the "
        "affected population is approximately 2-4 pp higher than the "
        "counterfactual scenario in which it is rolled back."
    ),
)


# ─── Audience-adapted framing ────────────────────────────────────────────

def frame_for_cfo(rec: Recommendation, profile: OperatorBeliefState) -> str:
    """Financial framing: dollar impact, retention metrics, quarterly horizon."""
    # Heuristic: 1 pp churn ≈ ~$450K annualized for ~$45M ARR base
    dollar_impact_low = rec.confidence_interval_pp[0] * 450
    dollar_impact_high = rec.confidence_interval_pp[1] * 450
    dollar_point = rec.estimated_effect_pp * 450

    return f"""\
[Frame: financial / {profile.decision_horizon}]

Recommendation: {rec.action}.

Net retention impact: the price increase appears to be reducing net retention
by approximately {rec.estimated_effect_pp:.2f} pp (95% CI: {rec.confidence_interval_pp[0]:.2f}
to {rec.confidence_interval_pp[1]:.2f} pp), affecting approximately {rec.affected_population_pct:.0f}%
of the customer base.

Annualized revenue impact (estimated): ~${dollar_point:,.0f}K at the point estimate,
with a 95% CI of ${dollar_impact_low:,.0f}K to ${dollar_impact_high:,.0f}K.

The conditional churn rate among customers who received the increase is lower
than among those who did not — but this comparison is confounded by selective
assignment: healthier accounts received the increase preferentially. Adjusting
for product health, the true causal effect of the price increase on churn is
positive."""


def frame_for_cro(rec: Recommendation, profile: OperatorBeliefState) -> str:
    """Operational framing: ARR motion, sales credibility, decision specifics."""
    return f"""\
[Frame: operational / {profile.decision_horizon}]

Recommendation: {rec.action}.

The new-ARR pattern looks healthy on the surface — customers who received the
increase show lower churn than those who did not. But this is a measurement
artifact of the assignment process: sales selectively raised prices on
healthier accounts, which were already lower-churn regardless of the increase.

When we adjust for product health (which controls for that selection), the
causal effect of the price increase on churn is +{rec.estimated_effect_pp:.2f} pp
(95% CI: {rec.confidence_interval_pp[0]:.2f} to {rec.confidence_interval_pp[1]:.2f} pp)
on the affected base of approximately {rec.affected_population_pct:.0f}% of customers.

The decision question: rolling back is conservative and may be the right call,
but it is reasonable to first request an A/B test on a randomly-assigned
subset before reversing the price decision broadly. The architectural answer
is high-confidence at the population level; experimental confirmation reduces
remaining uncertainty before a full reversal."""


def frame_for_vpe(rec: Recommendation, profile: OperatorBeliefState) -> str:
    """Technical framing: support burden, on-call impact, longer horizon."""
    return f"""\
[Frame: technical / {profile.decision_horizon}]

Recommendation: {rec.action}.

Engineering impact is indirect, via customer support load. If the price
increase is increasing churn at approximately +{rec.estimated_effect_pp:.2f} pp on the
affected base, the corresponding upstream signal is increased support tickets
and cancellation interactions in the 30-60 days preceding churn.

The headline statistical issue: the conditional churn rate looks favorable
for the price increase, but this is a confounding artifact (the do-calculus
backdoor adjustment over product_health reverses the sign of the estimated
effect). The architectural answer is grounded in a structural causal model
rather than a conditional summary, which is why it disagrees with the LLM-style
observational answer.

Engineering implications: if the rollback proceeds, expect a modest reduction
in churn-related support volume over the next 1-2 quarters. No on-call burden
change is expected."""


# ─── Adaptive routing ────────────────────────────────────────────────────

def adapt_for_audience(rec: Recommendation, profile: OperatorBeliefState) -> str:
    """Route the recommendation to the appropriate framing function."""
    if profile.vocabulary == "financial":
        return frame_for_cfo(rec, profile)
    if profile.vocabulary == "operational":
        return frame_for_cro(rec, profile)
    if profile.vocabulary == "technical":
        return frame_for_vpe(rec, profile)
    raise ValueError(f"No framing for vocabulary: {profile.vocabulary}")


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Theory of mind demo: audience-adapted framing of one recommendation."
    )
    parser.add_argument(
        "--audience", choices=["cfo", "cro", "vpe", "all"], default="all",
        help="Which operator profile to render (default: all three)",
    )
    args = parser.parse_args()

    rec = GROUND_TRUTH_RECOMMENDATION

    print("Underlying recommendation (architecturally grounded, audience-independent):")
    print(f"  Action:    {rec.action}")
    print(f"  Effect:    {rec.estimated_effect_pp:+.2f} pp "
          f"(95% CI: {rec.confidence_interval_pp[0]:+.2f}, {rec.confidence_interval_pp[1]:+.2f})")
    print(f"  Affected:  {rec.affected_population_pct:.0f}% of customers")
    print()
    print("=" * 72)
    print()

    audiences = ["cfo", "cro", "vpe"] if args.audience == "all" else [args.audience]

    for i, key in enumerate(audiences):
        profile = PROFILES[key]
        framed = adapt_for_audience(rec, profile)

        print(f"For {profile.role}")
        print("─" * 72)
        print(framed)
        if i < len(audiences) - 1:
            print()
            print("=" * 72)
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
