"""
Synthetic structural causal model for the headline causal-reasoning demo.

This module specifies a structural causal model with one latent confounder,
one binary treatment, and one binary outcome. The true causal effect of
treatment on outcome is +3.00 percentage points by construction.

The model is the smallest non-trivial confounded SCM that exhibits the
sign-reversal phenomenon central to the architecture's causal-reasoning
claim: the conditional probability E[Y | X=1] - E[Y | X=0] has the
opposite sign to the true causal effect E[Y | do(X=1)] - E[Y | do(X=0)],
because treatment is non-randomly assigned by the confounder.

Variables
---------
product_health (H) : binary, latent confounder
    P(H=1) = 0.66   ("healthy")
    P(H=0) = 0.34   ("unhealthy")

price_increase (X) : binary, treatment
    P(X=1 | H=1) = 0.75    healthy customers more likely to receive increase
    P(X=1 | H=0) = 0.30    unhealthy customers less likely

churn (Y) : binary, outcome
    P(Y=1 | X=0, H=1) = 0.05   healthy & no increase: low churn
    P(Y=1 | X=1, H=1) = 0.08   healthy & increase:    +3pp on healthy baseline
    P(Y=1 | X=0, H=0) = 0.24   unhealthy & no increase: high churn
    P(Y=1 | X=1, H=0) = 0.27   unhealthy & increase:  +3pp on unhealthy baseline

True causal effect
------------------
ATE = E[Y | do(X=1)] - E[Y | do(X=0)] = +0.03 = +3.00 pp

By construction: the price increase adds exactly 3pp to the churn probability,
regardless of product health. This is the additive structure that makes the
example clean.

Causal graph
------------
    H ──> X
    H ──> Y
    X ──> Y

H is a confounder on the X→Y relationship via the backdoor path X←H→Y.
The backdoor adjustment formula stratifies over H to recover the true effect:

    ATE = Σ_h [E(Y | X=1, H=h) - E(Y | X=0, H=h)] · P(H=h)
        = (0.08 - 0.05) · 0.66 + (0.27 - 0.24) · 0.34
        = 0.03 · 0.66 + 0.03 · 0.34
        = 0.03

For the full walkthrough, see docs/CAUSAL_DEMO.md.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ─── Model parameters ────────────────────────────────────────────────────
# These are the SCM parameters by construction. They define the ground truth.

P_HEALTHY = 0.66                     # P(product_health = 1)

P_TREAT_GIVEN_HEALTHY = 0.75         # P(price_increase = 1 | product_health = 1)
P_TREAT_GIVEN_UNHEALTHY = 0.30       # P(price_increase = 1 | product_health = 0)

P_CHURN_HEALTHY_NOTREAT = 0.05       # P(churn = 1 | X=0, H=1)
P_CHURN_HEALTHY_TREAT = 0.08         # P(churn = 1 | X=1, H=1) — +3pp on baseline
P_CHURN_UNHEALTHY_NOTREAT = 0.24     # P(churn = 1 | X=0, H=0)
P_CHURN_UNHEALTHY_TREAT = 0.27       # P(churn = 1 | X=1, H=0) — +3pp on baseline

TRUE_ATE = 0.03                      # +3.00 percentage points, by construction


# ─── Specification dataclass for self-documentation ──────────────────────

@dataclass(frozen=True)
class SCMSpec:
    """The full specification of the synthetic structural causal model."""

    p_healthy: float = P_HEALTHY
    p_treat_given_healthy: float = P_TREAT_GIVEN_HEALTHY
    p_treat_given_unhealthy: float = P_TREAT_GIVEN_UNHEALTHY
    p_churn_healthy_notreat: float = P_CHURN_HEALTHY_NOTREAT
    p_churn_healthy_treat: float = P_CHURN_HEALTHY_TREAT
    p_churn_unhealthy_notreat: float = P_CHURN_UNHEALTHY_NOTREAT
    p_churn_unhealthy_treat: float = P_CHURN_UNHEALTHY_TREAT

    @property
    def true_ate(self) -> float:
        """The true average treatment effect computed from the SCM parameters."""
        effect_in_healthy = self.p_churn_healthy_treat - self.p_churn_healthy_notreat
        effect_in_unhealthy = self.p_churn_unhealthy_treat - self.p_churn_unhealthy_notreat
        return (
            effect_in_healthy * self.p_healthy
            + effect_in_unhealthy * (1 - self.p_healthy)
        )


DEFAULT_SPEC = SCMSpec()


# ─── Data generation ─────────────────────────────────────────────────────

def generate_scm_data(
    n: int = 5000,
    seed: int = 100,
    spec: SCMSpec = DEFAULT_SPEC,
) -> pd.DataFrame:
    """
    Generate n samples from the structural causal model.

    Parameters
    ----------
    n : int
        Number of samples to draw.
    seed : int
        Random seed for reproducibility. seed=100 produces the documented
        numerical outputs (10.92% / 16.76% conditional, +2.51 pp ATE estimate).
    spec : SCMSpec
        Model parameters. Defaults to the standard specification.

    Returns
    -------
    pd.DataFrame with columns:
        product_health  (int, 0 or 1)
        price_increase  (int, 0 or 1)
        churn           (int, 0 or 1)
    """
    rng = np.random.default_rng(seed)

    # 1. Sample latent confounder H.
    product_health = rng.binomial(1, spec.p_healthy, size=n)

    # 2. Sample treatment X conditioned on H.
    p_treat = np.where(
        product_health == 1,
        spec.p_treat_given_healthy,
        spec.p_treat_given_unhealthy,
    )
    price_increase = rng.binomial(1, p_treat, size=n)

    # 3. Sample outcome Y conditioned on (X, H).
    p_churn = np.where(
        (product_health == 1) & (price_increase == 0), spec.p_churn_healthy_notreat,
        np.where(
            (product_health == 1) & (price_increase == 1), spec.p_churn_healthy_treat,
            np.where(
                (product_health == 0) & (price_increase == 0), spec.p_churn_unhealthy_notreat,
                spec.p_churn_unhealthy_treat,
            ),
        ),
    )
    churn = rng.binomial(1, p_churn, size=n)

    return pd.DataFrame({
        "product_health": product_health,
        "price_increase": price_increase,
        "churn": churn,
    })


# ─── Sanity check when run directly ──────────────────────────────────────

if __name__ == "__main__":
    spec = DEFAULT_SPEC
    print("SCM Specification")
    print("─" * 60)
    print(f"  P(product_health = healthy)              = {spec.p_healthy:.2f}")
    print(f"  P(price_increase | healthy)              = {spec.p_treat_given_healthy:.2f}")
    print(f"  P(price_increase | unhealthy)            = {spec.p_treat_given_unhealthy:.2f}")
    print(f"  P(churn | no_increase, healthy)          = {spec.p_churn_healthy_notreat:.2f}")
    print(f"  P(churn | increase,    healthy)          = {spec.p_churn_healthy_treat:.2f}")
    print(f"  P(churn | no_increase, unhealthy)        = {spec.p_churn_unhealthy_notreat:.2f}")
    print(f"  P(churn | increase,    unhealthy)        = {spec.p_churn_unhealthy_treat:.2f}")
    print()
    print(f"  True ATE (by construction)               = +{spec.true_ate * 100:.2f} pp")
    print()

    data = generate_scm_data(n=5000, seed=100)
    print("Sample of generated data (n=5000, seed=100):")
    print(data.head(10).to_string(index=False))
    print()
    print(f"  Marginal P(price_increase = 1) = {data['price_increase'].mean():.4f}")
    print(f"  Marginal P(churn = 1)          = {data['churn'].mean():.4f}")
