"""
cortex2.infra_uncertainty  —  Insular Cortex · uncertainty propagation

Architecture reference:  CORTEX-ARCH-002 §7.2 Updates to existing infrastructure.

v1's three-operation framework (attach, propagate, aggregate) is preserved.
v2 modification: uncertainty is now derived directly from prediction error
variance in the predictive hierarchy. The v1 source-attached, ensemble, and
log-prob attachment methods are retained as inputs to the prediction; the
central representation is the predictive variance.

Status: MODIFIED from v1 (was infra_uncertainty in LCIF / cortex v1).
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

from .data_contracts import (
    UncertaintyEstimate, SourceProvenance, ReasoningChain, ReasoningStep,
    PredictionError, CognitiveRepresentation,
)
from .enums import SourceReliability, InformationCredibility


# ─── IC reliability/credibility multipliers ────────────────────────────────
# A–F maps to multipliers (A is best). 1–6 maps to multipliers (1 is best).
RELIABILITY_MULTIPLIER: Dict[SourceReliability, float] = {
    SourceReliability.A: 1.00, SourceReliability.B: 0.85, SourceReliability.C: 0.70,
    SourceReliability.D: 0.55, SourceReliability.E: 0.40, SourceReliability.F: 0.50,
}
CREDIBILITY_MULTIPLIER: Dict[InformationCredibility, float] = {
    InformationCredibility.CONFIRMED: 1.00,    InformationCredibility.PROBABLY_TRUE: 0.85,
    InformationCredibility.POSSIBLY_TRUE: 0.65, InformationCredibility.DOUBTFUL: 0.40,
    InformationCredibility.IMPROBABLE: 0.20,    InformationCredibility.CANNOT_JUDGE: 0.50,
}


class UncertaintyPropagator:
    """v1 inherited operations:
        attach_from_source       — IC-rated reliability × credibility
        attach_from_ensemble     — mean ± 2·std
        attach_from_logprob      — exp(log_prob)
        propagate_through_step   — min(premises) × inference_confidence
        propagate_through_chain  — iterative step propagation
        aggregate_independent    — weighted mean across sources
        aggregate_conflicting    — mean × (1 − penalty·std)
        apply_temperature_scaling — sigmoid-logit calibration

    v2 NEW operation:
        attach_from_prediction_error — confidence = 1 − sigmoid(error / scale)
            Uncertainty derived directly from the predictive hierarchy. Used
            by the DMN-driven flow when no source-based provenance exists.
    """

    def attach_from_source(self,
                           base_confidence: float,
                           provenance: SourceProvenance) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub — apply IC multipliers")

    def attach_from_ensemble(self,
                             ensemble_outputs: List[float]) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub")

    def attach_from_logprob(self, log_prob: float) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub")

    def attach_from_prediction_error(self,
                                      error: PredictionError,
                                      scale: float = 1.0) -> UncertaintyEstimate:
        """v2 NEW. Derive uncertainty from predictive-coding error."""
        raise NotImplementedError("v2 stub — implement in Phase 0 (D0.4)")

    def propagate_through_step(self,
                                premise_uncertainties: List[UncertaintyEstimate],
                                inference_confidence: float) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub")

    def propagate_through_chain(self, chain: ReasoningChain) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub")

    def aggregate_independent(self,
                               estimates: List[UncertaintyEstimate],
                               weights: Optional[List[float]] = None
                               ) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub")

    def aggregate_conflicting(self,
                              estimates: List[UncertaintyEstimate],
                              variance_penalty: float = 1.0
                              ) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub")

    def apply_temperature_scaling(self,
                                   estimate: UncertaintyEstimate,
                                   temperature: float) -> UncertaintyEstimate:
        raise NotImplementedError("v1-inherited stub")


class CalibrationMonitor:
    """v1 inherited. ECE tracking over a configurable bin count."""

    def __init__(self, n_bins: int = 10, ece_alert_threshold: float = 0.15):
        self.n_bins = n_bins
        self.ece_alert_threshold = ece_alert_threshold
        self._observations: List[Tuple[float, bool]] = []

    def record(self, predicted_confidence: float, was_correct: bool) -> None:
        self._observations.append((predicted_confidence, was_correct))

    def expected_calibration_error(self) -> float:
        raise NotImplementedError("v1-inherited stub")

    def confidence_label_icd203(self, confidence: float) -> str:
        """ICD-203 confidence labels:
              ≥85% = HIGH CONFIDENCE
              55–84% = MODERATE CONFIDENCE
              <55% = LOW CONFIDENCE
        """
        if confidence >= 0.85: return "HIGH CONFIDENCE"
        if confidence >= 0.55: return "MODERATE CONFIDENCE"
        return "LOW CONFIDENCE"
