"""
cortex2.infra_locus_coeruleus  —  Locus Coeruleus · novelty signalling

Architecture reference:  CORTEX-ARCH-002 §7.1 Locus Coeruleus.
Roadmap reference:       CORTEX-ROAD-001 §2 Phase 0 (paired with predictive coding)

In mammalian brains, the Locus Coeruleus (LC) releases norepinephrine in
response to salient or novel stimuli, acting as a global "wake up" signal
that increases neural gain across the cortex. The CORTEX-2 LocusCoeruleus
is the architectural analogue: it monitors prediction-error magnitudes from
the predictive hierarchy (R9 → lower regions) and emits a global novelty
signal that gates two things:

    1. System engagement — high novelty → R7 routes to System 2 (R4 PFC)
    2. Learning rate    — high novelty → R6 boosts LoRA learning rate
    3. Attention gain    — high novelty → R2 amplifies salience scores

The LC is intentionally simple: a rolling distribution of prediction errors
per region with z-score outlier detection. Its value is not algorithmic
sophistication but in its role as a single, auditable signalling channel
that ties together the predictive, routing, and learning systems.

Status: v2 NEW — no v1 predecessor.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from .data_contracts import PredictionError
from .enums import RegionId


@dataclass
class NoveltySignal:
    """Output of NoveltyMonitor.assess(). The novelty_score is what
    downstream consumers (R7 router, R6 learning rate, R2 attention) read.
    """
    region:               RegionId
    novelty_score:        float                # [0,1]; >0.5 typically gates System 2
    z_score:              float = 0.0          # signed standard deviations from mean
    rolling_mean:         float = 0.0
    rolling_std:          float = 0.0
    sample_size:          int = 0
    timestamp:            datetime = field(default_factory=datetime.utcnow)


class NoveltyMonitor:
    """Tracks rolling distribution of prediction-error magnitudes per region.

    Uses Welford's online algorithm for numerically stable mean/variance
    over a configurable rolling window (default last 1000 errors per region).
    """

    def __init__(self, window_size: int = 1000, novelty_z_threshold: float = 2.0):
        self.window_size = window_size
        self.novelty_z_threshold = novelty_z_threshold
        self._buffers: Dict[RegionId, Deque[float]] = {}

    def observe(self, error: PredictionError) -> NoveltySignal:
        """Record a prediction error and compute the resulting novelty signal."""
        raise NotImplementedError("v2 stub — implement in Phase 0 (D0.5)")

    def current_signal(self, region: RegionId) -> Optional[NoveltySignal]:
        """Return the most recent novelty signal for a region, or None
        if no errors have been observed yet."""
        raise NotImplementedError("v2 stub")


class SystemEngagementGate:
    """Translates a novelty signal into a routing recommendation that R7's
    SystemRouter consumes.

    Below the engagement threshold: route to R5 (System 1).
    At or above threshold: route to R4 (System 2).

    The gate is advisory; R7's SystemRouter combines this with stakes and
    operator-belief uncertainty before making the final decision.
    """

    def __init__(self, engagement_threshold: float = 0.50):
        self.engagement_threshold = engagement_threshold

    def recommend(self, signal: Optional[NoveltySignal]) -> str:
        """Returns 'system_1' | 'system_2'. Defaults to 'system_1' when no
        signal is present (no surprise → cached path is fine)."""
        if signal is None:
            return "system_1"
        return "system_2" if signal.novelty_score >= self.engagement_threshold else "system_1"


class LearningRateModulator:
    """Boosts R6's effective learning rate proportionally to the recent
    novelty signal. High novelty (genuine surprise) → faster adaptation.

    Multiplier formula: 1.0 + boost_max · novelty_score.
    With boost_max=2.0 and novelty_score=1.0, learning rate is 3× baseline.
    """

    def __init__(self, baseline_lr: float = 1e-3, boost_max: float = 2.0):
        self.baseline_lr = baseline_lr
        self.boost_max = boost_max

    def modulated_lr(self, signal: Optional[NoveltySignal]) -> float:
        if signal is None:
            return self.baseline_lr
        multiplier = 1.0 + self.boost_max * signal.novelty_score
        return self.baseline_lr * multiplier


class AttentionGain:
    """Multiplicatively amplifies R2's salience scores when novelty is high.

    Mirrors the biological role of LC norepinephrine in increasing neural
    gain across cortex during salient events.
    """

    def __init__(self, gain_max: float = 1.5):
        self.gain_max = gain_max

    def gain_factor(self, signal: Optional[NoveltySignal]) -> float:
        if signal is None:
            return 1.0
        return 1.0 + (self.gain_max - 1.0) * signal.novelty_score


class LocusCoeruleus:
    """The cross-cutting LC component itself.

    Public interface used by R7 (router), R6 (learning), R2 (attention).
    Aggregates the four sub-components into a single audit-friendly handle.
    """
    region_id = RegionId.LOCUS_COERULEUS

    def __init__(self,
                 monitor: Optional[NoveltyMonitor] = None,
                 engagement: Optional[SystemEngagementGate] = None,
                 lr_modulator: Optional[LearningRateModulator] = None,
                 attention: Optional[AttentionGain] = None):
        self.monitor = monitor or NoveltyMonitor()
        self.engagement = engagement or SystemEngagementGate()
        self.lr_modulator = lr_modulator or LearningRateModulator()
        self.attention = attention or AttentionGain()

    def observe(self, error: PredictionError) -> NoveltySignal:
        """Process an incoming prediction error. Returns the resulting
        novelty signal for downstream consumers to react to."""
        return self.monitor.observe(error)

    def current_signal(self, region: RegionId) -> Optional[NoveltySignal]:
        return self.monitor.current_signal(region)
