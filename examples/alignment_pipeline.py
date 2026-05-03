"""
Alignment pipeline demo: the six-stage L6 authorization process.

This example demonstrates that every proposed action passes through six
checks before reaching L7 for execution. Failure at any stage either blocks
the action or escalates to a human operator. There is no path from L4
reasoning to L7 action that bypasses L6.

The constitutional checks are intentionally hardcoded predicates, not
learned classifiers. The logic is: a system that can re-write its own
safety constraints can be argued into not having them; a system whose
constraints are concrete code cannot.

For the architectural detail, see docs/ARCHITECTURE.md, Section "L6
Goals, Metacognition, Alignment" and docs/DESIGN_PRINCIPLES.md, Principle P5.

Usage:
    python examples/alignment_pipeline.py
    python examples/alignment_pipeline.py --case all
    python examples/alignment_pipeline.py --case low_confidence
"""

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ─── Action and result types ─────────────────────────────────────────────

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Reversibility(Enum):
    REVERSIBLE = "reversible"
    SOFT_REVERSIBLE = "soft_reversible"   # reversible at some cost
    IRREVERSIBLE = "irreversible"


@dataclass(frozen=True)
class ProposedAction:
    name: str
    description: str
    confidence: float                     # confidence in the action being correct
    confidence_ece: float                 # expected calibration error of the confidence
    in_competence_boundary: bool          # whether this query is within validated scope
    aligned_with_goals: bool              # consistent with active goal hierarchy
    severity: Severity
    reversibility: Reversibility
    affects_user_data: bool
    is_deceptive: bool
    bypasses_audit: bool


class Stage(Enum):
    CONFIDENCE = "confidence_threshold"
    COMPETENCE = "competence_boundary"
    GOAL_ALIGNMENT = "goal_alignment"
    CONSTITUTIONAL = "constitutional_checks"
    REVERSIBILITY = "reversibility_or_authorization"
    AUDITABILITY = "audit_log_writable"


class Outcome(Enum):
    AUTHORIZED = "authorized"
    BLOCKED = "blocked"
    ESCALATED = "escalated_to_human"


@dataclass(frozen=True)
class StageResult:
    stage: Stage
    passed: bool
    note: str
    escalates: bool = False


@dataclass(frozen=True)
class AuthorizationResult:
    action: ProposedAction
    outcome: Outcome
    stages: tuple[StageResult, ...]
    final_note: str


# ─── Configuration ───────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.85               # below this, action requires escalation
ECE_BOUND = 0.10                          # confidence values with ECE > this are untrusted


# ─── The six stages ──────────────────────────────────────────────────────

def check_confidence(action: ProposedAction) -> StageResult:
    """Stage 1: confidence threshold met, with calibration also acceptable."""
    if action.confidence_ece > ECE_BOUND:
        return StageResult(
            stage=Stage.CONFIDENCE,
            passed=False,
            note=(
                f"Calibration is poor (ECE={action.confidence_ece:.2f} > "
                f"{ECE_BOUND:.2f}); confidence value is not trustworthy."
            ),
            escalates=True,
        )
    if action.confidence < CONFIDENCE_THRESHOLD:
        return StageResult(
            stage=Stage.CONFIDENCE,
            passed=False,
            note=(
                f"Confidence ({action.confidence:.2f}) is below threshold "
                f"({CONFIDENCE_THRESHOLD:.2f})."
            ),
            escalates=action.severity in (Severity.HIGH, Severity.CRITICAL),
        )
    return StageResult(
        stage=Stage.CONFIDENCE,
        passed=True,
        note=f"Confidence {action.confidence:.2f} ≥ {CONFIDENCE_THRESHOLD:.2f}, ECE acceptable.",
    )


def check_competence(action: ProposedAction) -> StageResult:
    """Stage 2: query falls within validated competence boundary."""
    if not action.in_competence_boundary:
        return StageResult(
            stage=Stage.COMPETENCE,
            passed=False,
            note="Query is outside the validated competence boundary; escalating.",
            escalates=True,
        )
    return StageResult(
        stage=Stage.COMPETENCE,
        passed=True,
        note="Query is within validated competence boundary.",
    )


def check_goal_alignment(action: ProposedAction) -> StageResult:
    """Stage 3: action is consistent with the active goal hierarchy."""
    if not action.aligned_with_goals:
        return StageResult(
            stage=Stage.GOAL_ALIGNMENT,
            passed=False,
            note=(
                "Action is inconsistent with the active goal hierarchy; "
                "escalating for operator review."
            ),
            escalates=True,
        )
    return StageResult(
        stage=Stage.GOAL_ALIGNMENT,
        passed=True,
        note="Action is consistent with active goals.",
    )


# ─── Constitutional check predicates (hardcoded, non-learnable) ──────────

def constitutional_no_irreversible_harm(action: ProposedAction) -> Optional[str]:
    """Constitutional check: no irreversible harm."""
    if action.severity == Severity.CRITICAL and action.reversibility == Reversibility.IRREVERSIBLE:
        return "Action is critical-severity AND irreversible. Constitutional violation."
    return None


def constitutional_no_deception(action: ProposedAction) -> Optional[str]:
    """Constitutional check: action is not deceptive."""
    if action.is_deceptive:
        return "Action is flagged as deceptive. Constitutional violation."
    return None


def constitutional_preserve_auditability(action: ProposedAction) -> Optional[str]:
    """Constitutional check: action does not bypass the audit log."""
    if action.bypasses_audit:
        return "Action bypasses the audit log. Constitutional violation."
    return None


def constitutional_respect_user_data(action: ProposedAction) -> Optional[str]:
    """Constitutional check: actions affecting user data require high confidence."""
    if action.affects_user_data and action.confidence < 0.95:
        return (
            f"Action affects user data; confidence {action.confidence:.2f} is below "
            f"the elevated 0.95 threshold required for such actions."
        )
    return None


CONSTITUTIONAL_CHECKS = (
    constitutional_no_irreversible_harm,
    constitutional_no_deception,
    constitutional_preserve_auditability,
    constitutional_respect_user_data,
    # In the real architecture, "honor operator authority" is a fifth check.
    # It is omitted here because it requires evaluating against operator-issued
    # directives, which adds machinery that obscures the example.
)


def check_constitutional(action: ProposedAction) -> StageResult:
    """Stage 4: action passes all hardcoded constitutional checks."""
    violations = []
    for check in CONSTITUTIONAL_CHECKS:
        violation = check(action)
        if violation is not None:
            violations.append(violation)

    if violations:
        return StageResult(
            stage=Stage.CONSTITUTIONAL,
            passed=False,
            note="\n".join(f"  ✗ {v}" for v in violations),
            escalates=False,  # constitutional violations BLOCK, do not escalate
        )
    return StageResult(
        stage=Stage.CONSTITUTIONAL,
        passed=True,
        note=f"All {len(CONSTITUTIONAL_CHECKS)} constitutional checks passed.",
    )


def check_reversibility(action: ProposedAction) -> StageResult:
    """Stage 5: action is reversible, or has explicit human pre-authorization."""
    if action.reversibility == Reversibility.REVERSIBLE:
        return StageResult(
            stage=Stage.REVERSIBILITY,
            passed=True,
            note="Action is reversible; proceeding without explicit authorization.",
        )
    if action.reversibility == Reversibility.SOFT_REVERSIBLE:
        return StageResult(
            stage=Stage.REVERSIBILITY,
            passed=True,
            note="Action is soft-reversible (reversible at some cost); proceeding.",
        )
    # Irreversible actions require explicit human pre-authorization.
    return StageResult(
        stage=Stage.REVERSIBILITY,
        passed=False,
        note=(
            "Action is irreversible. Escalating to human operator for explicit "
            "pre-authorization before execution."
        ),
        escalates=True,
    )


def check_auditability() -> StageResult:
    """Stage 6: audit log is writable. (Always true in this demo; included for completeness.)"""
    return StageResult(
        stage=Stage.AUDITABILITY,
        passed=True,
        note="Audit log writable; chain hash computable.",
    )


# ─── The pipeline ────────────────────────────────────────────────────────

def authorize(action: ProposedAction) -> AuthorizationResult:
    """Run the six-stage authorization pipeline."""
    stage_results: list[StageResult] = []

    sequence = (
        check_confidence(action),
        check_competence(action),
        check_goal_alignment(action),
        check_constitutional(action),
        check_reversibility(action),
        check_auditability(),
    )

    for result in sequence:
        stage_results.append(result)
        if not result.passed:
            outcome = Outcome.ESCALATED if result.escalates else Outcome.BLOCKED
            final_note = (
                f"Pipeline halted at stage '{result.stage.value}'."
                f" Action {outcome.value}."
            )
            return AuthorizationResult(
                action=action,
                outcome=outcome,
                stages=tuple(stage_results),
                final_note=final_note,
            )

    return AuthorizationResult(
        action=action,
        outcome=Outcome.AUTHORIZED,
        stages=tuple(stage_results),
        final_note="All six stages passed; action dispatched to L7.",
    )


# ─── Demo cases ──────────────────────────────────────────────────────────

CASES = {
    "routine": ProposedAction(
        name="Send weekly summary email",
        description="Compose and send the routine weekly summary to the operator.",
        confidence=0.94,
        confidence_ece=0.04,
        in_competence_boundary=True,
        aligned_with_goals=True,
        severity=Severity.LOW,
        reversibility=Reversibility.REVERSIBLE,
        affects_user_data=False,
        is_deceptive=False,
        bypasses_audit=False,
    ),
    "low_confidence": ProposedAction(
        name="Recommend a price-strategy reversal",
        description="Recommend rolling back the price increase based on causal analysis.",
        confidence=0.72,
        confidence_ece=0.05,
        in_competence_boundary=True,
        aligned_with_goals=True,
        severity=Severity.HIGH,
        reversibility=Reversibility.SOFT_REVERSIBLE,
        affects_user_data=False,
        is_deceptive=False,
        bypasses_audit=False,
    ),
    "out_of_competence": ProposedAction(
        name="Diagnose customer's network issue",
        description="The operator asked a question about their network that the architecture has not been validated to handle.",
        confidence=0.88,
        confidence_ece=0.06,
        in_competence_boundary=False,
        aligned_with_goals=True,
        severity=Severity.MEDIUM,
        reversibility=Reversibility.REVERSIBLE,
        affects_user_data=False,
        is_deceptive=False,
        bypasses_audit=False,
    ),
    "irreversible": ProposedAction(
        name="Delete archived customer records",
        description="Delete records that the operator flagged as archival.",
        confidence=0.96,
        confidence_ece=0.03,
        in_competence_boundary=True,
        aligned_with_goals=True,
        severity=Severity.HIGH,
        reversibility=Reversibility.IRREVERSIBLE,
        affects_user_data=True,
        is_deceptive=False,
        bypasses_audit=False,
    ),
    "constitutional_violation": ProposedAction(
        name="Send mass email with disclaimer omitted",
        description="Send a marketing email without the legally-required disclaimer.",
        confidence=0.97,
        confidence_ece=0.02,
        in_competence_boundary=True,
        aligned_with_goals=True,
        severity=Severity.HIGH,
        reversibility=Reversibility.IRREVERSIBLE,
        affects_user_data=False,
        is_deceptive=True,            # missing required disclaimer = deceptive framing
        bypasses_audit=False,
    ),
    "high_stakes_authorized": ProposedAction(
        name="Execute the recommended price-rollback",
        description="Execute the recommended price-rollback after operator pre-authorization.",
        confidence=0.93,
        confidence_ece=0.04,
        in_competence_boundary=True,
        aligned_with_goals=True,
        severity=Severity.HIGH,
        reversibility=Reversibility.SOFT_REVERSIBLE,   # rollback can be undone
        affects_user_data=False,
        is_deceptive=False,
        bypasses_audit=False,
    ),
}


# ─── Output formatting ───────────────────────────────────────────────────

OUTCOME_GLYPHS = {
    Outcome.AUTHORIZED: "✓",
    Outcome.BLOCKED: "✗",
    Outcome.ESCALATED: "⚠",
}

OUTCOME_LABELS = {
    Outcome.AUTHORIZED: "AUTHORIZED",
    Outcome.BLOCKED: "BLOCKED",
    Outcome.ESCALATED: "ESCALATED",
}


def print_authorization(result: AuthorizationResult) -> None:
    print(f"Action: {result.action.name}")
    print(f"  {result.action.description}")
    print()

    for stage_result in result.stages:
        glyph = "✓" if stage_result.passed else ("⚠" if stage_result.escalates else "✗")
        print(f"  [{glyph}] {stage_result.stage.value}")
        for line in stage_result.note.split("\n"):
            print(f"      {line}")

    print()
    print(f"  → {OUTCOME_GLYPHS[result.outcome]} {OUTCOME_LABELS[result.outcome]}")
    print(f"    {result.final_note}")


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alignment pipeline demo: six-stage authorization for proposed actions.",
    )
    parser.add_argument(
        "--case", choices=list(CASES.keys()) + ["all"], default="all",
        help=(
            "Which case to run: routine, low_confidence, out_of_competence, "
            "irreversible, constitutional_violation, high_stakes_authorized, or all "
            "(default: all)"
        ),
    )
    args = parser.parse_args()

    cases_to_run = list(CASES.keys()) if args.case == "all" else [args.case]

    for i, case_name in enumerate(cases_to_run):
        if i > 0:
            print()
            print("=" * 72)
            print()
        action = CASES[case_name]
        result = authorize(action)
        print_authorization(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
