# Contributing

Thank you for your interest in this project. This document describes how to contribute, what kinds of contributions are most valuable right now, and what to expect from the review process.

This is independent research, currently maintained by a single author. That shapes the contribution dynamics in a few ways worth knowing up front: response times are measured in days rather than hours, the bar for merging is "does this make the work better" rather than "does this satisfy a feature request," and substantive intellectual disagreements are welcome — they make the work stronger.

---

## Most valuable contributions right now

Some kinds of contribution are unusually high-leverage given where the project is. In rough order of how useful they would be:

**Independent reproduction of the causal-reasoning demo.** Run `examples/causal_demo.py` on your machine, with different seeds, on different hardware. If your results match the documented output, file an issue confirming reproduction. If they don't, file an issue with the discrepancy. Both are valuable — confirmation strengthens the central claim; disagreement reveals an environmental dependency that needs to be documented or fixed.

**Reproduction with different SCMs.** The headline demo uses one synthetic structural causal model. The architectural argument should generalize to any confounded SCM. If you construct a different SCM (different confounding structure, different effect sizes, continuous variables instead of binary) and the architecture's answer remains within its 95% CI of the ground truth while a transformer-based recommender produces a sign-reversed estimate, that's a meaningful extension of the claim. Pull request the SCM specification and the demo script.

**Implementation of additional causal-discovery algorithms.** The current implementation uses the PC algorithm. FCI (Fast Causal Inference) for handling latent confounders, GES (Greedy Equivalence Search), and LiNGAM are natural additions. Each lives behind the `CausalReasoner.discover_graph` interface in `src/layer4_reasoning/causal.py`.

**Evaluation harnesses for FANToM and BigToM.** The paper makes a falsifiable prediction that the architecture should exceed 85% on these benchmarks where transformer baselines plateau at 60–75%. A repeatable evaluation harness — code that runs the architecture against the benchmark and produces a comparable score — is the natural way to test the prediction. This is the highest-leverage empirical contribution available.

**Adversarial test cases for the alignment pipeline.** The L6 six-stage authorization pipeline should not be circumventable by adversarial inputs. If you can construct an input that produces an authorized action that the constitutional checks should have blocked, that's a real bug and a meaningful contribution. File it as an issue with the input, the produced authorization, and your reasoning for why it should have been blocked.

**Real-world domain pilots.** If you're working in a domain where the architecture's specific capabilities (causal reasoning, calibrated uncertainty, theory of mind, alignment) would be useful, get in touch. Pilot deployments require data-handling agreements and substantive collaboration; they aren't a pull-request matter. Open an issue or email directly.

For typo-level fixes, broken links, formatting cleanup, and other small mechanical contributions, just open a PR.

---

## Before you start substantive work

Open an issue first. Two reasons.

The first is coordination — there may already be work in progress on what you're considering, or design constraints that aren't visible from the outside. A short conversation up front saves both sides time later.

The second is calibration. Some contributions that look small are load-bearing (anything touching `core/` or the L6 alignment pipeline), and some that look large are actually straightforward (new examples, additional benchmarks). The issue thread is where you find out which kind you're looking at before committing to the full implementation.

For typos, broken links, and obvious cleanup: skip the issue, just open a PR. The "open an issue first" rule is for substantive contributions, not for fixing a misspelling.

---

## Pull request process

When you open a PR:

- Reference the issue it addresses, if there is one.
- Describe what changed and why. "What changed" is usually obvious from the diff; "why" is usually the part that needs explanation.
- If the PR changes behavior, update the relevant documentation — README, ARCHITECTURE, INSTALLATION, DESIGN_PRINCIPLES, or CAUSAL_DEMO. PRs that change behavior without updating docs will be sent back.
- If the PR changes the demo's numerical output, update CAUSAL_DEMO and the README to match. The numbers in the docs need to track the numbers the demo actually produces.
- For code changes, run the existing tests and add new ones where appropriate. The standard is "the test would fail without your change" — not "the test exists and passes."

Review will typically take 3–7 days. Substantive changes may take longer because they involve more careful review. If a PR has been open for two weeks with no response, ping the issue politely.

PRs may be sent back, accepted with modifications, or accepted as-is. They may also occasionally be declined — this happens most often when a contribution would conflict with one of the architectural principles in DESIGN_PRINCIPLES.md, or when the change has scope implications that aren't yet resolved. A decline isn't a judgment on the contribution; it's a statement that the project isn't ready for that change yet.

---

## Code style

Python code should follow standard conventions:

- **Formatting**: handled by `ruff format` (configured in `pyproject.toml`). Run before committing.
- **Linting**: `ruff check` should pass cleanly. Most issues auto-fix with `ruff check --fix`.
- **Typing**: code should be `mypy --strict` clean. Untyped third-party dependencies can use `# type: ignore` with a comment explaining why.
- **Imports**: standard library first, then third-party, then local. `ruff` will sort these for you.
- **Docstrings**: public functions and classes should have docstrings. The format is informal — describe what it does, what the inputs mean, what it returns. Keep it short.

For commit messages: a short imperative subject line ("Add FCI causal discovery") followed by a blank line and a longer description if needed. Avoid "fixed bug" and "updated code" as commit subjects.

---

## Architectural commitments

A few things are deliberately not up for discussion at the PR level. Changes to any of these require a full design discussion, which means an issue with substantive engagement before any code is written:

- The seven-layer numbering and the layer functions. Adding "Layer 8" or splitting Layer 4 is a real architectural change, not a refactor.
- The five constitutional checks in L6. These are the alignment foundation; revision is deliberate.
- The audit log format. Tamper-resistance requires stability.
- The three-store memory discipline (semantic / episodic / procedural). Conflating them is the bug, not the feature.
- The "cognitive function" naming convention for layers (e.g., L4 is "Reasoning and Planning," not "Inference Service"). Cognitive coherence is load-bearing.

The full rationale for each is in [docs/DESIGN_PRINCIPLES.md](docs/DESIGN_PRINCIPLES.md). If you find yourself wanting to change any of them, read the relevant principle first — your reasoning may already be addressed there. If it isn't, the issue thread is the place to make the case.

---

## Reporting bugs

For functional bugs (something doesn't work as documented):

1. Check the existing issues to see if it's already reported.
2. If not, open an issue with: what you tried, what you expected, what actually happened, and your environment (OS, Python version, key dependency versions). The output of `python -m src.diagnostics` is helpful when relevant.
3. If you can produce a minimal reproduction (smallest possible input that triggers the bug), include it.

For documentation bugs (something is unclear, wrong, or missing): open an issue or just submit a PR with the fix. Both are welcome.

For security-related issues: do **not** open a public issue. See [SECURITY.md](SECURITY.md) for the responsible-disclosure process.

---

## Asking questions

For research questions that aren't quite issues — "I'm curious whether X would work," "I'm trying to understand Y," "I want to use this for Z, does that make sense?" — open a Discussion (if Discussions are enabled on the repo) or an issue tagged `question`.

For private research conversations or potential collaborations: email directly. The address is in the repo's metadata and in the citation block of the README.

---

## What to expect from the maintainer

- Response within 3–7 days for issues and PRs in most cases.
- Substantive engagement on substantive contributions. If you've put real work into something, you'll get a real review.
- Disagreement when warranted. If your contribution conflicts with the project's direction, you'll be told so directly, with reasoning. The goal is not to be agreeable but to make the work better.
- Honest accounting of where the project is. The implementation status table in the README is meant to be accurate. If you find places where the documentation overclaims or underclaims, that's worth flagging.

---

## What I won't do

- Merge contributions that conflict with the architectural commitments listed above without a real design discussion first.
- Accept large contributions that arrive without prior issue discussion — even if the contribution is technically correct, the scope and integration questions need to be worked out before the merge.
- Promise specific timelines for response. This is a part-time project for me; I'll get to things as I can.
- Engage with contributions that are hostile or disrespectful in tone. Disagreement is welcome; rudeness isn't.

---

## License

By contributing, you agree that your contributions will be licensed under the same Apache-2.0 license as the project. The patent grant in Apache-2.0 means contributors grant a patent license on their contributions, which protects everyone using the project. If you're not in a position to make this grant (for example, if your employer has IP claims on your work), please coordinate with them before contributing.

---

Thanks again for your interest. The project is more useful with collaborators, and I'm grateful for the time and attention contributors put into it.
