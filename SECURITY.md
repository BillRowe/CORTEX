# Security Policy

This document describes how to report security vulnerabilities in this project. The short version: please report privately rather than publicly, give a reasonable disclosure timeline, and expect a response within a week.

---

## Scope

This project is a research-stage cognitive architecture. Security concerns fall into two broad categories:

**Conventional software security** — vulnerabilities in the code that could allow unauthorized access, code execution, data exposure, or denial of service. Examples: injection attacks against the L7 action dispatcher, deserialization bugs in the audit log, authentication bypass in the human oversight interface, or supply-chain attacks via the dependency graph.

**Architectural security** — issues specific to the cognitive architecture that could allow the system to behave in ways the alignment pipeline should have prevented. Examples: inputs that produce authorized actions which constitutional checks should have blocked, mechanisms that allow the audit log to be silently truncated or rewritten, prompt-injection attacks that propagate through the L7 action interface, or seam-exploitation attacks where an adversary uses the boundary between two layers to bypass alignment guarantees.

Both categories should be reported through the same channel. Architectural security issues are particularly valuable contributions because they engage with the project's core claims; if you find one, please report it.

Out of scope:

- Vulnerabilities in third-party dependencies (Neo4j, Qdrant, language model APIs, etc.). Report those to the upstream projects directly. If you find a vulnerability in this project that is *triggered* by a third-party dependency, that may be in scope — file a report and let the maintainer triage.
- Issues in deployments of the architecture by other parties. This repository is the reference implementation; if a third party has deployed it with their own configuration and you've found an issue with their deployment specifically, contact them.
- Theoretical concerns about the architecture's design that don't have a concrete exploitation path. These are valuable but belong in regular issues or design discussions, not the security-disclosure channel.

---

## How to report

Please report security vulnerabilities by **opening a private security advisory** on GitHub, using the "Security" tab of the repository. This routes the report directly to the maintainer through GitHub's private-disclosure workflow.

If you cannot use the GitHub private advisory system, email the maintainer directly at `[security contact email]`. Use the subject line `SECURITY: <brief description>` so the report is recognizable. PGP keys are not required, but if you wish to encrypt the report, the maintainer's public key is available on request.

**Do not open a public issue or pull request for security vulnerabilities.** Even if the project is small, a public report gives potential attackers more information than the eventual fix gives defenders. Private disclosure protects users.

When reporting, please include:

1. A description of the vulnerability — what it is, what it allows.
2. The steps to reproduce it, ideally as a minimal example.
3. The version or commit hash you tested against.
4. Your assessment of the severity (high / medium / low) and the kind of attacker who could exploit it (anyone with internet access, anyone with code-execution access, an authenticated user, etc.).
5. Any suggestions for how to fix it, if you have them. (Not required — finding the bug is the contribution; you don't owe a patch.)

If you've already developed a patch, you can include it in the report. The maintainer will review the patch alongside the disclosure.

---

## What to expect from the maintainer

The maintainer commits to:

- **Acknowledging your report within 7 days.** This is an acknowledgment that the report has been received and is being investigated, not necessarily a full triage decision.
- **Providing a triage decision within 14 days.** This includes an assessment of severity, whether the issue is in scope, and a rough timeline for a fix.
- **Producing a fix proportional to the severity.** Critical vulnerabilities (remote code execution, alignment bypass, audit-log compromise) will be prioritized. Lower-severity issues may take longer.
- **Coordinating disclosure with you.** The standard is to publish a fix and a public advisory together, with a credit to you (unless you prefer to remain anonymous). Disclosure timeline is typically 90 days from the initial report, but can be shorter for low-severity issues or longer for issues requiring coordinated upstream changes.
- **Crediting your contribution publicly** in the security advisory and project changelog, unless you request otherwise.

This is a part-time research project, so the timelines above are best-effort rather than contractual. If a critical issue requires faster turnaround, please flag that in your report.

---

## What constitutes responsible disclosure

In short: report privately, give the maintainer a reasonable amount of time to fix, then disclose publicly with full technical detail.

The typical timeline is:

```
Day 0:    You report the issue privately.
Day 7:    Maintainer acknowledges receipt.
Day 14:   Maintainer provides triage decision and rough timeline.
Day 30-90: Fix is developed, tested, and released. Longer for complex issues.
Day of fix release: Public advisory is published, with credit to you.
Day of fix release + 30: Detailed write-up may be published, if you wish.
```

If the maintainer is unresponsive or unable to fix within 90 days, you are within your rights to publish your findings independently. Please give written notice 14 days before public disclosure so any partial mitigations can be communicated to users.

If the vulnerability is being actively exploited in the wild, the disclosure timeline is compressed — let the maintainer know immediately, and expect coordination on a faster fix.

---

## Architectural-security specifics

A few categories of issue deserve particular attention because they engage with the project's central technical claims:

**Alignment-pipeline bypass.** Any input that causes L7 to dispatch an action that would have failed one of the L6 constitutional checks if those checks had been correctly evaluated. This is a core safety property of the architecture; reports here are highly valued.

**Audit-log compromise.** Any mechanism that allows the audit log to be silently modified, truncated, or rewritten. The chain hash is supposed to make tampering detectable; if you find a way around that, it's a serious issue.

**Causal-reasoning manipulation.** Any input that causes the L4 causal reasoner to produce a confidently-wrong causal estimate that the architecture nonetheless surfaces as "high confidence." The architecture's safety story depends on the calibration of its own confidence; manipulations that defeat that calibration are alignment-relevant.

**Memory-store contamination.** Any mechanism by which a single bad input can cause permanent contamination of the semantic memory store. The three-store discipline is supposed to prevent this; if you find a way around it, it matters.

**Seam exploitation.** Attacks that exploit the boundary between two layers — for example, by producing an L4 reasoning trace that L5 will interpret in a way that contaminates L3. These are the hardest issues to find and the most interesting to report.

For all of these, please err on the side of reporting. A "soft" vulnerability that requires unusual conditions is still worth knowing about; a clean reproduction is not required for the report to be valuable.

---

## What this policy does not cover

This policy covers vulnerabilities in the codebase. It does not cover:

- **Misuse of the architecture by third parties.** If someone deploys this code in a way that produces harmful outcomes, the responsibility is theirs, not the project's. The architecture provides safety primitives; deployers are responsible for using them correctly.
- **Disagreements about design choices.** If you think a design choice is wrong, that's a regular issue or a discussion thread, not a security report. Calling something a security issue when it's actually a design disagreement makes the disclosure process less useful for everyone.
- **Reports that amount to "this could be misused."** Most software could in principle be misused. Reports need to identify a specific mechanism by which the project's stated security properties fail, not a general concern about the existence of the project.
- **Performance issues that aren't denial-of-service vulnerabilities.** Slow code is a regular issue; an unbounded resource consumption that an attacker can trigger is a security issue. Use judgment.

---

## Bug bounties

The project does not currently offer a bug bounty. Reports are entirely voluntary, and credit is the only compensation offered. If the project's funding situation changes, this may be revisited.

If you've put substantial work into a report and would like to be credited prominently — for example, you found a serious vulnerability and helped develop the fix — please mention this in your report. Substantive contributions can be acknowledged in the project's documentation in addition to the security advisory.

---

## Contact

Primary: GitHub Security Advisory (Security tab of the repository).
Backup: webrep@yahoo.com

For non-security questions about the project, please use regular issues or the contact information in the README.

---

Thank you for taking the time to report responsibly. Security work is often invisible and unappreciated; this project will try to do better than that.
