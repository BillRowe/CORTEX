# cortex2 — Python skeleton package

CORTEX-2 ("The Predictive Brain") reference Python implementation.

This is the **skeleton** package: a complete, type-annotated, importable
Python package whose method bodies raise `NotImplementedError`. It exists
to:

1. Validate that the architecture described in `CORTEX-ARCH-002` is
   implementable as Python code with consistent type signatures.
2. Provide a concrete contract for the engineering team building the
   real implementation in Phases 0 through 4 of `CORTEX-ROAD-001`.
3. Allow downstream consumers (test harnesses, integration code,
   documentation tooling) to be written and unit-tested before the
   bodies are filled in.

## Quick start

```python
import cortex2

brain = cortex2.CORTEX()         # instantiates all 10 regions + 5 infra components

print(brain.r1)                   # MultimodalPerceptionLayer
print(brain.r10)                  # TheoryOfMindNetwork
print(brain.locus_coeruleus)      # LocusCoeruleus

brain.process_text("hello", session_id="s1")
# raises NotImplementedError — Phase 0 deliverable D0.6
```

## Package structure

```
cortex2/
├── __init__.py                    # CORTEX facade (the public API)
├── enums.py                        # ModalityType, RegionId, ReasoningMode, ...
├── data_contracts.py               # 12 dataclasses (7 v1 + 5 v2 NEW)
│
├── r1_thalamus_pulvinar.py         # multi-modal perception + Pulvinar gating
├── r2_dlpfc.py                     # working memory (v1 unchanged)
├── r3_hippocampus.py               # long-term memory (v1 unchanged)
├── r4_pfc.py                       # System 2 + CausalDiscovery + ActiveInference
├── r5_caudate.py                   # System 1 fast pathway (v2 NEW)
├── r6_cerebellum.py                # learning + glymphatic consolidation
├── r7_acc.py                       # alignment + System 1/2 routing + AI goals
├── r8_motor_cortex.py              # action and output (v1 + ToM hook)
├── r9_default_mode_network.py      # predictive coding core (v2 NEW)
├── r10_theory_of_mind.py           # operator modelling (v2 NEW)
│
├── infra_uncertainty.py            # Insular Cortex (v2 modified)
├── infra_audit.py                  # Entorhinal Cortex (v2 + 4 event types)
├── infra_gnn_bridge.py             # Corpus Callosum (v2 + 2 translators)
├── infra_human_oversight.py        # Dorsomedial PFC (v2 + ToM framing)
└── infra_locus_coeruleus.py        # novelty signalling (v2 NEW)
```

17 modules, ~4,000 lines of typed Python with full docstrings.

## The seven hero advances

| #  | Advance                          | Lives in                       |
|----|----------------------------------|--------------------------------|
| 1  | Predictive Cortical Hierarchy    | R9 DefaultModeNetwork          |
| 2  | Dual-Process Reasoning           | R5 Caudate + R4 PFC + R7 router |
| 3  | Theory of Mind                    | R10 TheoryOfMindNetwork        |
| 4  | Active Inference                  | R7 + R4.ActiveInferencePlanner |
| 5  | Causal Discovery                  | R4.CausalDiscovery             |
| 6  | Sparse Thalamic Routing           | R1.Pulvinar                    |
| 7  | Glymphatic Consolidation          | R6.GlymphaticConsolidation     |

## Backward compatibility with v1

All v1 facade methods are preserved verbatim:

```
start_session, end_session, process_text, reason, respond, add_goal, status
```

All v1 dataclasses are preserved verbatim (the new optional fields are
additive, so v1 client code that deserialises a v2 `CognitiveRepresentation`
or `ReasoningChain` continues to work).

## v2 additions on the facade

**Multi-modal ingestion:**
`process_image`, `process_video`, `process_audio`, `process_timeseries`,
`process_documents`, `process_multimodal`, `stream_ingest`

**Hero-advance methods:**
`discover_causal_graph`, `reason_with_discovered_graph` (#5);
`model_operator`, `update_operator_belief`, `get_operator_state`,
`adapt_output` (#3);
`plan_active_inference` (#4);
`get_prediction`, `get_prediction_error`, `incubate_hypotheses` (#1);
`trigger_consolidation`, `get_consolidation_log` (#7);
`get_routing_stats`, `report_region_load` (#2/#6).

## Validation

Run from the package root:

```bash
python -c "import cortex2; cortex2.CORTEX(); print('OK')"
```

UNCLASSIFIED // FOR OFFICIAL USE ONLY
