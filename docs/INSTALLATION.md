# Installation

This document covers installation in three configurations: demo-only (the headline causal-reasoning demonstration), full-system (all layers and infrastructure active), and air-gapped (deployment in environments without external network access).

For architectural detail see [ARCHITECTURE.md](ARCHITECTURE.md). For design rationale see [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md).

---

## Quick start

If you only want to reproduce the headline causal-reasoning demonstration, the minimum installation takes about two minutes and requires no external services.

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
python -m venv .venv && source .venv/bin/activate
pip install -e ".[demo]"
python examples/causal_demo.py --seed 100
```

If everything is configured correctly, you should see the demo output reproducing the −5.84 pp / +2.51 pp comparison described in Section 6 of the paper. Skip to [Verification](#verification) below for what to expect.

---

## Prerequisites

### Hardware

| Configuration | CPU       | RAM   | Disk   | GPU                     |
| ------------- | --------- | ----- | ------ | ----------------------- |
| Demo only     | Any       | 4 GB  | 1 GB   | Not needed              |
| Full system   | 4+ cores  | 16 GB | 20 GB  | Optional (8 GB+ for L5) |
| Production    | 8+ cores  | 32 GB | 100 GB | Recommended             |

The demo runs comfortably on a laptop. The full system is heavier mainly because of the L3 memory backends (Neo4j and Qdrant) and the LLM dependency.

### Software

- **Python 3.11 or later.** The codebase uses `match` statements and several typing features that require 3.11+.
- **Operating system:** Linux and macOS are tested. Windows works under WSL2; native Windows is untested.
- **Docker** (optional, recommended for full system) — used to run Neo4j and Qdrant locally.
- **Git** for cloning the repository.

---

## Demo-only installation

The demo extra installs only the dependencies needed for the headline causal-reasoning demonstration:

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[demo]"
```

The `[demo]` extra includes:
- `numpy`, `pandas`, `scipy` — for the synthetic SCM
- `dowhy` — Pearl-style causal inference
- `causal-learn` — causal discovery (PC algorithm and descendants)
- `matplotlib` — for the diagnostic plots in the demo

It does **not** include:
- LLM client libraries
- Database drivers (Neo4j, Qdrant)
- PyTorch or PyTorch Geometric
- Test runners or development tools

To verify the demo install:

```bash
python -c "from src.layer4_reasoning.causal import CausalReasoner; print('OK')"
```

---

## Full-system installation

The full system requires three external services in addition to the Python dependencies:

1. A language model callable (any HuggingFace causal-LM or hosted API)
2. A Neo4j instance for the semantic memory graph
3. A Qdrant instance for the episodic vector store

### Step 1: Install Python dependencies

```bash
pip install -e ".[full]"
```

The `[full]` extra includes everything in `[demo]` plus:
- `transformers`, `torch` — local LLM support
- `openai`, `anthropic` — hosted LLM clients
- `neo4j` — graph database driver
- `qdrant-client` — vector database driver
- `torch-geometric` — for the GNN bridge (with fallback if absent)
- `z3-solver` — for the L4 deductive reasoner

Installation may take several minutes; PyTorch and the transformers library are large.

### Step 2: Start external services

The repository ships with a `docker-compose.yml` that configures local Neo4j and Qdrant instances with sensible defaults:

```bash
docker compose up -d
```

This starts:
- **Neo4j** on `bolt://localhost:7687` (browser at http://localhost:7474)
- **Qdrant** on `http://localhost:6333`

Default credentials are in `docker-compose.yml` and should be changed before any deployment.

To verify the services are running:

```bash
curl -s http://localhost:7474 > /dev/null && echo "Neo4j: OK"
curl -s http://localhost:6333/healthz && echo "Qdrant: OK"
```

### Step 3: Configure the language model

Choose one. The system reads configuration from environment variables.

**Option A: OpenAI**

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o    # or any model the key has access to
```

**Option B: Anthropic**

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-opus-4-7
```

**Option C: Local HuggingFace**

```bash
export LLM_PROVIDER=huggingface
export LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export LLM_DEVICE=cuda    # or cpu, or mps
```

Local models require sufficient RAM/VRAM for the chosen size. The 8B class works on a 16 GB GPU; larger models need more.

**Option D: Custom**

Implement the `LLMClient` protocol in `src/core/contracts.py` and register it via configuration. Useful for self-hosted deployments behind APIs other than the above.

### Step 4: Initialize the system

```bash
python -m src.bootstrap
```

This:
1. Verifies all services are reachable.
2. Creates the Neo4j schema (constraints, indexes).
3. Creates the Qdrant collection with the configured embedding dimension.
4. Loads any seed knowledge (none by default; configurable).
5. Writes a startup record to the audit log.

### Step 5: Verify

```bash
python -m src.diagnostics
```

This runs a battery of integration checks across all layers and reports which components are operational.

---

## Configuration

Configuration uses environment variables with sensible defaults. The full reference is in `src/core/config.py`. The most commonly adjusted variables:

| Variable                    | Default                | Purpose                            |
| --------------------------- | ---------------------- | ---------------------------------- |
| `LLM_PROVIDER`              | `openai`               | Which LLM backend to use           |
| `LLM_MODEL`                 | `gpt-4o`               | Model identifier within backend    |
| `NEO4J_URI`                 | `bolt://localhost:7687`| Semantic memory backend            |
| `QDRANT_URL`                | `http://localhost:6333`| Episodic memory backend            |
| `WORKING_MEMORY_CAPACITY`   | `50`                   | L2 workspace size                  |
| `CONFIDENCE_THRESHOLD`      | `0.85`                 | L6 minimum for autonomous action   |
| `AUDIT_LOG_PATH`            | `./audit.jsonl`        | Where the audit log is persisted   |
| `ESCALATION_INTERFACE`      | `cli`                  | `cli`, `web`, or `none`            |
| `NOVELTY_THRESHOLD`         | `0.7`                  | Novelty signal firing threshold    |

For production deployments, override these via a `.env` file or your secrets manager rather than exporting them in shell.

---

## Verification

### Demo verification

```bash
python examples/causal_demo.py --seed 100
```

You should see output approximately matching:

```
Generating synthetic structural causal model (n=5000, seed=100)...
  Latent confounder: product_health   P(healthy) = 0.60
  Treatment:         price_increase   P(treated | healthy) = 0.65
                                      P(treated | unhealthy) = 0.43
  Outcome:           churn (true ATE = +3.00 pp by construction)

[1] LLM-style observational answer
    E[churn | price=1] = 10.92%   (n=2,959)
    E[churn | price=0] = 16.76%   (n=2,041)
    Difference         = -5.84 pp
    Recommendation:     KEEP the price increase

[2] Architecture-grounded answer
    Causal graph recovered via PC algorithm: ✓
    Backdoor set:       {product_health}
    Identifiable:       YES

    ATE = +2.51 pp   (95% CI: +0.20, +4.82)
    Recommendation:    ROLL BACK the price increase
```

The numbers should be exactly reproducible at `--seed 100`. If they differ, something is wrong with the install (most commonly a version skew in `numpy` or `dowhy`).

### Full-system verification

```bash
python -m src.diagnostics
```

You should see one line per component, all green. If any are red, see [Troubleshooting](#troubleshooting) below.

### Other examples

```bash
python examples/theory_of_mind.py        # operator-aware framing
python examples/predictive_hierarchy.py  # System 1 → System 2 routing
python examples/alignment_pipeline.py    # six-stage authorization
```

---

## Air-gapped deployment

For environments without external network access, the system can run with locally-hosted services and a locally-served LLM. The configuration is the same as the full-system install with these adjustments:

### LLM

Use `LLM_PROVIDER=huggingface` with a model downloaded ahead of time:

```bash
# On a connected machine:
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir ./models/llama-3-8b-instruct

# Transfer ./models/ to the air-gapped environment, then:
export LLM_PROVIDER=huggingface
export LLM_MODEL=./models/llama-3-8b-instruct
export LLM_DEVICE=cuda
```

### Embeddings

Sentence-transformers also need to be pre-downloaded:

```bash
huggingface-cli download sentence-transformers/all-mpnet-base-v2 \
    --local-dir ./models/embeddings/mpnet
export EMBEDDING_MODEL=./models/embeddings/mpnet
```

### Python packages

Build a wheelhouse on a connected machine:

```bash
pip download -d ./wheels -e ".[full]"
```

Transfer `./wheels/` to the air-gapped environment, then:

```bash
pip install --no-index --find-links ./wheels -e ".[full]"
```

### Services

Neo4j and Qdrant Docker images can be saved and transferred:

```bash
# On a connected machine:
docker pull neo4j:5
docker pull qdrant/qdrant:latest
docker save neo4j:5 qdrant/qdrant:latest -o services.tar

# In the air-gapped environment:
docker load -i services.tar
docker compose up -d
```

### Audit log

In air-gapped deployments, the audit log is the primary record of system behavior. Configure it to write to durable storage:

```bash
export AUDIT_LOG_PATH=/var/log/cogarch/audit.jsonl
```

The log format is line-delimited JSON; each line is a self-contained event with a SHA-256 hash linking it to the previous line. Standard log-rotation tools work, but rotation must preserve the chain — see `src/infra/audit.py` for the rotation utility.

---

## Troubleshooting

### "Module 'src' not found"

You ran the script from outside the repository root, or the package is not installed in editable mode. Run from the repo root, and verify:

```bash
pip show [package-name]   # should show "Location: /path/to/repo"
```

### "Connection refused: bolt://localhost:7687"

Neo4j is not running. Check `docker compose ps`. If Neo4j shows as `unhealthy`, check its logs with `docker compose logs neo4j`. The most common cause is insufficient memory — Neo4j needs at least 2 GB of free RAM.

### "Demo numbers don't match exactly"

The synthetic SCM is deterministic at `--seed 100` if and only if `numpy>=1.26` and `dowhy>=0.11` are installed. Other versions may produce slightly different numbers due to changes in the underlying random-number generators or estimator implementations. Pin the versions:

```bash
pip install numpy==1.26.4 dowhy==0.11.1
```

### "GNN bridge falling back to heuristics"

PyTorch Geometric is not installed or not importable. This is a soft fallback, not an error — the system still works, just without learned graph attention. If you want the full GAT implementation:

```bash
pip install torch-geometric
```

PyTorch Geometric has notoriously finicky installation; consult their docs for your specific Torch/CUDA combination.

### LLM rate limits

The default L4 orchestrator can issue several LLM calls per reasoning request. For hosted APIs, this can hit rate limits quickly during testing. Lower the rate via:

```bash
export LLM_MAX_REQUESTS_PER_MINUTE=20
```

Or run the local-LLM configuration for development.

### "Audit log chain verification failed"

The audit log has been tampered with, truncated, or corrupted. This is intentionally a hard error — the chain is the safety case. Recovery requires manual review of the log to identify the inconsistent entry, then either accepting the divergence (with explicit annotation) or restoring from a backup.

### "AuthorizationBlocked: outside competence boundary"

The L6 alignment pipeline blocked an action because the system flagged the query as outside its validated competence. This is the system working correctly. Either: (a) the query is genuinely outside the system's competence and should be escalated to a human, (b) the competence boundary needs to be expanded — see `src/layer6_alignment/metacognition.py` for how competence is defined and updated.

---

## Development setup

For contributors:

```bash
pip install -e ".[full,dev]"
pre-commit install
pytest tests/
```

The `[dev]` extra adds:
- `pytest`, `pytest-cov` — test runners
- `mypy` — static type checking
- `ruff` — linting and formatting
- `pre-commit` — pre-commit hooks

The CI configuration in `.github/workflows/` runs the same checks. If the local checks pass, CI should pass.

---

For runtime concerns and operational best practices not covered here, see the operations documentation in `docs/OPERATIONS.md` (planned).
