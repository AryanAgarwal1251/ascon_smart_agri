# Federated GRU Intrusion Detection with Ascon-Authenticated Telemetry for Smart Agriculture IoT

Three simulated edge clients train a GRU-based intrusion detector on **CICIoT2023** using
sample-weighted federated averaging (raw records never leave a client). At runtime, the
current global model classifies a window of network-flow features and produces a verdict:

- **Benign** → the telemetry payload is encrypted and authenticated with **Ascon-AEAD128**
  (NIST SP 800-232) and sent toward a mock cloud receiver.
- **Malicious** → the payload is diverted to an alerting path and, *by construction*, can
  never reach the cloud transport.

The two data paths are **disjoint in the code, not merely by convention** (see
[`routing/`](src/ascon_smart_agri/routing/)).

> This repository is currently **scaffolding only**. Every module is a typed stub tied to the
> phase in which its logic lands. No detection, federation, or cryptography logic is
> implemented yet. See the work plan below.

The authoritative specification is [`docs/design_paper.md`](docs/design_paper.md). Where any
summary here and the paper disagree, **the paper wins** — raise the discrepancy rather than
resolving it silently.

## Architecture

Two planes share a model but nothing else (Section III-A):

- **Training plane** (offline, periodic): clients train locally and exchange only the
  parameter vector `θ_k` and their sequence count `n_k` with an aggregator. Raw records never
  cross the client boundary.
- **Runtime plane** (continuous): telemetry arrives, is classified by the current global
  model, and is routed per Eq. (5). The malicious-path handler holds **no reference** to the
  cloud transport.

```
data → dedup(before split) → block split → feature selection → windowing → GRU
                                                                     │
                                          training plane ◄───────────┤ (FedAvg over K=3 clients)
                                                                     │
runtime plane:  telemetry ─┬─ application payload ──────────► Ascon ─┴─► [benign]  → mock cloud
                           └─ network features (held-out) → GRU verdict ─► [malicious] → alert sink
```

### Declared limitation — runtime feature provenance (Section III-H, gap G6)

The application-layer JSON payload (e.g. `{"deviceId": "...", "temperature": ...,
"soilMoisture": ...}`) and the network-flow feature vector consumed by the GRU are **different
objects with different origins**. The network features come from **held-out CICIoT2023 records
never seen during training** — never from parsing the JSON payload, and never synthesised. A
runtime demonstration therefore establishes **architectural correctness only** (the pipeline
separates the two planes, routes on the verdict, encrypts/verifies correctly, and never places
a malicious-verdict payload on the cloud path). It does **not** show that a CICIoT2023-trained
model would detect attacks against a live agricultural MQTT deployment. That claim would
require capture and feature re-extraction on the target network, and is not made.

### Ascon variant (Section III-G1, risk R5)

We target **SP 800-232 Ascon-AEAD128**, which is *not* the pre-standard Ascon-128/128a of
Ascon v1.2 (revised initial values, little-endian formatting). The widely installed `ascon`
PyPI package implements the v1.2 variants, so it is included only as a **KAT-gated candidate**:
conformance is unverified until [`tests/test_ascon_kat.py`](tests/test_ascon_kat.py) passes
against the official known-answer vectors. If it fails, the paper-consistent fallback is to
bind the **official reference implementation** — the primitive is never hand-rolled.

## Work plan — seven gated phases (Section III-J4)

Each phase begins only after the previous one meets its exit criterion.

| # | Phase | Exit criterion |
|---|-------|----------------|
| 1 | Characterisation report | Reproducible report of real columns/types, nulls, zero-variance, exact duplicate count, label vocab + counts, correlation matrix. Nothing downstream cites a figure this didn't produce. |
| 2 | Leakage-controlled preprocessing + feature selection | Dedup **before** split (leakage gate passes); all transforms fitted on training blocks only; four-stage selector persisted. |
| 3 | Centralised GRU + full evaluation | **HARD GATE.** Single-client detection proven (macro-F1, per-class, confusion, FPR) before any federation begins. |
| 4 | Three-client federated simulation | Weighted FedAvg (`n_k` = sequences); global model loads into every client; per-client class histograms published. |
| 5 | Telemetry simulation + provenance adapter | Two planes separated in code; every feature value's provenance recorded; no value invented. |
| 6 | Ascon integration + alerting path | KAT conformance passes; tamper (ct + AD) rejected; nonce-collision + path-disjointness gates pass. |
| 7 | End-to-end integration | Single-command reproducible run; manifest logged. |

## Repository layout

```
configs/     typed pydantic config schema (base.py) + run values (default.yaml)
docs/        design_paper.md (authoritative spec)
src/ascon_smart_agri/
  data/        characterisation, subsampling, dedup/leakage, block splitting   (III-B)
  features/    four-stage feature selection                                    (III-C)
  sequences/   windowing over contiguous same-label runs                       (III-D)
  model/       GRU detector + centralised training                             (III-E)
  federated/   clients, server, partition, scaler stats, serialization         (III-F)
  telemetry/   MQTT/JSON simulation + feature-provenance adapter               (III-H)
  crypto/      Ascon-AEAD128 wrapper + KAT gate                                 (III-G)
  routing/     verdict router, mock cloud receiver, alert sink (disjoint)      (III-A)
  eval/        metrics, five baselines, reporting, run manifest                (III-I)
tests/       gating tests (leakage, path-disjointness, nonce, Ascon KAT) + sanity tests
```

## Evaluation discipline (Section III-I)

- **Never accuracy alone** (IR ≈ 5751). Primary: macro-F1, per-class F1, balanced accuracy,
  MCC, confusion matrix, and **false-positive rate**; binary projection adds PR-AUC.
- **Five baselines** every configuration: RF (single record), MLP (single record), centralised
  GRU (upper bound), three local-only GRUs (lower bound), federated global GRU.
- **≥ 3 seeds**, reported mean ± std. Single-run numbers are never findings.
- **Stated expectation:** near-ceiling accuracy (98–99 %) on CICIoT2023 is a known dataset
  artifact, not a good model. Reporting directs attention to macro-F1 on rare families and FPR.

## Out of scope

Stubbed with a comment, never implemented to pad a demo: Byzantine-robust aggregation,
differential privacy, secure aggregation, gradient-inversion defences, adversarial-example
robustness, production key management, device attestation, dashboards, cloud agronomic
analytics, physical hardware. Dataset is **CICIoT2023 only**.

## Ethics & keys (Section III-J3)

No PII; CICIoT2023 is machine-generated testbed traffic. No live attack traffic is generated —
every malicious instance is a replayed record. **Key material is excluded from version
control** (see [`.gitignore`](.gitignore)); demonstration keys must be labelled as such.

## Development

Runtime deps (torch CPU, numpy, pandas, scikit-learn, scipy, safetensors, pydantic, pyyaml)
are already present in the target environment. Install the dev tooling and hooks:

```bash
python -m pip install -e ".[dev]"      # add ,crypto once the Ascon backend is chosen
pre-commit install
```

Quality gates run **locally as pre-commit hooks before each commit** (no remote CI): ruff
(lint + format), vulture (dead code), mypy (`src`), and pytest — each as its own hook. Run them
on demand:

```bash
ruff check . && ruff format --check .
mypy src
vulture
pytest
```

### Tooling decisions

- **Build backend: hatchling** — PEP 621-native and installer-agnostic (works with `pip`
  today; `uv` can drive it later as the resolver). `uv` was not installed, so a `uv.lock`-gated
  flow would have blocked local setup.
- **ruff** replaces black/isort/flake8 (`ruff check` + `ruff format`).
- **safetensors** for parameter transport — a structured tensor format, never pickle
  (arbitrary-object deserialization across a transport boundary is a code-execution hazard).
- Configs are **typed pydantic models**, not scattered dicts, and are serialized into the
  per-run manifest.
