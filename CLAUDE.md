# CLAUDE.md — operating guide for AI agents on this repo

This file is instructions for any AI coding agent (Claude Code and others) working in this
repository. Read it before making changes. It is operational: what the rules are, what to run,
and what will bite you. For the architecture narrative see [README.md](README.md); for the
authoritative specification see [docs/design_paper.md](docs/design_paper.md).

Project: **Federated GRU intrusion detection with Ascon-authenticated telemetry for
smart-agriculture IoT**, on the CICIoT2023 benchmark. The repo is currently **scaffolding** —
every module under `src/` is a typed stub tagged with the phase in which its logic lands.

## Golden rules

1. **The Phase 8 architecture overrides the design paper.** On 2026-09-18 the user redirected
   the project (see [docs/plans/phase8-hardware-federation.md](docs/plans/phase8-hardware-federation.md)):
   Ascon-AEAD128 protects the **model weights exchanged between each local GRU and the master
   GRU, in both directions**; the local GRU takes the benign/malicious decision with the weights
   it receives back; the demo runs on two Raspberry Pis + a laptop + six ESP32s. Where
   [docs/design_paper.md](docs/design_paper.md) disagrees with that architecture, **the
   architecture wins and the paper is amended to match** — do not block on paper conformance.
   For everything Phases 1–7 already settled (leakage control, Ascon variant, path
   disjointness), the paper still describes what was built; flag, don't silently change.
2. **Respect the seven-phase gating discipline (III-J4).** Do not skip ahead. Each phase
   begins only after the previous one's exit criterion is met. **Phase 3 (centralised GRU
   proven) is a hard gate before any federation work.** Stop and ask the user before starting
   the substantive logic of a new phase.
3. **Never hand-roll cryptography (III-G1).** Ascon-AEAD128 must come from a maintained library
   or the official reference implementation, gated behind the NIST SP 800-232 known-answer
   test. No primitive from scratch, no placeholder encryption at any stage.
4. **No remote CI.** Quality gates run locally as **pre-commit hooks**, each as its own hook.
   Do not add a `.github/workflows` pipeline unless the user asks.
5. **Don't pad demos with out-of-scope features.** If tempted to add one, stub it with a
   comment citing this constraint instead (see "Out of scope").
6. **Keep [CHANGELOG.md](CHANGELOG.md) current.** After every change to this repository
   (implementing or editing any module, adding/modifying tests, changing configs or gates,
   or advancing a phase), update `CHANGELOG.md` in the same piece of work: move completed
   items out of "Unreleased" if appropriate, update the phase-status table, and add a dated
   entry describing what changed and why. Do this before considering the change finished —
   don't defer it to a later commit.

## The seven phases (do them in order)

1. Characterisation report (real columns/types, nulls, zero-variance, exact duplicate count,
   label vocab + counts, correlation matrix). Nothing downstream cites a figure this didn't
   produce.
2. Leakage-controlled preprocessing + four-stage feature selection.
3. Centralised GRU + full evaluation. **HARD GATE.**
4. Three-client federated simulation with weighted FedAvg.
5. Telemetry simulation + feature-provenance adapter.
6. Ascon integration + alerting path.
7. End-to-end integration.
8. Ascon-protected bidirectional weight exchange on physical hardware (Pi clients, laptop
   aggregator, ESP32 sensors) — plan in `docs/plans/phase8-hardware-federation.md`.
   **Steps 1–2 done** (sealed channel + networked nodes over loopback); **software twin
   done** (`docs/plans/phase8-software-twin.md`: MQTT sensor contract, Pi runtime, TLS cloud
   receiver, ESP32/Wokwi sketch, Docker Compose topology); 3–5 need the hardware itself.
9. Multi-dataset generalisation (CICIoT2023 + CICIoMT2024 + Edge-IIoTset; leave-one-dataset-out
   is the headline) — plan in `docs/plans/phase9-multi-dataset.md`. Scaffolding done, data pending.

## Non-negotiable invariants (each has a test that must stay green)

- **Dedup BEFORE split** (leakage gate R3): no record hash may appear in both train and test.
  `tests/test_leakage.py`.
- **Path disjointness (G1):** the malicious-path handler (`routing/alert_sink.py`) holds **no
  reference** to the cloud transport — structurally, not by convention. Never give `AlertSink`
  a `CloudTransport`. Zero encrypted payloads may be emitted on a malicious-only run.
  `tests/test_path_disjointness.py`.
- **Ascon KAT conformance (R5)** gates all crypto integration. `tests/test_ascon_kat.py`.
  Decryption returns `⊥`/`None` on any modified ciphertext or associated data
  (`tests/test_ascon_tamper.py`). AD layout = `⟨edge_id, device_id, counter, schema_version⟩`
  (Eq. 27). Fresh CSPRNG nonce per message; zero nonce reuse per key
  (`tests/test_nonce_collision.py`).
- **FedAvg weight `n_k` = number of training SEQUENCES, not raw rows** (Eq. 21). Common bug.
  `tests/test_fedavg_weighting.py`.
- **Federated scaler == pooled scaler** via Chan's parallel formula (Eqs. 23–24), without
  pooling raw data. `tests/test_scaler_equivalence.py`.
- **Parameters serialized with safetensors, never pickle** (`federated/serialization.py`).
- **Weights cross the network only Ascon-sealed (Phase 8):** every client update and every
  global-state download is an AEAD frame under a per-client, per-direction key with AD
  `⟨client_id, round, direction, schema_version⟩`; the aggregator never applies a frame that
  opens to `⊥`. `tests/test_weight_channel.py`, `tests/test_node_loopback.py` (networked
  FedAvg == in-process FedAvg bit-for-bit).
- **The sensor node is a contract, and the scenario switch never synthesises (Phase 8):**
  `farm/<farm>/sensor/<device>` JSON with a monotonic `seq`; `farm/<farm>/scenario` only
  selects which *held-out* record pool the provenance adapter draws from (G6 unchanged).
  `tests/test_mqtt_source.py`.
- **TLS benign path keeps G1 (Phase 8):** `routing/tls_path.py`'s router is the only holder
  of the cloud transport; the alert sink never sees one; a malicious-only run sends nothing.
  `tests/test_tls_path.py`, `tests/test_software_twin.py`.
- **No corpus trains with a placeholder-spelling collision (Phase 9):** a text column whose
  "absent" value is spelled two ways encodes file provenance (Edge-IIoTset's `"0"`/`"0.0"`).
  `characterize` reports it; `datasets.normalise_placeholders` removes it. `tests/test_datasets.py`.
- **Harmonisation never invents a column:** a canonical feature a corpus cannot supply is
  reported missing and selection runs over the intersection. Leave-one-dataset-out is the
  generalisation claim; pooled accuracy is not.
- **LayerNorm, not BatchNorm** (batch stats don't aggregate across federated clients).
- **No PCA** (kills column interpretability the provenance adapter and operators need).
- **No synthetic oversampling** (fabricates temporal structure). Class-weighted CE instead,
  weights from training data only.
- **Default model config reproduces Eq. (19) = 33,800 params** (F=16, H=96, C=8).
  `tests/test_model_param_count.py`.
- **Windows only over contiguous same-label runs; rows never shuffled before windowing;**
  label = final record (`y_i = y_{i+W-1}`).
- **Evaluation never reports accuracy alone** (IR ≈ 5751). Primary: macro-F1, per-class F1,
  balanced accuracy, MCC, confusion matrix, FPR; binary adds PR-AUC. **≥ 3 seeds, mean ± std.**
- **Feature-provenance boundary (G6):** network features come only from held-out CICIoT2023
  records, never from the JSON payload, never synthesised. The runtime demo shows architectural
  correctness only — see the docstring in `telemetry/provenance.py`.
- **Keys never in version control** (III-J3); demo keys labelled as such.

## Out of scope — stub with a comment, never implement

Byzantine-robust aggregation, differential privacy, secure aggregation, gradient-inversion
defences, adversarial robustness, production key management, device attestation, dashboards,
cloud agronomic analytics. Physical hardware is in scope **only** as the Phase 8 demo nodes.
**Datasets are the three registered in `data/datasets.py`** (CICIoT2023, CICIoMT2024,
Edge-IIoTset) — the "CICIoT2023 only" rule was lifted on 2026-09-19; wire up no fourth without
asking, and never train on a corpus whose `asa characterize --dataset` report has not been
reconciled with the registry.

## Commands

**Python 3.12 or newer is required.** This is a floor set by the dependencies, not a
preference: numpy (>= 2.4) and scipy (>= 1.16) both declare `requires-python >= 3.12`, so the
pinned stack cannot install on 3.11 at all. `uv` is not installed; the build backend is
hatchling. Everything — the scientific stack and the dev tools — lives in a project-local
`.venv`; there is no reliance on packages preinstalled in a system Python.

### Setup (once per machine)

The venv layout differs by platform: POSIX puts executables in `.venv/bin/`, Windows puts them
in `.venv\Scripts\` with an `.exe` suffix. Both are shown throughout; run the pair for your
platform.

```bash
# macOS / Linux
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/pre-commit install
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\pre-commit.exe install
```

### Running the gates

```bash
# macOS / Linux
./.venv/bin/ruff check .             # lint
./.venv/bin/ruff format --check .    # format
./.venv/bin/mypy src                 # types (strict-ish; strict on crypto/federated)
./.venv/bin/vulture                  # dead code
./.venv/bin/python -m pytest         # tests (+ coverage)

./.venv/bin/pre-commit run --all-files   # everything above, as the hook runs it
```

```powershell
# Windows (PowerShell)
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe src
.\.venv\Scripts\vulture.exe
.\.venv\Scripts\python.exe -m pytest

.\.venv\Scripts\pre-commit.exe run --all-files
```

All gates must be green before a commit.

### Running a phase

`asa` (installed by `pip install -e .`) runs a phase's driver under `scripts/` and passes any
further arguments straight through, so `asa federate --rounds 2` is
`scripts/run_phase4.py --rounds 2`; `asa <phase> -h` prints that driver's help.

```bash
# macOS / Linux
./.venv/bin/asa characterize                                   # Phase 1 -> artifacts/phase1_*.json
./.venv/bin/asa train-centralized --cache artifacts/phase4_cache.npz   # Phase 3
./.venv/bin/asa federate --cache artifacts/phase4_cache.npz            # Phase 4 (--save-cache builds it)
./.venv/bin/asa run-e2e                                        # Phase 7 (needs the trained checkpoint)

# Phase 8 (one aggregator + K client processes; demo keys are gitignored under keys/)
./.venv/bin/asa aggregator --generate-keys --clients pi-1,pi-2,sim-3
./.venv/bin/asa aggregator --port 7700 --rounds 20
./.venv/bin/asa client-node --client-id pi-1 --client-index 0 --server <laptop-ip>:7700 --sequence-cap 40000

# Phase 8 software twin (whole topology, no hardware; needs Docker) -- or see the plan for the
# no-Docker process recipe. Manifests are tagged platform=docker-arm64-simulation.
docker compose up -d --build   # NOT --abort-on-container-exit: it aborts on init-secrets
docker compose logs -f pi-1 pi-2
docker compose down -v

# Phase 9 (per-corpus characterisation, the first thing to run when a corpus lands)
./.venv/bin/asa characterize --dataset ciciomt2024 --root data/ciciomt2024
./.venv/bin/asa characterize --dataset edge_iiotset --root data/edge_iiotset
```

```powershell
# Windows (PowerShell)
.\.venv\Scripts\asa.exe federate --cache artifacts\phase4_cache.npz
```

Phases 2, 5 and 6 have no driver of their own: Phase 2 runs inside every training driver, and
Phases 5-6 are library modules exercised by `run-e2e` and the tests. `asa preprocess`,
`asa telemetry` and `asa secure` say so and exit 2. The Phase 7 checkpoint comes from
`scripts/train_federated_model.py --cache artifacts/phase4_cache.npz`.

### Two things that will bite you

- **The pytest hook is `language: system`**, so it uses whatever `pytest` is on `PATH` rather
  than an isolated hook environment (it needs torch and the scientific stack, which pre-commit
  would not install for it). Commit with the venv activated — `source .venv/bin/activate` on
  POSIX, `.\.venv\Scripts\Activate.ps1` on Windows — or the hook either fails or silently
  tests against the wrong interpreter. Running `pre-commit run` by the full path above is not
  enough on its own; prefix `PATH` or activate first.
- **Hook revs in `.pre-commit-config.yaml` must track the `dev` extra in `pyproject.toml`.**
  pre-commit builds each hook its own isolated environment at the pinned rev, so a drifted pin
  means the hook and your local `ruff check .` are different programs — one can pass while the
  other fails. `vulture` is pinned with an exact `==` in `pyproject.toml` and must match the
  hook rev exactly.

## Conventions in this codebase

- **Stub pattern:** unimplemented functions `del` their unused params (marks them deliberately
  unused for vulture, keeps the public signature clean) then
  `raise NotImplementedError("Phase N: … not implemented yet.")`. Replace this body when
  implementing the phase.
- **`Array`** (`src/ascon_smart_agri/_types.py`) is the numpy array alias — bare `np.ndarray`
  fails mypy's `disallow_any_generics`. Narrow to a dtype where known.
- **Math-notation names** (`W_z`, `U_r`, `M_2`, …) are allowed so code reads like Section III;
  ruff `N803`/`N806`/`E741` are disabled for that reason.
- **Config is typed pydantic** (`configs/base.py`), never scattered dicts. Sweep values encode
  the six ablations. Everything is serialized into the per-run manifest.
- Reference the paper by section/equation in docstrings (e.g. "Eq. (21)", "Section III-F2").

## Environment notes

- Git root is **this folder** (a dedicated repo), not the user's home directory.
- Line endings are normalised to LF via `.gitattributes` (ruff-format emits LF).
- Do not commit or push unless the user asks.

## Open decision (surfaces at Phase 6)

Paper III-G1 ("no primitive from scratch") vs. an earlier "vendor a reference impl" fallback:
resolution is to pin the `ascon` candidate, gate on the KAT test, and if it fails, bind the
**official reference implementation** rather than hand-rolling. Confirm with the user at Phase 6.
See [docs/plans/phase6-ascon-implementation.md](docs/plans/phase6-ascon-implementation.md) for
the library research (candidates, licenses, KAT vector sources) and a proposed step-by-step
sequence — read it before starting Phase 6 work, but re-verify its factual claims rather than
trusting them as current.
