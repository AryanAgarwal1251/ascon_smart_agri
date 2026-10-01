# Paper claims inventory — what we can prove, and what we must not say

Built before writing, so that every number in the paper traces to a committed manifest and every
limitation is stated by us rather than found by a reviewer. **Nothing in the paper should assert
anything absent from the first table; nothing should contradict the second.**

The old `docs/design_paper.md` is a starting draft only. The source of truth is the implemented
system and its measured results.

---

## A. Claims we can make, with evidence

### A1. Detector performance

| Claim | Figure | Source |
| --- | --- | --- |
| Centralised GRU, W=16, 3 seeds | **macro-F1 0.8297 ± 0.0013** | `manifest_phase3_complete_default.json` |
| Recurrence earns its place vs MLP | **+0.2322** (0.8297 vs 0.6070), 51× seed std | same |
| vs random forest | 0.8297 vs 0.6855 | same |
| Centralised ceiling, compute-matched | **0.8543 ± 0.0040** | `manifest_phase4_default.json` |
| Federated global, K=3 | **0.8308 ± 0.0150** | same |
| Recovers of the local→centralised gap | **82.4 %** | same |
| Default model parameter count | **33,800** (F=16, H=96, C=8), Eq. (19) | `tests/test_model_param_count.py` |

### A2. Federation is worth it, per client

| Claim | Figure | Source |
| --- | --- | --- |
| Per-client gain, single corpus | +0.1023 / +0.1565 / +0.0721 | `manifest_phase4_default.json` |
| Per-client gain, mixed farms, CICIoT2023 | +0.1338 / +0.0789 / +0.0516 | `manifest_phase9_federation_gain_mixed.json` |
| Per-client gain, mixed farms, CICIoMT2024 | +0.1568 / +0.1547 / +0.1970 | same |
| **Positive for every client, every seed, both corpora** | — | same |
| A client missing classes still benefits | local-only 0.6145 → federated **0.8145** | `manifest_phase4_default.json`, seed 1 |

**A2 is the strongest result in the project.** It is the direct answer to "why federate at all".

### A3. Client-count scalability (K sweep)

| Claim | Figure | Source |
| --- | --- | --- |
| K ∈ {3,5,10,20,30,40,50} × 3 seeds, Phase 4 budget | 21 points | `k_sweep_results.json` |
| Analytical prediction | knee K\*≈48, pigeonhole ceiling 42, comms break-even K≈51 | `k_threshold_analysis.json` |
| Measured: one step K=3→5, then flat to K=40 | 0.8367 → 0.7835 … 0.7794 | `k_sweep_results_progress.json` |
| **Rarest class declines monotonically** | BruteForce/WebBased 0.6415 → 0.4888 | same |
| 0 empty clients at every K up to 40 | dilution, **not** starvation | same |
| Communication scales linearly | 815 KB → 10.9 MB per round (K=3 → 40) | same |
| Federated scaler == pooled scaler at every K | gap ~5×10⁻¹² | same |

> ⏳ Pending: K=50 (3 seeds). Numbers above are provisional until the sweep completes.

### A4. Cryptography

| Claim | Evidence |
| --- | --- |
| Ascon-AEAD128 passes NIST SP 800-232 known-answer tests | `tests/test_ascon_kat.py` |
| Decryption returns ⊥ on any modified ciphertext or AD | `tests/test_ascon_tamper.py` |
| Zero nonce reuse per key; fresh CSPRNG nonce per message | `tests/test_nonce_collision.py` |
| AD layout ⟨edge_id, device_id, counter, schema_version⟩, Eq. (27) | `tests/test_ascon_ad_encoding.py` |
| Weight-channel AD ⟨client_id, round, direction, schema_version⟩ | `tests/test_weight_channel.py` |
| Never hand-rolled: vendored reference implementation, gated on KAT | `crypto/_vendor/pyascon/` |

### A5. The sealed weight channel (the architecture's security claim)

| Claim | Figure | Source |
| --- | --- | --- |
| Every client update and global download is an AEAD frame, both directions | — | `manifest_phase8_aggregator.json` |
| 20 rounds × 3 clients × 2 directions | **120 sealed frames, 0 rejected** | same |
| Bandwidth per client per round | **271,883 B** both ways (135,942 up + 135,943 down) | `manifest_phase8_client_pi-*.json` |
| Reconciles with Phase 4's independent measurement | 815,136 B per round = 3 clients × 2 directions | `manifest_phase4_default.json` |
| Networked FedAvg == in-process FedAvg bit-for-bit | — | `tests/test_node_loopback.py` |

### A6. G1 path disjointness

| Claim | Figure | Source |
| --- | --- | --- |
| Enforced structurally: `AlertSink.__init__` takes no dependencies | — | `tests/test_path_disjointness.py` |
| Malicious-only run emits nothing, and nothing is even *sent* | `received_count == 0` **and** `rejected_count == 0` | same |
| Under load: malicious readings alerted locally | **4,239 → 4,239 alerts** | `manifest_phase8_runtime_pi-*.json` |
| **Malicious readings reaching the cloud** | **0** | same |
| Benign delivered, cross-checked by the receiver's own count | 8,924 sent = 8,924 accepted, 0 rejected | `manifest_phase8_cloud_receiver.json` |

### A7. Leakage control and reproducibility

| Claim | Evidence |
| --- | --- |
| Dedup **before** split; no record hash in both train and test | `tests/test_leakage.py`, r3_overlap = 0 in every manifest |
| Windows only over contiguous same-label runs; rows never shuffled | `tests/test_windowing.py` |
| FedAvg weight n_k = training **sequences**, not raw rows (Eq. 21) | `tests/test_fedavg_weighting.py` |
| Federated scaler == pooled, via Chan's parallel formula (Eqs. 23–24) | `tests/test_scaler_equivalence.py` |
| safetensors, never pickle | `federated/serialization.py` |
| LayerNorm not BatchNorm; no PCA; no synthetic oversampling | `model/gru.py`, config |
| ≥3 seeds, mean ± std, never accuracy alone, FPR first-class | every manifest |
| Full suite | **438 passed, 0 skipped** |

### A8. Dataset selection, as a measured decision

| Claim | Figure | Source |
| --- | --- | --- |
| ~~Adding CICIoMT2024 costs accuracy~~ | ~~0.8443 → 0.8211~~ | **STRUCK — contaminated by the same feature-semantics bug.** The corpus is still excluded, on domain coherence and on supplying no missing attack family. |
| It supplies no family CICIoT2023 lacks | 6 of 8; no Mirai, no BruteForce | same |
| Edge-IIoTset in the canonical language | macro-F1 **0.12** | `manifest_phase9_generalised.json` |
| Edge-IIoTset on its **native** 26-column schema | macro-F1 **0.9822 ± 0.0043** | `manifest_agri_native.json` |
| Its only novel attack type vs CICIoT2023's 34 leaves | **Ransomware alone** | `data/taxonomy.py` |
| Three of its classes are inexpressible on leakage-safe features | DDoS_UDP 1 distinct window, DDoS_ICMP 38 | `manifest_agri_native.json` |

### A9. Negative results — report, do not hide

| Result | Figure | Source |
| --- | --- | --- |
| ~~Leave-one-dataset-out fails~~ | ~~0.0853 / 0.0763~~ | **STRUCK 2026-10-01 — contaminated.** Three of the 16 features (`Variance`, `Header_Length`, `IAT`) do not denote the same quantity in the two corpora, so these numbers measure a broken alias mapping, not generalisation. See `docs/generalisation-plan.md`. |
| A G1 run that passed vacuously | 1 malicious reading; a 10 %-leak system would pass 90 % of the time | `docs/g1-path-disjointness.md` |
| The analytical K model mispredicted the mechanism | predicted a knee from class starvation; measured gradual dilution with 0 empty clients | `k_threshold_analysis.json` vs sweep |

**A9 is not a weakness section — it is a contribution**, with one entry withdrawn. The vacuous-G1
finding is a reusable methodological point about verifying safety properties, and the
analytical-vs-measured K discrepancy is a real result. The LODO entry is struck because the
measurement was contaminated; a corrected LODO number is pending the work in
`docs/generalisation-plan.md` §3, and only that number may be called a generalisation result.

### A10. Pending — planned, with the measurement that will fill it in

| Claim | Will come from |
| --- | --- |
| Validated on two Raspberry Pis + six ESP32s | the hardware run; per-round `round_seconds` are the real feasibility numbers |
| Real Pi timing and energy feasibility | same |
| Corrected leave-one-dataset-out | re-run after the alias verification of `generalisation-plan.md` §3 |

---

## B. Claims we must NOT make

| Do not claim | Why | Say instead |
| --- | --- | --- |
| "Detects attacks on a live agricultural deployment" | The runtime classifies **held-out CICIoT2023 records** paired to each message (G6). No live feature extractor exists. | "demonstrates architectural correctness; detection on live farm traffic requires capture and feature re-extraction on the target network" |
| "The model generalises across IoT testbeds" | LODO = 0.0853 / 0.0763 | "cross-testbed transfer was measured and **failed**; per-farm capture and fine-tuning is the deployment path" |
| "Ascon protects the telemetry end-to-end" | Ascon protects the **weight channel**. Pi→cloud is plain TLS, by design. | "Ascon-AEAD128 secures the bidirectional weight exchange; the cloud hop uses TLS — one cipher per hop" |
| Any latency/throughput number as a Pi figure **before the hardware run** | Container timings are not Pi timings; manifests are tagged `platform=docker-arm64-simulation` | Report Pi timings from the hardware run, which is planned and is where they come from (see §E) |
| "K=3 is the optimal client count" | It is a budget constraint, and K=2 federation is degenerate. | "K=3 is the hardware configuration; the sweep measures behaviour to K=50" |
| "Edge-IIoTset's 0.9822 shows generalisation" | 434,553 rows hold only **1,791 distinct packets**; train/test share the packet vocabulary and differ only in ordering. | "within-corpus class separation; not demonstrated generalisation" |
| "Three datasets were used" | Two were measured and **rejected**. | "three were characterised and evaluated; one was selected, and §X gives the measurements behind rejecting the other two" |
| "No ransomware blind spot" | CICIoT2023 has no ransomware class. | State it as a limitation |

---

## C. Figures the paper needs

| # | Figure | Data ready? |
| --- | --- | --- |
| 1 | System architecture (ESP32 → Pi → aggregator/cloud, crypto per hop) | ✅ `docs/architecture-diagram.md` |
| 2 | Two planes + G6 boundary | ✅ same |
| 3 | Confusion matrix, centralised and federated | ✅ in manifests |
| 4 | Per-class F1, centralised vs federated | ✅ |
| 5 | Federation bracket: local-only / federated / centralised | ✅ |
| 6 | Per-client gain bar chart | ✅ |
| 7 | **K sweep: macro-F1 and rarest-class F1 vs K** | ⏳ K=50 pending |
| 8 | **Communication cost vs K**, with break-even | ⏳ same |
| 9 | Convergence curves (per_round_macro_f1) at several K | ✅ recorded |
| 10 | Ascon overhead / bytes per round | ✅ |

---

## D. Structure (IEEE conference, two-column)

```
I    Introduction
II   Related work
III  Threat model and system architecture
IV   Methodology
     A  Corpus, leakage control (dedup-before-split, R3)
     B  Four-stage feature selection -> F = 16
     C  GRU detector, windowing
     D  Federated learning: weighted FedAvg, federated scaler
     E  Ascon-AEAD128 sealed weight channel
     F  Path disjointness (G1)
V    Experimental setup
VI   Results
     A  Centralised baselines, recurrence ablation
     B  Federated performance, per-client gain       <- headline
     C  Client-count scalability (K sweep)
     D  Cryptographic and communication overhead
     E  Multi-dataset study (negative result)
VII  Twin validation and G1 under load
VIII Limitations and threats to validity           <- the whole of §B, stated by us
IX   Conclusion and future work (live extractor, hardware)
```

**Pending before drafting §VI-C:** the K sweep's final 3 points (K=50).
**Pending before drafting §VI-E:** the feature-semantics fix and a corrected LODO re-run.
**Pending before §VII can claim hardware:** the hardware run itself.
**Needed from the user:** target venue and page limit, which decide how much of §VI survives.
