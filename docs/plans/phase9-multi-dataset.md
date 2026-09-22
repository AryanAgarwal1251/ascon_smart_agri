# Phase 9 plan — multi-dataset generalisation (CICIoT2023 + CICIoMT2024 + Edge-IIoTset)

**Status (2026-09-22): first full run done and read; second design decided and running.**
The 2026-09-21 run over all three corpora (§3 step 1) returned leave-one-dataset-out ≈ 0 and
showed why (per-corpus feature scales, and an Edge-IIoTset that cannot be made to speak the
canonical language). The user's decision on 2026-09-22, now implemented in `scripts/run_phase9.py`
and the registry: **Edge-IIoTset is `not_for_training`** (kept in the registry for
characterisation), the training corpora are **CICIoT2023 + CICIoMT2024** with all 16 Phase 2
features back, standardisation is **per corpus**, K = 3 is allocated across corpora with the
Phase 4 Dirichlet split inside a corpus, a **pooled-centralised twin** on the same windows is
the upper bound, and leave-one-dataset-out over the two corpora is the generalisation claim.
See §5.

On 2026-09-19 the user lifted the "CICIoT2023 only" rule: the detector must be trained across
two to three IoT intrusion corpora so it does not fail on attacks that are in scope but absent
from CICIoT2023. This plan is authoritative for that work.

## 1. The three corpora and why

| Corpus | Role | Integration cost | Where to get it |
| --- | --- | --- | --- |
| CICIoT2023 | reference; its raw columns are the canonical schema | none | already in `data/ciciot2023_raw/` |
| CICIoMT2024 (WiFI_and_MQTT only) | same DPKT extractor; its README feature table (39 rows) **is the canonical vocabulary verbatim**, so all 16 selected features carry over; adds MQTT floods; BLE excluded | low: taxonomy only (spelling absorbed by `canonical_name`) | UNB "Browse files" → `WiFI_and_MQTT/attacks/CSV/{train,test}/*.csv` → `data/ciciomt2024/`. **Not** `profiling/` (benign device-fingerprinting captures, outside the attack benchmark) and not `PCAP/` |
| Edge-IIoTset (`DNN-EdgeIIoT-dataset.csv`) | agriculture sensors over MQTT; 14 attacks incl. injection, MITM, ransomware | medium: per-packet rows, partial schema, known artefact | Kaggle `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot` → `data/edge_iiotset/` |

## 2. What is implemented (`src/ascon_smart_agri/data/`)

- `datasets.py` — `REGISTRY` of `DatasetSpec`s: file discovery glob, label derivation
  (filename or column), column aliases onto `CANONICAL_COLUMNS` (CICIoT2023's 39 raw
  columns), identifier columns to drop, and `harmonise_columns()` which **never invents a
  column** — gaps are reported as `missing_canonical`. `SELECTED_COLUMNS` is Phase 2's F=16.
- `normalise_placeholders()` / `placeholder_collisions()` — the Edge-IIoTset lesson (arXiv
  2608.15761): "0" vs "0.0" for an absent field encodes which file a row came from and scores
  1.0000 accuracy on its own. Every "absent" spelling becomes one missing value; the
  characterisation report's `placeholder_collisions` field proves it is gone.
- `taxonomy.py` — `CICIOMT2024_LEAF_TO_FAMILY` (19 leaves) and `EDGE_IIOTSET_LEAF_TO_FAMILY`
  (15 leaves) into the fixed C=8 families, so every corpus trains the same head. Judgement
  calls are documented in the module: MQTT-Malformed_Data → WebBased; Ransomware → WebBased;
  Fingerprinting → Reconnaissance; Password → BruteForce; MITM → Spoofing.
- `packet_windows.py` — `aggregate_packet_windows()` turns packet rows into CICIoT2023-style
  window rows, never across a label change or a source file; `edge_iiotset_packets()` derives
  the ingredients from Wireshark columns. Edge-IIoTset has no frame length, TTL or header
  length, so `Time_To_Live` and `Header_Length` will be **missing** for it.
- `asa characterize --dataset <name> [--root PATH]` — Phase-1 discipline per corpus.

## 3. Sequence once the files arrive (each step gated)

1. **Characterise** both corpora: `asa characterize --dataset ciciomt2024` and
   `--dataset edge_iiotset`. Reconcile the report's `columns` and `label_counts` with the
   registry's aliases and taxonomy maps; fix `datasets.py` / `taxonomy.py` / `packet_windows.py`
   where the file disagrees with the documentation. `placeholder_collisions` must be non-empty
   for raw Edge-IIoTset (proving the check works) and empty after harmonisation.
   **DONE 2026-09-21** — reports at `artifacts/phase1_characterization_report_{ciciomt2024,
   edge_iiotset}.json`. Findings, all reconciled in code (see the CHANGELOG entry):
   - CICIoMT2024: 8,775,013 rows, 19 labels (all in the taxonomy), 45 columns, 0 text
     columns, 0 collisions; **16/16 selected features** come through `harmonise_columns`.
     `Drate` is zero-variance, as in CICIoT2023.
   - Edge-IIoTset: 2,219,201 rows, 15 labels (all in the taxonomy), 63 columns, 20 text.
     **17 columns collide on `"0"`/`"0.0"`** raw; **0 after `normalise_placeholders`**
     (checked over the full file, not a sample — the file is sorted by class, so a head
     sample is all `Normal`). `udp.port` is the placeholder on every `DDoS_UDP` row and
     `udp.stream` is the real UDP marker. **`frame.time` is the literal `"6.0"` on every
     `DDoS_UDP` and `MITM` row** (122,741 rows): no time base for two classes, so `IAT` and
     `Rate` are dropped for this corpus rather than the classes (user decision). The corpus
     supplies **12/16 selected features**: `Header_Length`, `Time_To_Live`, `Protocol Type`
     and `IAT` are reported missing. `Protocol Type` is now derived (6/17/1 from the TCP,
     UDP and `icmp.checksum` markers; 0 for non-IP) → **13/16**.
   - **Step 4 measured early (2026-09-21), before any federation code:** the intersection is
     18 canonical columns (17 candidates; `Number` excluded as a testbed marker), which
     Stage 2 collapses to F=13. Cost on CICIoT2023, same driver/split/epochs as the Phase 3
     default run: macro-F1 0.7866 → 0.7629 (−0.024, ~1.2 σ), FPR 0.257 → 0.306; BruteForce
     takes most of it. `configs/generalised.yaml`, `artifacts/manifest_phase3_generalised.json`.
   - **Steps 2-6 built and run 2026-09-21** (`scripts/run_phase9.py`, `asa generalise`;
     see the CHANGELOG entry for the table). Leave-one-dataset-out ≈ 0 on every fold with flat
     convergence curves; in-distribution 0.60 / 0.56 / 0.12. The cause is in the cache: the
     same canonical column sits on a different scale per corpus (`ack_count`, `Protocol
     Type`, `Tot sum`), so the global scaler leaves each corpus its own offset and the model
     learns the testbed. **Open decisions (user's):** (a) per-client standardisation or
     rank/quantile features -- a change to III-F4's single global scaler; (b) re-verify
     CICIoMT2024's feature *values* against its README (`Protocol Type` is fractional there,
     `ack_count` never exceeds ~1); (c) the stretch item below, one extractor over raw pcaps,
     which is the only fix that makes the columns mean the same thing.
2. **Harmonise + aggregate**: CICIoMT2024 through `harmonise_columns`; Edge-IIoTset through
   `edge_iiotset_packets` → `aggregate_packet_windows(window_packets=?)` → `harmonise_columns`.
   The window size is an experimental knob (start at 10, the value most often cited for the
   CICIoT2023 extractor) and is recorded in the manifest. The per-column definitions in
   `packet_windows.py` follow the CICIoMT2024 README's feature table (same UNB extractor):
   Header_Length is a *mean*, Protocol Type a *mode*, flag numbers proportions, counts counts.
3. **Per-corpus leakage control**: subsample → dedup → block split → R3 gate, exactly as
   Phase 2, for each corpus separately. Source-file identity is kept so windows never cross.
4. **Feature intersection + re-selection**: the usable feature set is the intersection of
   what every training corpus provides; the four-stage selection is re-run over it. Report
   how many of the original 16 survive.
5. **Federation design — one corpus per farm**: client 1 = CICIoT2023, client 2 =
   CICIoMT2024, client 3 = Edge-IIoTset. Non-IID by construction, replacing the Dirichlet
   partition; the Phase 8 node scripts take a per-client data source instead of a shared
   cache (`cache_partition.py` gains a per-corpus loader).
6. **Evaluation — leave-one-dataset-out is the headline**: train on two corpora, test on the
   third, for each of the three holdouts, ≥ 3 seeds, the full metric set (macro-F1, per-class
   F1, balanced accuracy, MCC, confusion, FPR). Pooled-test accuracy is reported but is not
   the generalisation claim. A corpus aggregated by `packet_windows.py` is never reported
   without its leave-one-out number beside it.
7. `results/phase9_multi_dataset.md`, manifest, CHANGELOG, design-paper amendment.

## 4. Known risks, stated up front

- **Extractor mismatch.** The CICIoT2023 extractor is unpublished; `packet_windows.py`
  implements the documented intent of each column. Edge-IIoTset windows will differ in
  distribution from UNB windows for reasons unrelated to attacks. Leave-one-dataset-out is the
  only honest measure, and a poor number there is a legitimate finding, not something to tune
  away.
- **Testbed shortcut.** Pooling corpora from different testbeds lets a model learn "which
  testbed" instead of "which attack". Step 6 is designed to expose this.
- **CICIoMT2024 column names are confirmed by its README** (2026-09-19); only the CSV
  header *spelling* is unconfirmed, and `canonical_name` absorbs any of the three known
  spellings. The TCP/IP flood labels are mapped under both `TCP_IP-DDoS-ICMP` (filename) and
  `DDoS-ICMP` (README chart) spellings. Edge-IIoTset's column list was verified against the
  real file on 2026-09-21 (step 1 above).
- **Stretch, not now:** re-extracting all three corpora from their raw pcaps with one
  extractor of our own — the principled fix for both dataset mixing and live ESP32 traffic,
  but hundreds of GB of pcap for CICIoT2023 alone.

## 5. Second design (2026-09-22): two corpora, per-corpus scaling, pooled twin

Decided by the user after reading the 2026-09-21 run (the reasoning is in the CHANGELOG entry
of that date and in `data/datasets.py`'s `not_for_training` reason):

1. **Edge-IIoTset out of training.** Its packet-field reconstruction gives near-constant
   windows (99 % TCP, 93 % ACK-flagged), 81 % of them duplicates; trained *on* it the model
   scored 0.12 macro-F1 / 0.93 FPR on its own split; and it alone forced `Header_Length`,
   `IAT` and `Time_To_Live` (Phase 2's top-ranked features) out of the candidate set. It stays
   registered so `asa characterize --dataset edge_iiotset` and the placeholder check keep
   working, and `TRAINING_CORPORA` is what the driver trains on. Naming it on `--corpora`
   still works, with a printed warning, so the decision is reversible and recorded.
2. **All 16 features back.** `configs/generalised.yaml` lists the full 39-column canonical
   vocabulary (CICIoMT2024 supplies all of it); selection re-runs on the pooled, scaled
   training rows of each experiment.
3. **Per-corpus standardisation** (`--scaling per-corpus`, the default). Each corpus's farms
   combine their sufficient statistics (Eqs. 23-24) into that corpus's scaler; a held-out
   corpus is scaled with the label-free statistics of its own training split -- what a new
   farm computes over its own traffic before running the model. This departs from III-F4's
   single global scaler (kept as `--scaling global`, the ablation) and the paper is amended if
   the result is kept.
4. **K = 3 across corpora.** The larger corpus takes two farms (Dirichlet α = 0.5 over its
   blocks, Phase 4's partition), the other one; a single-corpus fold takes three Dirichlet
   farms. Client ids are `corpus:index`; every frame is Ascon-sealed as before.
5. **Pooled-centralised twin.** One GRU on the concatenation of the *same* capped farm windows
   for R × E epochs. Same sequences, no federation: its gap to `in_distribution` is what
   federation costs; its gap to the single-corpus ceilings (below) is what mixing costs.
6. **Leave-one-dataset-out over the two corpora**, each fold also scored on the training
   corpus's own test split -- the single-corpus ceiling at the same budget, for free.
7. **Budget: Phase 4's** (R = 20, E = 3, 400k sequences per farm ≈ Phase 4's per-farm load),
   3 seeds, seed-major so a full picture lands per seed; `artifacts/phase9_generalised_percorpus_progress.json`
   is rewritten after every run and the manifest is `manifest_phase9_generalised_percorpus.json`.

Expectation, stated before the run: in-distribution should recover most of the Phase 4
number on CICIoT2023 (the 13-feature cost was 0.024 and is gone; the budget is matched);
leave-one-dataset-out should be well below that but no longer near zero. A poor LODO number
under per-corpus scaling is the honest cross-testbed generalisation finding.

**Third corpus: checked and closed (2026-09-22).** [CIC IoT-DIAD 2024](https://www.unb.ca/cic/datasets/iot-diad-2024.html)
was the one candidate; its "packet-based" set is the IoTDevID-style per-packet schema
(`stream`, `ttl`, `eth_size`, `payload_entropy`, `stream_{1,5,10,30,60}_{count,mean,var}`,
133 columns) and its flow set is CICFlowMeter (84 columns). Neither is the CICIoT2023 window
extractor: none of the 39 canonical columns appears in either. There is no verified corpus in
CICIoT2023's feature family beyond CICIoMT2024, so the training set is those two; no fourth
schema, and no reconstruction of the canonical columns from another extractor's output --
that is exactly what failed for Edge-IIoTset.
