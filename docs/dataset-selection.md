# Which datasets this project trains on, and why the other two were rejected

**Decision (2026-10-01): the detector trains on CICIoT2023 alone.** CICIoMT2024 and
Edge-IIoTset were both downloaded, characterised, trained on, and evaluated — and both were
rejected on measured grounds. They remain in the registry as reproducible ablations.

This document exists because "we used one dataset" invites the obvious question, and the
answer is not "we ran out of time". We ran the experiments. Adding either corpus made the
system worse or forced an architecture split, and this records the numbers that show it.

---

## The three candidates

| Corpus | What it is | Rows | Schema |
| --- | --- | --- | --- |
| **CICIoT2023** | UNB's general IoT testbed, 105 devices | 46.7M raw | 39 columns from UNB's DPKT packet-window extractor |
| **CICIoMT2024** | UNB's **medical** IoT testbed (WiFi/MQTT part) | 924,291 | The **same** extractor — a drop-in |
| **Edge-IIoTset** | Agricultural/industrial testbed: soil-moisture, temperature, pH, water-level sensors over MQTT, plus Modbus TCP | 2,219,201 | **Different** — one row per *packet*, 63 Wireshark field columns |

All three went through the identical discipline: `asa characterize` → dedup **before** split →
the R3 leakage gate (`tests/test_leakage.py`) → four-stage feature selection. No corpus was
judged on reputation.

---

## The headline numbers

Every figure below is macro-F1 on the **CICIoT2023 test set**, 3 seeds, at the same budget
(K = 3 farms, R = 20 rounds, E = 3 local epochs, W = 16, 400k sequences per client). Source:
`artifacts/manifest_phase9_generalised_percorpus_mixed.json`.

| Training configuration | macro-F1 | vs. best |
| --- | --- | --- |
| **CICIoT2023 alone, federated** | **0.8443 ± 0.0003** | — |
| CICIoT2023 + CICIoMT2024, mixed farms | 0.8211 ± 0.0153 | **−0.023** |
| CICIoT2023 + CICIoMT2024, one corpus per farm | 0.7742 ± 0.0119 | **−0.070** |

**Adding a second corpus never helped.** It cost 0.023 in the best layout and 0.070 in the
worst.

---

## Why CICIoMT2024 was rejected

It is the *easy* case technically — same extractor, same 39 columns, a genuine drop-in. It
still failed on substance.

**1. It supplies no attack family CICIoT2023 lacks.** CICIoMT2024 carries 6 of the 8 families;
**Mirai and BruteForce are absent entirely**. CICIoT2023 carries all 8, across 34 leaf classes.
So the second corpus adds volume, not coverage.

**2. It is a medical testbed, and the deployment domain is agriculture.** The project's design
principle is that the federated client structure models the deployment domain: a detector for a
hospital federates over hospital sites, one for agriculture over farms. You do not put a
hospital and a farm in the same federation. Extra corpora may widen *attack coverage* — they do
not get to define clients — and CICIoMT2024 widens nothing.

**3. It measurably degrades the model.** 0.8443 → 0.8211 when mixed into every farm. When
allocated as its own farm, 0.7742 — because that farm holds 6/8 families and FedAvg averages
its weights into the global model every round.

**Kept as:** the labelled cross-domain ablation. "Here is what federating across two domains
costs" is a real result, and the corpus-per-client configuration is the only place this project
reports it.

---

## Why Edge-IIoTset was rejected

This is the harder and more interesting case, because **the corpus is good** and we proved it.

### First attempt: forcing it into the shared feature language — 0.12

Edge-IIoTset is packet-granularity, so its rows were aggregated into windows and mapped onto
CICIoT2023's 16 selected features. Result: **macro-F1 0.12, FPR 0.93 on its own test split.**
It was also the only corpus that could not supply `Header_Length`, `IAT` and `Time_To_Live` —
three of Phase 2's top-ranked features — forcing them out of the candidate set for everyone.

It was registered `not_for_training` on that basis.

### Second attempt: training on its own native columns — 0.9822

A later re-read showed the 0.12 measured *our reconstruction*, not the corpus: the raw file has
815 exact duplicates in 2,219,201 records (0.04 %). So we trained the same GRU on Edge-IIoTset's
**own** 26 behavioural columns — 8 MQTT fields, 2 Modbus TCP fields, plus TCP/DNS/ARP/ICMP/HTTP
flags, lengths and timings.

**Result: macro-F1 0.9822 ± 0.0043, FPR 0.0017, over 12 classes** (3 seeds). Source:
`artifacts/manifest_agri_native.json`; full write-up in
[`results/agri_native_edge_iiotset.md`](../results/agri_native_edge_iiotset.md).

The corpus is fine. Our translation was the problem. **And it was still rejected**, for two
reasons.

### Reason 1: it cannot share a model

Its value lives in `mqtt.*` and `mbtcp.*` columns that **do not exist** in CICIoT2023's
extractor output. FedAvg requires every client's model to have an identical input width, so a
26-input GRU cannot be averaged with a 16-input one.

The obvious workaround — a 42-column union with zeros where a corpus has no value — does not
work either. The model immediately learns the zero pattern as corpus identity and silently
becomes two sub-models inside one, producing inflated in-distribution numbers. That is worse
than an honest split, because the inflation is invisible.

### Reason 2: its unique contribution is one attack type

Set against CICIoT2023's 34 leaf classes, almost everything Edge-IIoTset offers is already
present:

| Edge-IIoTset class | Already in CICIoT2023 as |
| --- | --- |
| Backdoor | `Backdoor_Malware` |
| MITM | `MITM-ArpSpoofing` |
| SQL_injection, XSS, Uploading | `SqlInjection`, `XSS`, `Uploading_Attack` |
| Password | `DictionaryBruteForce` |
| Port_Scanning, Vulnerability_scanner, Fingerprinting | `Recon-*`, `VulnerabilityScan` |
| DDoS_UDP / ICMP / TCP / HTTP | `DDoS-*` |
| **Ransomware** | **nothing — genuinely new** |

**Only Ransomware.** Splitting the model architecture to gain one attack family is not a trade
worth making.

### A further finding, recorded because it is not obvious

Three of Edge-IIoTset's own classes are **not expressible at all** once identifier columns are
removed, as leakage discipline requires. Measured distinct-window counts:

- `DDoS_UDP`: **1**
- `DDoS_ICMP`: **38**
- `Fingerprinting`: 347

Verified directly on raw strings: **every one of the 26 features is literally zero for every
UDP and ICMP flood packet.** Those two classes are not separable even from each other. Their
only distinguishing content lived in `udp.stream`, `icmp.checksum` and `icmp.seq_le` — transport
nonces that must not be features. The `frame.time` column cannot rescue them either: the corpus
build stripped month and day, and DDoS_UDP timestamps do not parse.

**Kept as:** a documented finding that the native-schema approach works, and a candidate for
proper inclusion later (see below).

---

## Why there is no fourth dataset

The only serious candidate in CICIoT2023's feature family was **CIC IoT-DIAD 2024**. It was
checked: its packet-based set uses the IoTDevID per-packet schema and its flow set uses
CICFlowMeter, so **none of the 39 canonical columns appears**. There is no drop-in fourth
corpus.

More importantly, **more public data does not address the actual weakness.** Our own
leave-one-dataset-out results:

| Trained on | Tested on | macro-F1 |
| --- | --- | --- |
| CICIoMT2024 | CICIoT2023 (unseen testbed) | **0.0853 ± 0.0288** |
| CICIoT2023 | CICIoMT2024 (unseen testbed) | **0.0763 ± 0.0251** |

A model trained on one testbed does not transfer to another. Adding a fourth corpus yields one
more testbed the model also fails to transfer to — which is exactly what adding the second one
demonstrated. The fix is capture on the *target* network, not more public corpora.

---

## What this decision costs, stated plainly

- **No ransomware coverage.** CICIoT2023 has no ransomware class. This is the one real loss.
- **No MQTT- or Modbus-level features.** The detector operates on IP/TCP/UDP-layer features, so
  attacks visible only in MQTT or Modbus payload semantics are outside its view.
- **Cross-testbed transfer is unproven and measured poor** (above). The deployment path is for
  each farm to capture its own traffic — including controlled attacks via the ESP32 scenario
  switch — and federate on that. This is where federated learning earns its place, and the gain
  is already measured: every farm scored better federated than alone (+0.1338 / +0.0789 /
  +0.0516 across the three farms, every seed).

## What would reverse this decision

One thing: **a common feature extractor run over both corpora's raw pcaps.** That would give
Edge-IIoTset the canonical 16 features natively, and it would then join a single model exactly
as CICIoMT2024 does — bringing ransomware and the MQTT/Modbus sensing surface without splitting
the architecture. It requires the raw pcap distribution, which we do not currently hold.

Short of that, one corpus, one feature language, one model.

---

## Sources

| Claim | File |
| --- | --- |
| Corpus characterisations | `artifacts/phase1_characterization_report_*.json` |
| Two-corpus federated results, LODO | `artifacts/manifest_phase9_generalised_percorpus_mixed.json` |
| Per-farm federation gain | `artifacts/manifest_phase9_federation_gain_mixed.json` |
| Edge-IIoTset native-schema result | `artifacts/manifest_agri_native.json` |
| Registry and exclusion reasons | `src/ascon_smart_agri/data/datasets.py` |
| Full write-ups | `results/` |
