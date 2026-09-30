# Phase 8 plan — Ascon-protected weight exchange on physical edge hardware

**Status (2026-09-19): Steps 1–2 implemented and green; Steps 3–5 need the hardware.**
Implemented: `crypto/ascon_aead.py::WeightAssociatedData`, `federated/transport.py` (sealed
frames, per-direction keys, labelled demo key files), `federated/node.py` (`AggregatorNode`,
`ClientNode`, TCP framing, sealed scaler round 0), `federated/cache_partition.py`,
`scripts/run_aggregator.py`, `scripts/run_client_node.py`, `asa aggregator` / `asa client-node`;
tests `test_weight_channel.py` (22) and `test_node_loopback.py` (2, incl. a tampering proxy).
A four-process loopback run on the real Phase 4 cache completed round 0 + 2 weight rounds with
135,917 B per sealed frame and zero rejections.

**This document, not the design paper, is authoritative for Phase 8.** On 2026-09-18 the user
redirected the project: where [docs/design_paper.md](../design_paper.md) disagrees with the
architecture below, the architecture wins and the paper is amended to match. Do not block on
paper conformance and do not re-raise "the paper is authoritative" for anything in this file.

## 1. The architecture (the user's specification)

- A farm field has sensors (three ESP32 nodes per farm). Their readings go to a **local GRU**
  running on that farm's Raspberry Pi. Raw readings never leave the farm.
- Every local GRU is a client of one **master (federated) GRU** — the aggregator on a laptop.
  Through weighted FedAvg the *i*-th farm's model absorbs what the other *K−1* farms have
  seen; there is no separate "attack information" channel, the weights *are* that channel.
- **The weight exchange is Ascon-AEAD128 encrypted in both directions**: local → master
  (the client update) and master → local (the aggregated global state). This is the path
  that crosses the untrusted IoT network, and it is what Ascon protects in this project.
- **The benign/malicious decision is taken by the local GRU** using the weights it received
  back from the master. The master never classifies traffic.
- After deciding, the local GRU is the only party that talks to the cloud: benign → cloud,
  malicious → local alert path (the existing G1 disjointness).
- **Decided 2026-09-19:** the Pi → cloud hop uses **plain TLS** (HTTPS or MQTT-over-TLS),
  not Ascon. Ascon is the project's security claim on the *weight channel only*; layering it
  under TLS on the cloud hop would be redundant transport encryption. The cloud receives the
  benign sensor reading, not a verdict. Ascon-signed records on this hop are explicitly
  **not** built (they would only matter for per-record provenance past TLS termination).

## 2. Node roles and wiring

```
 FARM 1 (Pi #1 = client k=1)                  FARM 2 (Pi #2 = client k=2)
 ESP32 S1,S2,S3 --MQTT--> Pi#1 local broker    ESP32 S4,S5,S6 --MQTT--> Pi#2 local broker
   provenance adapter -> windows (W=16)          (identical code path)
   -> local GRU (current global weights)
   -> Eq.(5) verdict
        benign    -> TLS (HTTPS/MQTTS) ---------------> CLOUD RECEIVER  (laptop, process B)
        malicious -> AlertSink (holds no cloud reference)

 TRAINING PLANE (periodic; separate process, separate keys):
   Pi#k  --Ascon(k_up,k)  [safetensors(state_k, n_k)]--> AGGREGATOR   (laptop, process A)
   Pi#k  <-Ascon(k_down,k)[safetensors(global_state)]--  weighted FedAvg, Eq.(21)
   laptop-simulated client k=3 over loopback, same code path, so K=3 as in Phase 4
```

| Node | Role | Software |
| --- | --- | --- |
| Laptop | Aggregator (process A) + cloud receiver (process B) + simulated client 3 | this repo, `.venv` |
| Raspberry Pi ×2 | Local GRU client + local MQTT broker + runtime pipeline | this repo on aarch64, Mosquitto |
| ESP32 ×6 | Sensor node: publishes JSON readings; can be switched into an attack scenario | Arduino/ESP-IDF sketch, PubSubClient |

Aggregator and cloud receiver are two processes with disjoint key sets. They share nothing
but the model, which keeps the two-plane separation already tested in Phase 7.

## 3. Cryptographic design of the weight channel

- **Cipher:** the existing KAT-gated `crypto/ascon_aead.py` (`AsconAEAD128`, `NonceRegistry`).
  Nothing new is implemented at the primitive level (golden rule 3).
- **Plaintext:** the existing safetensors blob from `federated/serialization.py`, which
  already carries `n_k` as metadata. Because the whole blob is under the tag, **`n_k` is
  authenticated** — an on-path attacker cannot skew the Eq. (21) weighting or substitute
  weights. (This does not defend against a *malicious client*; poisoning stays out of scope.)
- **Keys per client k:** `k_up,k` (client encrypts, aggregator decrypts) and `k_down,k`
  (aggregator encrypts, client decrypts). Separate up/down keys mean every key is only ever
  *encrypted under* by one process, so the per-process `NonceRegistry` and
  `test_nonce_collision.py` guarantee "zero nonce reuse per key" structurally. There is no telemetry key any more
  (TLS carries the cloud hop). All keys are pre-shared demo keys, labelled as such, loaded from a file
  outside version control (III-J3 still applies).
- **Associated data (new layout, alongside Eq. 27):**
  `⟨client_id, round, direction, schema_version⟩`, `direction ∈ {"up","down"}`, serialised
  with the same length-prefixed TLV scheme as `AssociatedData` (see
  `phase6-ad-serialization.md`). `round` gives replay rejection; `direction` prevents an
  uplink blob being reflected back as a downlink.
- **Nonce:** fresh 16-byte CSPRNG nonce per message, transmitted alongside, as now.
- **Overhead:** 135,200 B of weights + safetensors header + 32 B → ≈ 0.02 % expansion per
  message, versus the 33 % of Eq. (29) at 96-byte telemetry. Report both.
- **Pure-Python backend on a Pi:** encrypting ~135 KB per round will take seconds. Benchmark
  with `eval/crypto_benchmark.py` on the Pi and record it in the manifest; do not assume.

## 4. Transport

- ESP32 → Pi: MQTT to a broker **on the Pi** (farm-local; readings never traverse the laptop).
- Pi ↔ aggregator: a minimal length-prefixed TCP protocol carrying opaque
  `nonce || ad || ciphertext` frames. One request/response per round per client:
  client sends its encrypted update, blocks, receives the encrypted global state. The
  aggregator runs the round once all K updates for that round have arrived.
- Pi → cloud receiver: the existing `CloudTransport` protocol, with an HTTPS (or MQTTS)
  implementation replacing `MockCloudReceiver`; self-signed demo certificate outside git.
  The Phase 6/7 Ascon-on-telemetry code path is retired from the runtime pipeline; its
  crypto core is reused unchanged by the weight channel.

## 5. Modules to add or change

| Path | Change |
| --- | --- |
| `crypto/ascon_aead.py` | add `WeightAssociatedData` (§3 layout) next to `AssociatedData` |
| `federated/transport.py` (new) | `SecureWeightChannel`: `seal(state, n_k, round, direction)` / `open(frame)` returning `⊥` on any failure; per-direction key handling |
| `federated/server.py` | split into aggregation-only `FederatedServer` (unchanged) + `AggregatorNode` that serves the TCP protocol and calls it |
| `federated/client.py` | `ClientNode` wrapper: local training as now, then seal/open via the channel |
| `routing/cloud_sink.py` | `TlsCloudReceiver` implementing `CloudTransport` (plain TLS, no Ascon); `send_encrypted` renamed to reflect that the transport, not the app layer, encrypts |
| `telemetry/mqtt_source.py` (new) | subscribe to the farm broker, hand JSON payloads to the provenance adapter |
| `scripts/run_aggregator.py`, `scripts/run_client_node.py`, `scripts/run_cloud_receiver.py` | node entry points; `asa aggregator` / `asa client-node` / `asa cloud-receiver` |
| `firmware/esp32_sensor/` | sketch: publish readings at an interval; `scenario` topic toggles attack mode |
| `configs/base.py` | `HardwareConfig` (addresses, ports, key-file path, per-client sequence cap, rounds) |

## 6. Tests that must be green before Phase 8 closes

- `test_weight_channel_tamper.py`: flipping any byte of ciphertext, AD, nonce, or tag →
  `⊥`; the aggregator never updates state from a `⊥`.
- `test_weight_channel_replay.py`: a valid round-r frame re-sent at round r+1 is rejected;
  an uplink frame presented as downlink is rejected.
- `test_weight_channel_roundtrip.py`: seal → open reproduces the state dict bit-for-bit and
  the authenticated `n_k`; FedAvg over the channel equals in-process FedAvg (`test_fedavg_weighting.py` stays green).
- `test_nonce_collision.py` extended to cover `k_up`/`k_down`.
- `test_path_disjointness.py` unchanged and green — the runtime plane is not touched.
- Existing Phase 6 KAT/tamper tests unchanged.

## 7. Hardware feasibility notes

- **Python ≥ 3.12** is a hard floor. Raspberry Pi OS Bookworm ships 3.11 and cannot install
  the stack; use current Raspberry Pi OS (Trixie) or Ubuntu 24.04. PyTorch publishes aarch64
  CPU wheels.
- **Compute:** Phase 4 clients hold 284k–458k sequences and 20 rounds took ~2.7 h on the
  laptop. A Pi 4 is roughly 10–20× slower. **Cap the per-client partition for the hardware
  run** (`sequence_cap`, ~30–50k) and use fewer rounds. The hardware run demonstrates the
  architecture; headline metrics remain the laptop simulation, and the manifest records which
  is which.
- Pi 5 / 8 GB preferred; Pi 4 works, slower. Bandwidth per round (~270 KB per client) is
  trivial over Wi-Fi.

## 7a. Bill of materials (what to actually buy)

Derived from §2's node roles and the real firmware wiring in `firmware/esp32_sensor/`
(`diagram.json` pins, `libraries.txt`). K = 3 is the hardware target: two Pis plus the
laptop-simulated third client, so **only two Raspberry Pis are needed**.

### Core compute

| Item | Qty | Why |
| --- | --- | --- |
| Raspberry Pi 5, 8 GB | 2 | One per farm: local GRU, Mosquitto broker, runtime pipeline. Pi 4 works (§7) but is 10–20× slower than the laptop. |
| ESP32 DevKitC V4 | 6 | 3 per farm (`soil01–03`, `soil04–06`) — the board `diagram.json` targets. |
| Laptop | 1 (existing) | Aggregator + TLS cloud receiver + the simulated third client. No third Pi. |

### Per sensor node (x6)

| Item | Qty | Wiring |
| --- | --- | --- |
| DHT22 temperature/humidity | 6 | data -> **GPIO 15** |
| Capacitive soil-moisture sensor (v1.2/v2.0) | 6 | analogue out -> **GPIO 34** |
| 10 kOhm resistor | 6 | DHT22 pull-up; omit if buying breakout modules, which include it |
| Half-size breadboard | 6 | |
| Jumper wires M-M / M-F | 1 pack | |
| USB cable (micro-USB or USB-C) | 6 | match the board revision before ordering |

### Power, storage, network

| Item | Qty | Why |
| --- | --- | --- |
| microSD 32 GB+ A2 | 2 | Pi boot |
| Official Pi 5 PSU (27 W USB-C) | 2 | Pi 4 needs 15 W instead |
| Active cooler / heatsink | 2 | on-device training thermally throttles without it |
| Wi-Fi router or access point | 1 | everything must share one LAN: ESP32s -> Pi brokers, Pis -> laptop aggregator |
| Ethernet cable | 2 | optional; puts the Pi->laptop weight channel on wire and leaves Wi-Fi to the ESP32s |
| Powered USB hub | 1 | optional, for flashing six ESP32s |

### Three constraints that affect what you buy

1. **Not Raspberry Pi OS Bookworm.** It ships Python 3.11 and this stack cannot install on it:
   numpy >= 2.4 and scipy >= 1.16 both declare `requires-python >= 3.12`. Flash **Pi OS Trixie
   or Ubuntu 24.04 (aarch64)** — §7's hard floor, restated here because it is a purchasing-time
   decision, not a setup-time one.
2. **Capacitive soil sensors, not resistive.** Resistive probes corrode within weeks in wet
   soil. GPIO 34 is on ADC1, which is correct: ADC2 is unusable while Wi-Fi is active on ESP32.
3. **Bandwidth is not a constraint.** The containerised twin measured 407,752 B up / 407,758 B
   down per client per round (§7 budgets ~270 KB). Ordinary Wi-Fi is sufficient.

## 8. Attack scenarios from the ESP32s — assumption to confirm

(The other open assumption — whether telemetry Ascon is kept — was settled on 2026-09-19:
it is not; see §1 and §4.)

The local GRU consumes the 16 selected CICIoT2023 flow features, not packets. Two ways to
drive attack scenarios from the ESP32s:

- **(A) Scenario-triggered replay (assumed for this plan).** The ESP32 publishes real
  readings; switching it to "attack" makes the Pi's provenance adapter pair that device's
  messages with held-out *malicious* CICIoT2023 records instead of benign ones. This keeps the
  G6 provenance boundary and needs no feature extractor.
- **(B) Real attack traffic + live extractor.** The ESP32 generates actual traffic (e.g. an
  MQTT flood) on the isolated farm Wi-Fi and the Pi extracts the 16 features from a live
  capture. Tractable because F=16, but the extractor semantics (window, `IAT`, `Protocol
  Type` encoding) are not published, so a distribution shift is expected. Would be its own
  gated phase after (A) works.

## 9. Exit criterion

Two Pis and the simulated third client complete R rounds against the laptop aggregator with
every weight frame Ascon-sealed in both directions; a tampered or replayed frame is rejected
and logged; the global model reached by the hardware run matches an in-process FedAvg run on
the same capped partitions bit-for-bit; each Pi classifies its own ESP32 stream with the
returned weights and routes benign → cloud receiver, malicious → alert sink, with zero
malicious-verdict payloads on the cloud path; a manifest records key labels, crypto backend
provenance, per-round timings, and Ascon overhead on the Pi.
