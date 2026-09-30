# Phase 8 — The software twin, verified under load

**Status: ✅ Software verification complete; hardware pending.** Source:
[`manifest_phase8_aggregator.json`](../artifacts/manifest_phase8_aggregator.json),
[`manifest_phase8_runtime_pi-1.json`](../artifacts/manifest_phase8_runtime_pi-1.json),
[`manifest_phase8_runtime_pi-2.json`](../artifacts/manifest_phase8_runtime_pi-2.json),
[`manifest_phase8_cloud_receiver.json`](../artifacts/manifest_phase8_cloud_receiver.json).
Topology: [`docker-compose.yml`](../docker-compose.yml). Plans:
[`phase8-hardware-federation.md`](../docs/plans/phase8-hardware-federation.md),
[`phase8-software-twin.md`](../docs/plans/phase8-software-twin.md).

## What the architecture requires

The architecture, not the design paper, is authoritative here (CLAUDE.md Golden Rule 1, user
direction of 2026-09-18). It requires that **Ascon-AEAD128 protects the model weights exchanged
between each local GRU and the master GRU, in both directions**, that the benign/malicious
decision is taken by the local GRU using the weights it received back, and that only benign
readings reach the cloud.

The point of the twin is to run that architecture as **separate processes on a real network**
before any hardware exists, so that moving to hardware is a wiring exercise rather than a
debugging exercise.

## What we achieved

The whole topology ran in containers — aggregator, two Mosquitto brokers, TLS cloud receiver,
`pi-1`, `pi-2`, the simulated third client `sim-3`, and two virtual sensor farms. **All ten
services exited 0.**

### The weight channel

| | Result |
| --- | --- |
| Clients | 3 (`pi-1`, `pi-2`, `sim-3`) |
| Scaler round | 1,480 B up / 1,462 B down, 25.2 s |
| Weight round | **407,752 B up / 407,758 B down**, 20.0 s and 10.7 s |
| **Frames rejected** | **0**, in either direction |

Every client update and every global-state download was an AEAD frame under a per-client,
per-direction key. Nothing opened to bottom, and nothing was applied that did.

The measured per-round figure is what the hardware phase needs for its bandwidth budget:
plan §7 estimated ~270 KB per client per round; the realised figure is ~408 KB, comfortably
inside ordinary Wi-Fi either way.

### The runtime plane, and G1

| | pi-1 | pi-2 | total |
| --- | --- | --- | --- |
| Readings received | 688 | 686 | 1,374 |
| Benign → cloud | 470 sent, **0 failed** | 323 sent, **0 failed** | 793 |
| Malicious → local alert | **173 → 173** | **318 → 318** | **491** |
| **Malicious reaching the cloud** | **0** | **0** | **0** |
| G1 equality check | True | True | |

The cloud receiver independently recorded **accepted 793, rejected 0** — exactly 470 + 323.
Both sides of the boundary agree, and the numbers reconcile without adjustment.

**G1 was exercised under real load.** 491 malicious readings were classified, alerted locally,
and dropped from the cloud path. Path disjointness is structural — `routing/alert_sink.py`
holds no reference to the cloud transport — and `tests/test_path_disjointness.py` asserts it,
but this is the first run in which a large number of malicious verdicts actually occurred on
the wire between containers.

### The test suite behind it

**64 of 64 hardware-path tests pass**: `test_weight_channel.py`, `test_node_loopback.py`,
`test_software_twin.py`, `test_tls_path.py`, `test_path_disjointness.py`, `test_ascon_kat.py`,
`test_ascon_tamper.py`, `test_nonce_collision.py`, `test_mqtt_source.py`,
`test_architecture_adversarial.py`. The full suite is 438 passed, 0 skipped.

## Decisions flagged, not silently made

**The first two attempts at this run were not valid tests, and both are recorded so the
configuration is not repeated by accident.**

1. With the compose default `MESSAGES=240`, each Pi saw exactly **1** malicious reading. The
   sensors begin publishing as soon as the broker is up, while the Pis are still federating
   (~160 s), and MQTT QoS 0 discards anything published before a subscription exists. G1
   "passed" on a single event, which is not evidence.
2. Raising `MESSAGES` to 800 without raising `CLOUD_SECONDS` produced a second false signal:
   190 benign sends appeared to "fail", but the receiver had simply reached its 600 s budget
   and exited cleanly while the Pis kept sending. Sent (355 + 288) still equalled accepted
   (643) exactly.

**The passing configuration is `MESSAGES=800 ATTACK_AFTER=400 CLOUD_SECONDS=2400`.** The rule
behind it: both the attack window and the receiver's window must cover the Pi's *post-
federation* listening period, not the wall-clock run.

The distinction matters for reading any future log: a security property (no malicious reading
reaches the cloud) and a delivery property (every benign reading arrives) fail for completely
different reasons, and only the first is a G1 claim.

## Honest gaps

- **Timings from a container are not Pi timings.** Every manifest is tagged
  `platform=docker-arm64-simulation` for this reason. A Pi 4 is roughly 10–20× slower than the
  laptop; the real `round_seconds` figures come only from hardware.
- **No physical ESP32 has been flashed.** The sensor side is virtual sensors plus the Wokwi
  sketch in [`firmware/esp32_sensor/`](../firmware/esp32_sensor/). The sketch and the Python
  virtual sensor publish the identical contract, and `asa pi-runtime` cannot tell them apart,
  but that is a design property, not a measurement.
- **Wokwi's free gateway reaches the public internet, not a LAN**, so an end-to-end
  Wokwi-ESP32 → real-Pi-broker path has not been exercised.
- **The verified run used R = 2 rounds.** A run at the demo's R = 20 — 120 sealed frames rather
  than 12 — was launched on 2026-10-01 and its result belongs in this file when it lands.

## What moving to hardware requires

Nothing in the code changes. Per node: install on Python ≥ 3.12 (Raspberry Pi OS Trixie or
Ubuntu 24.04 — **not Bookworm**, which ships 3.11 and cannot install the stack), copy the demo
key and cloud certificate, run Mosquitto on each Pi, flash the ESP32s with the Pi's IP as
`BROKER`, and pass `--platform raspberry-pi-<model>` so the manifests record what they ran on.
The bill of materials is [plan §7a](../docs/plans/phase8-hardware-federation.md).
