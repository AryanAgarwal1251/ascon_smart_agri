# The software twin, step by step: how the hardware was built without hardware

This document explains what the "twin" is, how two Raspberry Pis, six ESP32s, a laptop and a
cloud endpoint were reproduced in software, what each container does, and what the verified run
actually proved. It is written to be read start to finish by someone who has not seen the
topology before.

---

## 1. The problem the twin solves

Suppose you buy two Raspberry Pis and six ESP32s, wire them up, install everything, run the
demo — and it doesn't work.

**Where is the fault?** The code? The wiring? The Wi-Fi? The Ascon keys? The MQTT topics? The
TLS certificate? You have no way to tell, and every test cycle now costs a reflash and a reboot
on a board that is 10–20× slower than your laptop.

So the order is inverted. **Everything that can be wrong in software is made right first**, with
the real code, on a real network, where a test cycle costs seconds. Then the only thing left
that can break on hardware is the hardware — the wiring and the boards.

That is the twin's entire purpose: when the Pis arrive, a failure is **unambiguously** a
hardware problem.

---

## 2. What "twin" means here — and what it is not

A twin is **not** a mock, a stub, or a simulation of the logic.

| | |
| --- | --- |
| **Same code** | The containers run `asa aggregator`, `asa client-node`, `asa pi-runtime` — the identical commands a Pi will run. Nothing is swapped for a test double. |
| **Separate processes** | Each node is its own OS process in its own container, not threads in one program. A bug that only appears across a process boundary still appears. |
| **Real network** | Frames cross a real TCP/IP network between containers. Real sockets, real MQTT, real TLS handshakes. |
| **Same CPU architecture** | Images are built `linux/arm64` — the Raspberry Pi's architecture — so the aarch64 PyTorch wheels a Pi would install are the ones exercised. |

What changes going to hardware is **only** where the processes run. No code, no configuration
structure, no protocol.

**What the twin cannot give you:** real timing. A container on a laptop is far faster than a
Pi 4. Every manifest the twin writes is therefore tagged `platform=docker-arm64-simulation`, so
no timing number from it can be mistaken for a Pi measurement.

---

## 3. The three-layer strategy

No single free tool simulates a Linux Raspberry Pi and an ESP32 microcontroller together, because
they are different kinds of machine. So the problem is split:

| Layer | Tool | What it proves | What it cannot |
| --- | --- | --- | --- |
| ESP32 sensor nodes | **Wokwi** (browser, free) | the real sketch, real Wi-Fi/MQTT stack, virtual DHT22 + moisture probe | LAN reachability — the free gateway reaches the public internet only |
| Pis + laptop + cloud | **Docker Compose** | the exact repo code as separate processes on a real network, aarch64 install, sealed frames on the wire, TLS to the cloud | **Pi timing** |
| Pi OS image | QEMU aarch64 (optional) | the OS-image install path | nothing the Docker layer doesn't |

A Raspberry Pi is a **computer running Linux**, so "simulating" one means emulating ARM Linux —
that is QEMU, or Docker at process level. An ESP32 is a **microcontroller**, one program on one
chip, which is why a browser can simulate it. (Wokwi's "Raspberry Pi Pico" is a microcontroller
too, not a Pi 4/5 — it does not run Linux and is not a substitute.)

---

## 4. The mapping: hardware → software

```
   REAL WORLD                              SOFTWARE TWIN
   ──────────                              ─────────────

   3× ESP32 on farm 1          ──────►     sensors-farm1    (one container, 3 virtual devices)
   3× ESP32 on farm 2          ──────►     sensors-farm2
   Mosquitto on Pi #1          ──────►     broker-farm1     (eclipse-mosquitto:2)
   Mosquitto on Pi #2          ──────►     broker-farm2
   Raspberry Pi #1             ──────►     pi-1             (client-node → pi-runtime)
   Raspberry Pi #2             ──────►     pi-2
   Simulated 3rd client        ──────►     sim-3            (client-node only)
   Laptop: master GRU          ──────►     aggregator       (:7700)
   Cloud endpoint              ──────►     cloud-receiver   (:8443, TLS)
   Flashing keys/certs by hand ──────►     init-secrets     (one-shot, exits 0)
```

Ten containers. Six physical ESP32s collapse into two containers because a virtual sensor
process can drive three device identities at once — and the Pi cannot tell the difference, since
both the real sketch and `scripts/run_virtual_sensor.py` publish the identical contract.

---

## 5. Every container, in detail

All nodes share one image, built from [`docker/Dockerfile`](../docker/Dockerfile):
`python:3.12-slim` + `openssl`, then the CPU-only aarch64 PyTorch wheel and `pip install -e .`,
with `ENTRYPOINT ["asa"]`. The role is chosen purely by the compose `command`. One image, nine
roles.

Shared mounts: `./artifacts` read-write (manifests land on your disk), `keys:ro`, `certs:ro`.

### `init-secrets` — the one-shot

```sh
asa aggregator --generate-keys --keys /keys/phase8.demo.key --clients pi-1,pi-2,sim-3
asa cloud-receiver --generate-cert --cert /certs/cloud.demo.pem --cert-hostname cloud-receiver
```

Generates the per-client Ascon channel keys and a self-signed TLS certificate into shared
volumes, then **exits 0 by design**. On hardware this step is you copying a key file to each Pi.

> ⚠️ **This container is why you must never run `docker compose up --abort-on-container-exit`.**
> Compose treats *any* container exit as the abort signal — including a one-shot that succeeded.
> Measured 2026-09-22: init-secrets finished 4.2 s in, compose tore down the brokers and killed
> the aggregator 3.4 s later, and the Pis then spent **28 minutes** loading data before dying on
> "Name or service not known". `--exit-code-from` implies the same flag and fails the same way.

### `broker-farm1`, `broker-farm2` — the farm-local MQTT brokers

`eclipse-mosquitto:2` with [`docker/mosquitto.conf`](../docker/mosquitto.conf): `listener 1883`,
`allow_anonymous true`, `persistence false`. Anonymous is acceptable *inside* the compose
network, which stands in for the farm's own Wi-Fi; **the hardware brokers should set a
password.**

Two separate brokers, not one, because this is the farm boundary: farm 1's readings must never
be reachable from farm 2. Each Pi runs its own broker.

### `aggregator` — the master GRU

```sh
asa aggregator --keys /keys/phase8.demo.key --clients pi-1,pi-2,sim-3 \
               --host 0.0.0.0 --port 7700 --rounds $ROUNDS \
               --checkpoint artifacts/phase8_twin_global_model.safetensors
```

Listens on 7700. Each round it waits for all three clients' sealed updates, opens each frame
under that client's key, runs **weighted FedAvg** (weight `n_k` = number of training
*sequences*), reseals the new global state per client, and sends it back. **A frame that opens
to `⊥` is never applied.**

### `cloud-receiver` — the cloud endpoint

```sh
asa cloud-receiver --host 0.0.0.0 --port 8443 --cert /certs/cloud.demo.pem \
                   --edges pi-1,pi-2 --seconds $CLOUD_SECONDS
```

A TLS server that accepts benign readings and is replay-guarded per device. It keeps its **own**
independent count of what it accepted — which is what makes the G1 check a real cross-check
rather than a Pi reporting on itself.

### `pi-1`, `pi-2` — the Raspberry Pis

The only containers that run **two commands in sequence**, chained with `&&`:

```sh
asa client-node --client-id pi-1 --client-index 0 --server aggregator:7700 \
                --keys /keys/phase8.demo.key --rounds $ROUNDS --local-epochs $EPOCHS \
                --sequence-cap $SEQ_CAP --platform docker-arm64-simulation \
&& asa pi-runtime --client-id pi-1 --farm farm1 --broker broker-farm1:1883 \
                  --cloud https://cloud-receiver:8443 --ca-cert /certs/cloud.demo.pem \
                  --messages $MESSAGES --idle-timeout 120
```

**Stage 1, `client-node` — the training plane.** Loads its Dirichlet partition of the CICIoT2023
cache, trains its local GRU for E epochs, seals the parameters, sends them, receives the sealed
global state back, applies it, repeats for R rounds. Saves its own model.

**Stage 2, `pi-runtime` — the runtime plane.** Subscribes to `farm/farm1/sensor/#`, buffers each
device's readings into W = 16 windows, classifies with the model it just finished federating, and
routes: **benign → TLS to the cloud, malicious → local alert, and nothing else.**

This two-stage chain is exactly the real deployment shape: train periodically, classify
continuously.

### `sim-3` — the third federated client

`client-node` only, `--client-index 2`. No sensors, no runtime, no cloud. It exists so that
**K = 3** without buying a third Pi — and K = 3 matters because K = 2 federation is degenerate
(just averaging two models, with no heterogeneity story to tell).

### `sensors-farm1`, `sensors-farm2` — the ESP32 stand-ins

```sh
asa virtual-sensor --farm farm1 --devices soil01,soil02,soil03 \
                   --broker broker-farm1:1883 --interval $INTERVAL \
                   --messages $MESSAGES --attack-after $ATTACK_AFTER --attack-device soil02
```

Publishes the same JSON contract as the real sketch, with a monotonic `seq` per device. After
`--attack-after` messages it flips the scenario — farm 1 on **one device only** (`soil02`), farm 2
across the whole farm, so both the single-compromised-device and whole-farm cases are exercised.

> **The scenario switch never synthesises data.** It only selects *which held-out record pool*
> the provenance adapter draws from. This is the G6 boundary (§7).

---

## 6. What happens when you run it, step by step

```bash
docker compose up -d --build
```

| Step | What happens |
| --- | --- |
| 1 | Image builds (once): python 3.12-slim, aarch64 torch, the package |
| 2 | `init-secrets` generates Ascon keys + TLS cert into shared volumes, **exits 0** |
| 3 | Both brokers start and listen on 1883 |
| 4 | `aggregator` binds :7700 and waits for 3 clients; `cloud-receiver` binds :8443 |
| 5 | `pi-1`, `pi-2`, `sim-3` each load their Dirichlet partition (~25 s) |
| 6 | **Round 0 — the scaler round.** Each client sends only count/mean/M2 sufficient statistics, sealed. The aggregator combines them by Chan's parallel formula, so the federated scaler equals a pooled scaler **without pooling any raw data.** |
| 7 | **Rounds 1…R — the weight rounds.** Train locally → seal → send → FedAvg → reseal → send back → apply. R times. |
| 8 | Aggregator saves the global checkpoint and its manifest, then exits |
| 9 | `pi-1`/`pi-2` move to stage 2 and subscribe to their farm's sensor topic |
| 10 | Sensors publish; each Pi buffers per device until it has W = 16 readings, then classifies every new reading |
| 11 | Benign → TLS → cloud receiver. Malicious → local alert. Never both. |
| 12 | Each Pi writes a runtime manifest with its counts; the cloud writes its own |

Watch it with `docker compose logs -f pi-1 pi-2`, and tear down with `docker compose down -v`.

---

## 7. The G6 boundary — what the model actually classifies

This is the part most easily misunderstood, so it is stated plainly.

The ESP32 sends `{"deviceId": "soil01", "temperature": 24.9, "soilMoisture": 41.2, "seq": 137}`.
**The GRU never sees those numbers.** A model trained on CICIoT2023 flow features cannot consume
an application-layer JSON payload — they are different objects with different origins.

So the pipeline keeps two planes apart:

| Plane | Data | Role |
| --- | --- | --- |
| **Application** | the sensor JSON | this is what Ascon protects end-to-end; **never parsed into features** |
| **Network** | a flow-feature vector from a **held-out** CICIoT2023 record (the `phase4_cache.npz` test split, 311,565 rows, never trained on) | this is what the GRU classifies |

The provenance adapter pairs each arriving message with a held-out record and records the
provenance of every feature value. **No feature is ever invented to bridge the two planes.**

**What the twin therefore demonstrates:** architectural correctness — the pipeline separates the
planes, routes on the verdict, seals and verifies correctly, and never places a malicious-verdict
payload on the cloud path.

**What it does not demonstrate:** that a CICIoT2023-trained model would detect attacks against a
live agricultural MQTT deployment. That needs capture and feature re-extraction on the target
network, and the claim is not made.

---

## 8. The verified run

Configuration: `ROUNDS=20 SEQ_CAP=10000 EPOCHS=1 INTERVAL=1.0 MESSAGES=7200 ATTACK_AFTER=3600
CLOUD_SECONDS=10800`. All ten services exited 0.

### The weight channel

| | Result |
| --- | --- |
| Rounds completed | **20 / 20** |
| Sealed frames | **120** (3 clients × 20 rounds × 2 directions), plus the scaler round |
| **Frames rejected** | **0** |
| Scaler round | 1,480 B up / 1,462 B down |
| Per client, whole run | 2,718,833 B up / 2,718,867 B down, in 457–459 s |

**Reading the byte counts correctly:** the aggregator logs `407,752 B` per round, which is the
**aggregate over all three clients in one direction** — not one client's cost. Per client per
round it is 135,942 B up and 135,943 B down, so **271,883 B both ways**, matching plan §7's
~270 KB estimate and reconciling with Phase 4's independently measured 815,136 B per round
(3 clients × 2 directions).

### The runtime plane and G1

| | pi-1 | pi-2 | total |
| --- | --- | --- | --- |
| Readings received | 6,626 | 6,627 | 13,253 |
| Still buffering (< 16 for that device) | 45 | 45 | — |
| Benign → cloud | 5,421 sent, **0 failed** | 3,503 sent, **0 failed** | **8,924** |
| Malicious → local alert | **1,160 → 1,160** | **3,079 → 3,079** | **4,239** |
| **Malicious reaching the cloud** | **0** | **0** | **0** |

The cloud receiver independently recorded **`accepted 8924, rejected 0`** — exactly
5,421 + 3,503. **Both sides of the boundary agree, with no adjustment.**

`4,239` malicious readings were classified, alerted locally, and dropped from the cloud path.
G1 is structural — [`routing/alert_sink.py`](../src/ascon_smart_agri/routing/alert_sink.py) holds
no reference to the cloud transport, and `tests/test_path_disjointness.py` asserts it — but this
run is the evidence that it holds under volume.

How strongly G1 has been tested over time:

| Run | Malicious readings exercised |
| --- | --- |
| 2026-09-19 loopback | 119 |
| First containerised attempt (**invalid**) | 1 |
| Second containerised run | 491 |
| **This run** | **4,239** |

### Supporting test suite

**64 of 64** hardware-path tests pass: weight channel, node loopback, software twin, TLS path,
path disjointness, Ascon KAT, Ascon tamper, nonce collision, MQTT source, adversarial
architecture suite. Full suite: **438 passed, 0 skipped**.

---

## 9. Two runs that were not valid tests

Recorded so the configuration is not repeated by accident.

**Attempt 1 — G1 "passed" on one event.** With the compose default `MESSAGES=240`, each Pi saw
exactly **1** malicious reading. The sensors begin publishing as soon as the broker is up, while
the Pis are still federating, and MQTT QoS 0 discards anything published before a subscription
exists. "Zero malicious reached the cloud" is not evidence when only one malicious reading
existed.

**Attempt 2 — a delivery failure misread as a transport problem.** Raising `MESSAGES` to 800
without raising `CLOUD_SECONDS` made 190 benign sends appear to fail. The receiver had simply
reached its 600 s budget and exited cleanly while the Pis kept sending. Sent (355 + 288) still
equalled accepted (643) exactly.

**The rule:** both the attack window and the receiver's window must cover the Pi's
**post-federation** listening period, not the wall-clock run.

**And the distinction that matters when reading any log:** a *security* property (no malicious
reading reaches the cloud) and a *delivery* property (every benign reading arrives) fail for
completely different reasons, and only the first is a G1 claim.

---

## 10. What is left for hardware

Nothing in the code changes.

| Step | Detail |
| --- | --- |
| OS | Python ≥ 3.12 → **Raspberry Pi OS Trixie or Ubuntu 24.04**. **Not Bookworm** — it ships 3.11 and the stack cannot install |
| Secrets | copy `keys/phase8.demo.key` and `keys/cloud.demo.pem` to each Pi (this is what `init-secrets` did) |
| Broker | run Mosquitto on each Pi, **with a password** this time |
| Boards | flash the six ESP32s, `BROKER` = that farm's Pi IP — wiring in [`breadboard-wiring.md`](breadboard-wiring.md) |
| Tagging | pass `--platform raspberry-pi-<model>` so manifests record what they ran on |

The per-round `round_seconds` from the hardware run are then the **real** Pi feasibility numbers.
Components to buy: [plan §7a](phase8-hardware-federation.md). Topology and wiring overview:
[`architecture-diagram.md`](architecture-diagram.md).
