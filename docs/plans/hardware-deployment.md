# Hardware deployment plan: ESP32 sensor nodes + Raspberry Pi 5 federated gateways

**Status:** plan only. Nothing in this document has been built yet. Written 2026-09-29.

**Goal:** show the finished software system working on real hardware. Three farm "zones", each
with ESP32 sensor nodes publishing over MQTT to a Raspberry Pi 5 gateway. Each gateway runs a
local GRU on the live network traffic its sensors generate. A fourth machine is the aggregator
("master GRU"): it receives Ascon-encrypted weights from the three gateways, runs weighted
FedAvg, and sends Ascon-encrypted global weights back. The gateways then detect with the global
model.

**Dataset for the hardware build:** Edge-IIoTset instead of CICIoT2023. The feature pipeline
for it currently lives in a teammate's local repo and has not been merged here yet.

The steps are grouped into phases **H0–H8**. They are gated the same way the software phases
were (CLAUDE.md rule 2): each phase has an exit criterion, and a phase does not start until the
previous one's criterion is met. H0 comes first because it records decisions that would
otherwise have to be undone later.

---

## 0. Read this first: five things that change the plan

### 0.1 This work goes outside the scope the repo currently declares

Three rules in [CLAUDE.md](../../CLAUDE.md) and the [design paper](../design_paper.md) conflict
with this plan. Under golden rule 1 they are **flagged here, not quietly worked around**:

| Current rule | Where | What the hardware plan needs |
| --- | --- | --- |
| "Physical hardware" is listed under *Out of scope — stub, never implement* | CLAUDE.md | Physical hardware becomes in scope |
| "Dataset is CICIoT2023 only — wire up no other" | CLAUDE.md | Edge-IIoTset becomes the dataset for the hardware build |
| G6 / Section III-H: network features come only from held-out dataset records; live capture is named as future work and the paper explicitly does **not** claim live detection | Design paper III-H, `telemetry/provenance.py` | Network features come from **live packet capture** on the gateway |

**What to do:** before writing any code, add a second *Implementation Deviation* addendum to
`docs/design_paper.md` in the same form as the 2026-09-20 Ascon addendum. It should state what
changed, why, what still holds (G1 path disjointness, the Ascon weight channel, the KAT gate,
safetensors-only, FedAvg by sequence count, LayerNorm, no PCA, no oversampling), and what new
claim is being made. Update CLAUDE.md's out-of-scope list and dataset rule to match. The user
approves this, the same way the 2026-09-20 deviation was approved.

Keep the CICIoT2023 results. They are the published baseline. Edge-IIoTset should be added as a
**second dataset profile**, not a replacement: CICIoT2023 stays the default config, and its
tests stay green.

### 0.2 The Raspberry Pi 5 GPU cannot be used for training

The Pi 5's GPU is a Broadcom VideoCore VII. PyTorch has no backend for it (no CUDA, no ROCm,
and no usable Vulkan/OpenCL training path). **All training and inference on the Pi runs on
its 4-core Cortex-A76 CPU.**

That is fine for this project. The default model has 33,800 parameters (Eq. 19). One local
round of E=3 epochs over a few hundred thousand sequences takes minutes on the CPU, and a
single inference takes well under a millisecond. The Pi 5 is still a good choice, because it is
much faster than a Pi 4, has 8 GB RAM, and has a PCIe connection for NVMe storage. It just
isn't a GPU machine.

The *Raspberry Pi AI Kit / AI HAT+* (Hailo NPU) is **not needed**. It runs inference only, for
models compiled with Hailo's toolchain, and recurrent layers like GRU are poorly supported
there. Do not buy it for this project.

### 0.3 Python version: the repo pins 3.12 exactly

`pyproject.toml` says `requires-python = ">=3.12,<3.13"`. Default Raspberry Pi OS does not ship
3.12 (Bookworm ships 3.11; the Debian 13 "Trixie"-based release ships 3.13). The options are:

- **Recommended: Ubuntu Server 24.04 LTS (64-bit) for Raspberry Pi 5.** It ships Python 3.12
  natively and is officially supported on the Pi 5.
- Alternatively, Raspberry Pi OS 64-bit, with Python 3.12 built through `pyenv`. This works,
  but the build takes a long time and is one more thing that can break.

Use the same OS on all four Pis.

### 0.4 The GRUs run on the Pis, not on the ESP32s

An ESP32 cannot run training, and it cannot capture its own network traffic. In this design:

- **ESP32** = field sensor node. It reads sensors and publishes MQTT. This is the "device" on
  Channel 1 (device → gateway) in the threat model.
- **Raspberry Pi 5 gateway** = federated client *and* detector. It captures the traffic
  arriving from its ESP32s, extracts features, runs the GRU, routes by verdict (Eq. 5), and
  trains locally during federated rounds.
- **Aggregator** = the "master GRU". It holds the global model and runs FedAvg. It never sees
  raw records or captured traffic.

### 0.5 Features must be extracted the same way in training and on the Pi

This is the most important technical point in the plan, and the one most likely to be missed.

Section III-H of the paper refused to claim live detection because it had no way to reproduce
CICIoT2023's feature extractor on a live network. **Edge-IIoTset removes that obstacle.** Its
features are per-packet protocol fields extracted with **TShark/Zeek** (for example `tcp.flags`,
`tcp.len`, `mqtt.msgtype`, `mqtt.len`, `dns.qry.name.len`, `icmp.checksum`), and TShark runs
fine on a Raspberry Pi. Edge-IIoTset's testbed also used ESP32 boards, agricultural sensors
(temperature/humidity, soil moisture, water level, pH, ultrasonic, flame…), and a Mosquitto
MQTT broker. That is very close to the hardware planned here.

This only helps if **the Pi computes each selected feature exactly as the dataset did**. If the
training features and the live features are computed differently, the model's results on the
hardware mean nothing. That is the "distribution shift we could not document" the paper warns
about. Phase H2 exists to prevent it, with a parity test: run our extractor over Edge-IIoTset's
own pcap files and check that it reproduces the dataset's CSV values.

---

## 1. Target architecture

```
                  ┌───────────────────────────────────────────────┐
                  │  AGGREGATOR ("master GRU")                    │
                  │  Raspberry Pi 5 #4 (or a laptop)              │
                  │  - holds global θ, runs weighted FedAvg (Eq.21)│
                  │  - per-client 16-byte Ascon keys              │
                  └───────▲───────────────▲───────────────▲───────┘
      Channel 3: Ascon-   │               │               │  (wired Ethernet,
      AEAD128 weight blobs│               │               │   isolated lab LAN)
      both directions     │               │               │
          ┌───────────────┴──┐  ┌─────────┴────────┐  ┌───┴──────────────┐
          │ GATEWAY A (Pi 5) │  │ GATEWAY B (Pi 5) │  │ GATEWAY C (Pi 5) │
          │ Wi-Fi AP (2.4GHz)│  │  same stack      │  │  same stack      │
          │ Mosquitto broker │  │                  │  │                  │
          │ tshark capture   │  │                  │  │                  │
          │ feature extractor│  │                  │  │                  │
          │ scaler → windows │  │                  │  │                  │
          │ local GRU        │  │                  │  │                  │
          │ VerdictRouter    │──┼──► benign → cloud sink (plaintext,    │
          │ AlertSink (LED/  │  │    2026-09-20 deviation)              │
          │  buzzer/log)     │  │                  │  │                  │
          └──▲──────▲────────┘  └──▲──────▲────────┘  └──▲──────▲───────┘
   Channel 1:│      │              │      │              │      │
   MQTT/Wi-Fi│      │              │      │              │      │
          ESP32   ESP32         ESP32   ESP32         ESP32   ESP32
          soil+   water+        soil+   light+        soil+   pH+
          DHT22   HC-SR04       DS18B20 BH1750        DHT22   rain

   ATTACKER laptop (Kali), joins one zone's Wi-Fi as a "compromised device"
   — isolated lab network only, never a shared/campus network.
```

Each gateway's Wi-Fi access point **is** its capture point: every packet between its ESP32s,
its broker, and anything else on that zone's network crosses `wlan0` on the Pi. This avoids
needing a managed switch with port mirroring.

---

## 2. Bill of materials

Prices are **rough indicative Indian retail ranges** (Robu.in, Robocraze, ThinkRobotics, Amazon
India, etc.). They change often, so check before ordering. Quantities assume three zones and
two ESP32 nodes per zone.

### 2.1 Compute and networking

| Item | Qty | Notes | Approx. ₹ each |
| --- | --- | --- | --- |
| Raspberry Pi 5, **8 GB** | 3 (+1 if the aggregator is a Pi) | 4 GB is enough for inference only, but local training plus tshark plus the broker is more comfortable with 8 GB | 8,000–9,500 |
| Official 27 W USB-C PD power supply | 3–4 | The Pi 5 throttles USB current and misbehaves on weaker supplies | 1,000–1,300 |
| Official Active Cooler | 3–4 | Training at full load will thermally throttle a Pi 5 without it | 450–600 |
| microSD, 64 GB, A2 / U3 (SanDisk Extreme or similar) | 3–4 | Or the M.2 HAT+ plus a 256 GB NVMe SSD if you want faster I/O (optional) | 700–1,000 |
| Pi 5 case that fits the active cooler | 3–4 | | 400–800 |
| Wired router or 5–8 port gigabit switch | 1 | Carries the **aggregator ↔ gateway** backhaul. Keep it isolated from the campus network | 1,000–2,500 |
| Ethernet cables (Cat6, 1–2 m) | 4–5 | | 100–200 |
| Laptop for the attacker role (Kali Linux, live USB is fine) | 1 | Any existing team laptop | — |
| USB-to-TTL / spare USB cables | — | For flashing and powering ESP32s | — |

**Aggregator choice:** FedAvg on 33,800 parameters is trivial work, so a team laptop can be the
aggregator. A fourth Pi 5 makes the demo symmetric and fully "edge". Either works; pick one in
H0.

### 2.2 Sensor nodes (per zone; ×3)

| Item | Qty/zone | Why this one | Approx. ₹ |
| --- | --- | --- | --- |
| ESP32 DevKit V1 (ESP32-WROOM-32, 38-pin) | 2 | 2.4 GHz Wi-Fi and plenty of ADC1 pins. It is also the board family Edge-IIoTset's testbed used | 350–550 |
| DHT22 (AM2302) temp/humidity | 1 | Matches the telemetry schema's `temperature`. Choose it over the DHT11 for accuracy | 250–400 |
| Capacitive soil moisture sensor v1.2 | 1–2 | Matches `soilMoisture`. **Capacitive, not resistive.** Resistive probes corrode within days | 100–180 |
| DS18B20 waterproof probe + 4.7 kΩ resistor | 1 | Soil temperature | 150–250 |
| HC-SR04 ultrasonic or float/water-level sensor | 1 | Tank level (Edge-IIoTset used both) | 80–200 |
| BH1750 light sensor (I²C) | optional | Canopy light | 120–200 |
| Rain/raindrop sensor module | optional | | 80–150 |
| Analog pH sensor kit (e.g., PH-4502C + probe) | optional, 1 total | Expensive and needs calibration buffers. One for the whole build is enough | 1,500–2,800 |
| 1-channel 5 V relay module + small 5 V DC pump + tubing | optional, 1 total | An actuator makes the cyber-physical consequence visible in the demo (Section I-A) | 250–450 |
| LED + buzzer | 1 per gateway | Physical output of `AlertSink` | 20–50 |
| Breadboard, jumper wires, 5 V USB power banks | as needed | Power banks let nodes sit "in the field" on a table | — |

### 2.3 Electrical notes for the ESP32

- **Use ADC1 pins only (GPIO 32–39) for analog sensors.** ADC2 cannot be used while Wi-Fi is
  active, and every node here uses Wi-Fi.
- ESP32 GPIOs are **3.3 V only**. The HC-SR04 echo pin and some pH modules output 5 V. Put a
  voltage divider (e.g., 1 kΩ / 2 kΩ) or a level shifter on them.
- The ESP32's ADC is non-linear near its rails. Calibrate the soil probe against a dry reading
  and a water reading, and store the two endpoints in firmware.

---

## 3. Phases

### H0: Decisions and approvals (no code)

Record each decision in the design-paper addendum (§0.1) and CHANGELOG.

1. **Scope approval.** Physical hardware and Edge-IIoTset move in scope, as described in §0.1.
2. **Dataset variant.** Edge-IIoTset ships two CSV variants: `DNN-EdgeIIoT-dataset.csv` (large)
   and `ML-EdgeIIoT-dataset.csv` (smaller, pre-sampled). It also ships the **raw pcaps** per
   attack and per sensor. Decide which one is the source of truth. **Recommendation: the
   pcaps.** Phase H2 needs them to build and check the extractor anyway, and features extracted
   by our own code are the only ones we can guarantee match on the Pi.
3. **Class taxonomy (C).** Edge-IIoTset has Normal plus 14 attacks in 5 families: DDoS (UDP,
   ICMP, TCP-SYN, HTTP), information gathering (port scanning, OS fingerprinting, vulnerability
   scanning), MITM (ARP/DNS spoofing), injection (SQLi, XSS, uploading), and malware (backdoor,
   password, ransomware). Choose between 15-way, 6-way (Normal + 5 families), or binary. This
   sets `model.n_classes` and so changes Eq. (19)'s parameter count. **Recommendation:** 6-way
   as the headline, plus binary for PR-AUC. That mirrors how the CICIoT2023 taxonomy was grouped
   in `data/taxonomy.py`. Some 15-way classes are also impractical to reproduce live on the
   testbed (§H7).
4. **Aggregator hardware.** A fourth Pi 5 or a laptop (§2.1).
5. **ESP32-side encryption.** Under the 2026-09-20 deviation, Ascon protects Channel 3 only, so
   the ESP32s do **not** need Ascon. Adding it later would bring golden rule 3 back into play:
   it would have to be the official `ascon-c` reference implementation, KAT-checked *on the
   device*. **Recommendation: out of scope for this build.**
6. **Merge plan for the teammate's Edge-IIoTset work.** It is uncommitted in their local repo.
   It needs to land on a branch here before H1 starts, so the pipeline has one home and one
   test suite.

**Exit:** the addendum is written and approved, CLAUDE.md is updated, and the six decisions are
recorded.

---

### H1: Edge-IIoTset through the existing pipeline (Phases 1–3 again, on the new dataset)

Everything here happens on a laptop or desktop, not the Pis. Use the existing code; add a
dataset profile rather than forking the pipeline.

1. **Bring in the teammate's work** on a branch. Diff their feature list and preprocessing
   against what this repo's pipeline does (dedup → split → four-stage selection). Reconcile
   before trusting either.
2. **Add a dataset profile.** Add a `data.dataset` field (or equivalent) to
   `configs/base.py` and a new `configs/edge_iiotset.yaml`. **Leave `default.yaml` on
   CICIoT2023,** because `tests/test_model_param_count.py` pins the default config to 33,800
   parameters. Add a loader under `data/` and a taxonomy mapping in `data/taxonomy.py` for the
   Edge-IIoTset labels.
3. **Phase 1 characterisation on Edge-IIoTset** (`asa characterize` with the new profile): real
   columns and types, nulls, zero-variance columns, the exact duplicate count, label counts, and
   the correlation matrix. As before, no later step may cite a figure this report didn't produce.
4. **Drop identifier and leakage columns before feature selection.** These are columns that
   identify a *host or session* rather than describe *behaviour*: `frame.time`, `ip.src_host`,
   `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`, `tcp.payload`, `tcp.options`,
   `http.file_data`, `http.request.full_uri`, `http.request.uri.query`, `mqtt.msg`,
   `tcp.srcport`, `udp.port`, and similar. A model that learns "attacks come from 192.168.0.x"
   will score well on the dataset and fail on our testbed, where the IPs are different. Confirm
   the actual list from the characterisation report.
5. **Check temporal order before windowing.** The GRU's windows assume rows in capture order
   within contiguous same-label runs (Section III-D). **If the CSVs were shuffled when they were
   published, windows built from them are meaningless.** Check this in characterisation, for
   example by looking at `frame.time` monotonicity per source file. If the CSVs are shuffled,
   that is another reason to extract features from the pcaps (H0 decision 2), which are in time
   order by construction.
6. **Dedup before split** (R3). Edge-IIoTset contains many exact-duplicate rows, especially in
   the flood classes. `tests/test_leakage.py` must pass on the new dataset.
7. **Four-stage feature selection, then measure the knee once and pin it.** Run `--knee-sweep`
   once (`run_phase3.py`), read the knee, and set `features.selected_f` in
   `edge_iiotset.yaml`. **Constraint for this build:** Stage 4 may only pick features the
   H2 extractor can compute live. Run feature selection over the *extractable* column set, not
   all 61 columns.
8. **Centralised GRU plus the full evaluation protocol** (Phase 3 gate, applied to Edge-IIoTset):
   macro-F1, per-class F1, balanced accuracy, MCC, confusion matrix, FPR, PR-AUC (binary), ≥3
   seeds, mean ± std, plus the RF and MLP baselines. Never report accuracy alone.
9. **Three-client federated simulation** (Phase 4 gate on Edge-IIoTset): Dirichlet partition,
   weighted FedAvg by **sequence** count, scaler equivalence, and the three-way bracket against
   centralised and local-only.

**Exit:** the Phase 3 and Phase 4 gate checkers pass on the Edge-IIoTset profile, the
CICIoT2023 profile is still green, and the pinned `F`, `C`, and parameter count are recorded in
CHANGELOG. Add a parameter-count test for the Edge-IIoTset profile alongside the existing one.

---

### H2: A live feature extractor that matches the dataset

New module, for example `src/ascon_smart_agri/capture/extractor.py`. Keep it separate from
`telemetry/provenance.py`, whose held-out-record contract should stay as it is for the
simulation path.

1. **One field list.** Write the selected feature names from H1 once, in the config. Generate
   both the offline pcap → CSV step and the live step from that same list. They must never be
   two lists that can drift apart.
2. **Offline mode:** `tshark -r file.pcap -T fields -e <field> ... -E separator=,`, applied to
   Edge-IIoTset's pcaps. Apply the dataset's documented post-processing (how empty fields are
   filled, how hex/strings are encoded) exactly, and write each step down.
3. **Live mode:** the same `tshark` command with `-i wlan0 -l` (line-buffered), read from a
   subprocess pipe. `pyshark` is an alternative, but calling tshark directly is simpler and
   faster.
4. **Parity test (the key test of this phase).** Run the offline extractor over a sample of
   Edge-IIoTset pcaps and compare the result with the dataset's own CSV rows for those pcaps.
   The selected features must match (exactly, or within a stated tolerance for floats). Commit a
   small pcap fixture and the expected rows so the test runs on every commit.
5. **Device key for windowing.** `sequences/streaming.py` keeps one rolling buffer per device.
   On the Pi, a "device" is the source MAC or IP of the packet. Use the MAC, which stays the
   same when DHCP changes the IP. Packets from the broker to a device count toward that device's
   stream, the same way they did in training. Whatever rule you choose, use it identically
   offline (H1 windowing) and live.
6. **Scaler:** the live path applies the **global federated scaler** (Chan's formula, Eqs.
   23–24) received from the aggregator. Never fit a scaler on live traffic.

**Exit:** the parity test is green, and the live extractor runs on a Pi for 10 minutes at
normal sensor traffic without dropping packets. Check this with tshark's dropped-packet
counter and with `ss`/`ifconfig`.

---

### H3: ESP32 sensor-node firmware

A new top-level directory, for example `firmware/esp32_node/`, built with **Arduino-ESP32** or
**PlatformIO**. PlatformIO is recommended because its builds are reproducible and it pins
library versions.

1. **Libraries:** `PubSubClient` (MQTT), `DHT sensor library` (Adafruit), `OneWire` +
   `DallasTemperature`, `BH1750` as needed. Pin every version in `platformio.ini`.
2. **Payload schema:** match `telemetry/simulate.py` so the runtime routing code does not
   change:
   ```json
   {"deviceId":"soil01","counter":1287,"temperature":24.8,"soilMoisture":42.5,"schemaVersion":"v1"}
   ```
   Keep `counter` monotonic per device and store it in NVS (`Preferences`) so it survives
   reboots. `routing/replay_guard.py` depends on it.
3. **Topics:** e.g. `farm/<zone>/<deviceId>/telemetry`. Use QoS 0 or 1, and keep the choice
   **consistent with Edge-IIoTset's traffic**, which affects the `mqtt.*` features. Check what
   their testbed used.
4. **Publish interval:** ~1–5 s. Faster publishing fills GRU windows (W=16) sooner. Slower
   publishing means a window covers more time. Choose one rate and record it, because it
   affects how quickly an attack can be detected.
5. **Wi-Fi:** connect to that zone's gateway AP (2.4 GHz; the ESP32 has no 5 GHz). Put the
   credentials in a git-ignored `secrets.h`, with a committed `secrets.h.example`
   (**keys and credentials never in version control**, III-J3).
6. **Robustness:** reconnect automatically on Wi-Fi or MQTT drop, use a watchdog timer, and
   print sensor read failures to serial rather than publishing garbage values.

**Exit:** each zone's ESP32s publish steadily to their gateway's broker for 1 hour, and
`mosquitto_sub -t 'farm/#' -v` on the gateway shows every device with increasing counters.

---

### H4: Raspberry Pi 5 base setup (all four Pis)

1. Flash **Ubuntu Server 24.04 LTS 64-bit** with Raspberry Pi Imager. Pre-set the hostname
   (`gw-a`, `gw-b`, `gw-c`, `agg`), SSH key, and user.
2. `sudo apt update && sudo apt full-upgrade`, then install
   `python3.12-venv git tshark mosquitto mosquitto-clients chrony`.
   Allow the service user to capture without root: `sudo dpkg-reconfigure wireshark-common`
   (answer *yes*), then `sudo usermod -aG wireshark <user>`.
3. **Time sync:** `chrony` on all Pis, with the aggregator as the local NTP server if the lab
   network has no internet. Logs, counters, and latency measurements all need clocks that agree.
4. **Clone and install exactly as CLAUDE.md says:**
   `python3.12 -m venv .venv && ./.venv/bin/python -m pip install -e ".[dev]"`.
   Torch installs as a CPU-only aarch64 wheel from PyPI. If pip tries to build it from source,
   the Python version or architecture is wrong, so stop and check.
5. **Run the full test suite on the Pi.** Pay particular attention to
   `tests/test_ascon_kat.py`: the KAT gate (R5) has to pass on **this** CPU architecture before
   any crypto runs there. It is pure Python, so it should pass, but *should* is not evidence.
6. **Gateway Pis only: Wi-Fi access point.** Using NetworkManager on `wlan0`:
   `nmcli dev wifi hotspot ifname wlan0 band bg ssid farm-zone-a password <demo-password>`.
   Set it to start on boot. Label the password as a demo credential.
   Ethernet (`eth0`) connects to the aggregator LAN. **Do not enable IP forwarding or NAT**
   between `wlan0` and `eth0`. The zone network stays isolated, and only the gateway software
   talks upstream.
7. **Gateway Pis only: Mosquitto.** Listen on the `wlan0` address, and use username/password
   auth even for the demo.
8. Check thermals under load: `vcgencmd measure_temp` (or `/sys/class/thermal`) while training.
   With the active cooler it should stay well below the ~85 °C throttle point.

**Exit:** all four Pis pass the full test suite (KAT included), and every ESP32 can reach its
own zone's broker and nothing else.

---

### H5: Federated training over the real network (Channel 3 on hardware)

At the moment `federated/server.py` runs every client **in-process**. Its docstring notes that
each blob already takes a real serialize → encrypt → decrypt → deserialize trip "as if wired
to a socket". This phase adds the socket.

1. **A thin transport layer, not a new framework.** Add, for example,
   `federated/transport.py`: the aggregator runs a small HTTP server (stdlib `http.server`, or
   FastAPI if the team prefers), and each gateway runs a client. The request and response
   bodies are exactly the `protect_state` output (Ascon ciphertext of the safetensors blob),
   plus the plaintext AD fields `⟨client_id, round_index, direction, schema_version⟩`.
   **Keep `protect_state`/`unprotect_state` and the aggregation functions unchanged.** The
   invariant tests keep guarding them as long as the network layer only moves bytes.
   *Why not Flower (flwr)?* It would replace the round loop, serialization, and aggregation
   that the invariant tests currently cover. A new framework would mean re-proving all of them.
2. **Round protocol** (Algorithm 1, now across machines):
   1. Before round 1, each gateway sends its local scaler sufficient statistics
      `(n, mean, M₂)`. The aggregator combines them with `combine_stats` (Eqs. 23–24) and
      broadcasts the global scaler. **These statistics are not encrypted today,** because
      `scripts/run_phase4.py` computes them in-process. Once they cross a real network, send
      them over the same Ascon channel with a new `direction` value (e.g. `"stats"`). That
      needs a small deviation-addendum note and a test.
   2. For each round r: aggregator → gateways: encrypted global θ. Each gateway trains E local
      epochs and returns encrypted `(θ_k, n_k)`. The aggregator decrypts, checks the AD, runs
      weighted FedAvg (**n_k = sequences**, Eq. 21), and broadcasts again.
   3. A decryption or AD failure raises `WeightIntegrityError`. The aggregator **drops that
      client's update for the round and logs it**. It never averages in an unauthenticated
      blob.
   4. Timeouts: if a gateway misses the round deadline, run FedAvg over the clients that did
      respond (FedAvg's weighting already handles a subset) and record the dropout in the
      manifest.
3. **Keys:** one 16-byte key per gateway, generated on the aggregator with
   `secrets.token_bytes(16)`. Copy each gateway's key over SSH (`scp`) to
   `/etc/asa/keys/<client_id>.key` with mode `0600`, owned by the service user. The aggregator
   holds all three; each gateway holds only its own. **Keys are never committed and never
   written into manifests** (the existing `FederatedServer` rule). Label them as demo keys.
   Production key management stays out of scope.
4. **What data does each gateway train on?** Its **Edge-IIoTset partition** (the H1
   Dirichlet split), stored on the Pi. That keeps Phase 4's results reproducible on hardware.
   Captured testbed traffic is used for **evaluation** (H7), not training, unless the team
   deliberately adds a fine-tuning round with schedule-labelled captures. If you do, it needs
   its own leakage controls and its own writeup.
5. **Equivalence test (the key test of this phase).** A networked run over loopback (all
   processes on one machine) must produce **bit-for-bit the same global model** as the existing
   in-process `FederatedServer` run with the same seeds. That proves the transport changed
   nothing. The Phase 7 checkpoint already reproduced Phase 4's seed-0 result bit-for-bit, so
   there is a known-good reference to compare against.
6. **Measure Eq. (22) on the wire:** bytes per round (from the NIC counters or from the
   transport's own count) against `theoretical_bytes_per_round` plus
   `aead_overhead_bytes_per_round`.

**Exit:** a 20-round federated run across 3 physical gateways and the aggregator completes. The
global model's macro-F1 on the held-out Edge-IIoTset test split falls within the H1 simulation's
mean ± std. The loopback equivalence test is green. Measured bytes per round match the
theoretical figure.

---

### H6: Runtime detection on each gateway

A long-running service on each gateway (a systemd unit, e.g. `asa-gateway.service`, set to
restart automatically) that does:

```
tshark on wlan0 ─► extractor (H2) ─► global scaler ─► streaming per-device windows (W)
   ─► global GRU (latest checkpoint from H5) ─► VerdictRouter (Eq. 5)
         ├─ benign    ─► cloud sink (plaintext + AD metadata, per the 2026-09-20 deviation)
         └─ malicious ─► AlertSink ─► LED + buzzer + local log   (NO reference to cloud transport)
```

1. **Reuse** `sequences/streaming.py`, `routing/router.py`, `routing/alert_sink.py`, and
   `routing/replay_guard.py` unchanged. The new work is (a) the live extractor as the source of
   feature vectors, replacing `FeatureProvenanceAdapter`, and (b) a real `CloudTransport`
   implementation, for example an MQTT or HTTP publisher to a "cloud" process on the
   aggregator or a laptop.
2. **G1 path disjointness must survive the move to hardware.** `AlertSink` gets GPIO access for
   the LED and buzzer (via `gpiozero`, which supports the Pi 5), and **never** a
   `CloudTransport`. `tests/test_path_disjointness.py` stays green. Add a hardware-run check as
   well: during a malicious-only session, the cloud receiver logs **zero** payloads from that
   gateway.
3. **Pairing verdicts with payloads.** The GRU classifies *network traffic*. The router routes
   the *MQTT application payload* (the sensor JSON). A payload is routed using the verdict for
   the window that ends at the packet carrying it, for that device. Write this rule down in the
   module docstring. It is the hardware version of the paper's "two planes" (Section III-H).
4. **Model updates:** the gateway loads a new global checkpoint only between windows, never in
   the middle of one, and logs the round index it is running.
5. **Log** each decision (timestamp, device, window end, verdict, class probabilities, model
   round) to a local JSONL file. H7's evaluation reads these logs.

**Exit:** all three gateways run the service for ≥ 1 hour of normal traffic. Record the
false-positive rate on known-benign traffic. The cloud receiver gets only benign-verdict
payloads.

---

### H7: Attack testbed and field evaluation

**Only run this on the isolated lab network you built, against your own devices.** Never on the
campus network or any network you don't control. Get your guide/lab's sign-off before running
attack tooling.

1. **The attacker is a Kali laptop joined to one zone's Wi-Fi**, acting as the "compromised
   device on the same network" from Channel 1 (Section I-B).
2. **Reproduce the attack classes that can be reproduced safely on this testbed**, using the
   same tool families Edge-IIoTset used:
   - DoS/DDoS: `hping3` SYN / UDP / ICMP floods; an HTTP flood if a web service is running on
     the gateway
   - Reconnaissance: `nmap` port scan, OS fingerprinting (`-O`), vulnerability scan
   - MITM: `ettercap`/`arpspoof` ARP spoofing between an ESP32 and the broker
   - Injection (SQLi/XSS/upload): needs a deliberately vulnerable web app (e.g., DVWA) hosted
     on the gateway. Optional
   - Malware classes (backdoor/ransomware): **skip on hardware.** Report them as
     dataset-evaluated only.
   Evaluate only against the classes that were actually reproduced, and say so in the results.
3. **Labels come from the schedule:** a script runs each attack in a known time window and logs
   start and stop times. Ground truth is "time window × attacker MAC". Keep attack windows
   separated by benign gaps, so windows spanning a label boundary can be identified and excluded
   (the hardware version of the contiguous-run rule).
4. **Report with the same protocol as the paper:** macro-F1, per-class F1, balanced accuracy,
   MCC, confusion matrix, FPR, and binary PR-AUC, over **≥ 3 independent sessions** (the
   hardware counterpart of ≥ 3 seeds), mean ± std. **Report the dataset-test result and the
   hardware-test result side by side.** The gap between them is the distribution shift, and it
   is itself a result.
5. **Systems measurements** (the evidence the paper's C1/C2 claims need): per-window inference
   latency, time from attack start to first alert, CPU/RAM/temperature on the Pi during
   training and during detection, local-round wall time, bytes per round on the wire, and Ascon
   encrypt/decrypt time per blob on the Pi (extend `eval/crypto_benchmark.py` to run there).
   If possible, measure Pi power with a USB-C inline power meter (~₹500–1,000).
6. **Tamper demo for Channel 3:** a script that flips a byte in a weight blob in transit (or
   replays last round's blob) must be rejected with `WeightIntegrityError`, and the round
   continues without that update. This is the visible proof that the Ascon channel works.

**Exit:** a results manifest for the hardware sessions, in the same format as `artifacts/`,
plus a writeup in `results/`.

---

### H8: Documentation and closeout

1. `results/`: a hardware writeup with every figure traceable to a manifest, the same rule as
   the existing results.
2. Design-paper addendum: update it with what the hardware *does* show (live detection of the
   reproduced classes, within the measured distribution shift) and what it does **not** (the
   classes that weren't reproduced, Byzantine/poisoning robustness, production key management,
   radio-level attacks on the ESP32).
3. README: a "Running on hardware" section with a wiring diagram photo, the pinout table, and
   the service setup.
4. CHANGELOG: a dated entry per phase, and hardware rows added to the phase-status table
   (golden rule 6).

---

## 4. Risks and how to spot them early

| Risk | Early warning | Mitigation |
| --- | --- | --- |
| Live features don't match the dataset's features | H2 parity test fails, or the benign FPR on hardware is much higher than on the dataset test split | Fix the extractor until parity holds. Limit Stage 4 to extractable columns |
| Edge-IIoTset CSVs are shuffled, so windows are meaningless | H1 step 5 timestamp check | Extract from the pcaps |
| The model learned IP addresses / host identity | Near-perfect dataset scores, then collapse on the testbed | Drop identifier columns (H1 step 4) and check feature importances |
| Pi throttles during training | `vcgencmd get_throttled` ≠ `0x0` | Active cooler, official 27 W PSU |
| tshark drops packets during floods | tshark's dropped-packet count on exit | Capture fewer fields, use a ring buffer, and treat detection under flood as a measured limit |
| ESP32 ADC2 / 5 V damage | Sensor reads 0 or 4095; a board dies | ADC1 pins only; divider on 5 V outputs |
| Keys or Wi-Fi passwords committed | `git status` shows `secrets.h` or `*.key` | Add `firmware/**/secrets.h`, `*.key`, and `/etc/asa` paths to `.gitignore` in H3/H5 |
| Python 3.12 not available on the Pi image | `python3.12: command not found` | Ubuntu 24.04 (§0.3) |
| CICIoT2023 results break while adding the new profile | Existing tests go red | New profile as a separate config. `default.yaml` untouched |

---

## 5. Order-of-work checklist

- [ ] **H0** Addendum and CLAUDE.md scope update approved. Dataset variant, taxonomy,
      aggregator hardware, and ESP32 crypto decided. Teammate's branch merged
- [ ] Order hardware (§2). *Can happen in parallel with H1, since it doesn't depend on code*
- [ ] **H1** Edge-IIoTset profile: characterisation → dedup → split → selection (extractable
      columns) → knee pinned → Phase 3 gate → Phase 4 gate
- [ ] **H2** Live extractor with a green parity test on the Edge-IIoTset pcaps
- [ ] **H3** ESP32 firmware publishing steadily for 1 h per zone
- [ ] **H4** Four Pis set up, full test suite (KAT) green on aarch64, APs and brokers up
- [ ] **H5** Networked FedAvg over Ascon. Loopback bit-for-bit equivalence. 20 rounds on
      hardware
- [ ] **H6** Gateway detection service. G1 holds on hardware. Benign FPR measured
- [ ] **H7** Attack sessions (≥ 3), hardware-vs-dataset results, systems measurements, tamper
      demo
- [ ] **H8** Results, paper addendum, README, CHANGELOG
