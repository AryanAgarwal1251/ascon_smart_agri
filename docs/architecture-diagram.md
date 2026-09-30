# Architecture and connection diagram

Everything in one place: the logical topology with the cryptography on each hop, the physical
ESP32 wiring, and which container in the software twin stands in for which piece of hardware.
Pins and topics below are read from [`firmware/esp32_sensor/`](../firmware/esp32_sensor/), not
transcribed from memory.

---

## 1. Logical topology

```
        FARM 1                                      FARM 2
 ┌────────────────────┐                      ┌────────────────────┐
 │ ESP32  soil01      │                      │ ESP32  soil04      │
 │ ESP32  soil02      │                      │ ESP32  soil05      │
 │ ESP32  soil03      │                      │ ESP32  soil06      │
 └─────────┬──────────┘                      └─────────┬──────────┘
           │  MQTT, plain, farm-local                  │
           │  farm/farm1/sensor/<device>               │
           ▼                                           ▼
 ┌────────────────────┐                      ┌────────────────────┐
 │  Raspberry Pi #1   │                      │  Raspberry Pi #2   │
 │  ├ Mosquitto       │                      │  ├ Mosquitto       │
 │  ├ LOCAL GRU       │                      │  ├ LOCAL GRU       │
 │  └ runtime pipeline│                      │  └ runtime pipeline│
 └───┬────────────┬───┘                      └───┬────────────┬───┘
     │            │                              │            │
     │ Ascon-     │ TLS                          │ Ascon-     │ TLS
     │ AEAD128    │ (benign only)                │ AEAD128    │ (benign only)
     │ both ways  │                              │ both ways  │
     ▼            ▼                              ▼            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                          LAPTOP                                 │
 │  ┌───────────────────────┐      ┌──────────────────────────┐    │
 │  │ AGGREGATOR :7700      │      │ CLOUD RECEIVER :8443     │    │
 │  │ = MASTER GRU          │      │ TLS, replay-guarded      │    │
 │  │ weighted FedAvg       │      │ receives BENIGN only     │    │
 │  └───────────────────────┘      └──────────────────────────┘    │
 │  ┌───────────────────────┐                                      │
 │  │ sim-3 (3rd client)    │  so K = 3 without a third Pi         │
 │  └───────────────────────┘                                      │
 └─────────────────────────────────────────────────────────────────┘
```

**Malicious readings never appear above.** That is the point: a malicious verdict is raised as
a local alert on the Pi and goes nowhere else. The alert sink structurally holds no reference to
the cloud transport (gap G1, `tests/test_path_disjointness.py`).

---

## 2. What crosses each hop, and what protects it

| Hop | Payload | Protection | Why |
| --- | --- | --- | --- |
| ESP32 → Pi | sensor JSON | **none** (plain MQTT) | farm-local network; readings never leave the farm on this hop |
| Pi ⇄ Laptop aggregator | **model weights**, safetensors | **Ascon-AEAD128**, per-client per-direction key, AD `⟨client_id, round, direction, schema_version⟩` | this is the crossing that traverses the untrusted IoT network, in **both** directions |
| Pi → Laptop cloud receiver | benign sensor readings | **TLS** | one cipher per hop; the cloud is an ordinary HTTPS endpoint |
| Pi → local alert | malicious readings | stays on the Pi | G1: zero malicious payloads on the cloud path |

**Ascon protects the weights, not the telemetry.** That is the architecture's security claim.
The Pi → cloud hop is plain TLS by deliberate decision (2026-09-19): one cipher per hop, as
when pushing to S3.

---

## 3. The two planes, and the G6 boundary

The system runs two loops that must not be confused.

```
TRAINING PLANE (periodic)                RUNTIME PLANE (continuous)
─────────────────────────                ──────────────────────────
CICIoT2023 train split                   ESP32 JSON arrives at the Pi
  ↓ Dirichlet(α=0.5), 3 farms              ↓
each Pi trains its local GRU             FeatureProvenanceAdapter  ← G6 BOUNDARY
  ↓ Ascon-sealed weights                   ↓ pairs the message with a
MASTER GRU averages (FedAvg)               HELD-OUT CICIoT2023 record
  ↓ Ascon-sealed weights back              ↓
each Pi's GRU is updated                 local GRU classifies (W = 16 window)
                                           ↓
                                         benign → TLS → cloud
                                         malicious → local alert
```

**The G6 boundary is a declared limitation, not a detail.** Network features come only from
held-out CICIoT2023 records — never from the JSON payload, never synthesised:

| Data | Where it comes from | What it is used for |
| --- | --- | --- |
| Training rows | `phase4_cache.npz` **train** split, 1,237,911 rows | training the local GRUs |
| Runtime/test rows | `phase4_cache.npz` **test** split, 311,565 rows, R3-held-out | what the GRU actually classifies |
| ESP32 JSON | the sensor | the application payload; **never parsed into features** |

The `farm/<farm>/scenario` topic only selects **which held-out pool** the adapter draws from
(benign records or malicious records). It never synthesises a feature value.

What this demonstrates is architectural correctness. It does **not** demonstrate that a
CICIoT2023-trained model detects attacks on a live agricultural deployment — that needs capture
and feature re-extraction on the target network. See
[`telemetry/provenance.py`](../src/ascon_smart_agri/telemetry/provenance.py).

---

## 4. Physical wiring — one ESP32 node (×6)

From [`firmware/esp32_sensor/diagram.json`](../firmware/esp32_sensor/diagram.json), exactly:

```
        ESP32 DevKitC V4
      ┌───────────────────┐
      │                   │
      │  3V3 ●────────────┼──── red ─────┬──● VCC   DHT22
      │                   │              │
      │ GND1 ●────────────┼──── black ───┼──● GND   (temperature + humidity)
      │                   │              │
      │   15 ●────────────┼──── green ───┴──● SDA
      │                   │
      │  3V3 ●────────────┼──── red ─────┬──● VCC   Capacitive soil
      │                   │              │         moisture sensor
      │ GND2 ●────────────┼──── black ───┼──● GND
      │                   │              │
      │   34 ●────────────┼──── orange ──┴──● SIG   (analogue out)
      │                   │
      └───────────────────┘
```

| Signal | ESP32 pin | Wire | Note |
| --- | --- | --- | --- |
| DHT22 data | **GPIO 15** | green | needs a 10 kΩ pull-up; breakout modules include it |
| DHT22 power / ground | 3V3 / GND.1 | red / black | |
| Soil moisture analogue | **GPIO 34** | orange | **ADC1** — correct, because ADC2 is unusable while Wi-Fi is active |
| Soil moisture power / ground | 3V3 / GND.2 | red / black | use a **capacitive** sensor; resistive probes corrode |

Firmware config per board: `WIFI_SSID`, `WIFI_PASS`, `BROKER` = that farm's Pi IP, `FARM` =
`farm1`/`farm2`, `DEVICE` = `soil01`…`soil06`.

---

## 5. The sensor contract

Both the real sketch and `scripts/run_virtual_sensor.py` publish this, and `asa pi-runtime`
cannot tell them apart:

```
publish   farm/<farm>/sensor/<device>
          { "deviceId": "soil01", "temperature": 24.9,
            "soilMoisture": 41.2, "seq": 137 }

subscribe farm/<farm>/scenario
          "attack" | "benign"
          or { "deviceId": "soil02", "mode": "attack" }
```

`seq` is monotonic per device and is what the replay guard checks. Out-of-order, duplicate,
malformed and foreign-farm messages are dropped (`tests/test_mqtt_source.py`).

---

## 6. Twin mapping — which container is which piece of hardware

| Hardware | Twin service | Command |
| --- | --- | --- |
| ESP32 ×3, farm 1 | `sensors-farm1` | `asa virtual-sensor --devices soil01,soil02,soil03` |
| ESP32 ×3, farm 2 | `sensors-farm2` | `asa virtual-sensor --devices soil04,soil05,soil06` |
| Mosquitto on Pi #1 / #2 | `broker-farm1`, `broker-farm2` | `eclipse-mosquitto:2` |
| Raspberry Pi #1 / #2 | `pi-1`, `pi-2` | `asa client-node` **then** `asa pi-runtime` |
| Simulated 3rd client | `sim-3` | `asa client-node --client-id sim-3` |
| Laptop master GRU | `aggregator` | `asa aggregator --port 7700` |
| Cloud endpoint | `cloud-receiver` | `asa cloud-receiver --port 8443` |
| Key/cert provisioning | `init-secrets` | runs once, exits 0 |

Images are built `linux/arm64` so the stack is exercised on the Pi's architecture. **Container
timings are never Pi timings** — every manifest is tagged `platform=docker-arm64-simulation`.

### Ports

| Port | Service | Protocol |
| --- | --- | --- |
| 1883 | each farm's Mosquitto | MQTT, plain, farm-local |
| 7700 | aggregator | TCP, **Ascon-AEAD128 frames** |
| 8443 | cloud receiver | **HTTPS/TLS** |

---

## 7. Software needed

| For | Tool | Cost |
| --- | --- | --- |
| The whole twin | **Docker** + Compose | free |
| Flashing ESP32 | **Arduino IDE** + ESP32 board package | free |
| ESP32 libraries | `PubSubClient`, `DHT sensor library`, `ArduinoJson` | free |
| Pi OS onto SD | **Raspberry Pi Imager** | free |
| ESP32 without a board | **Wokwi** (browser) | free |

Nothing paid, and nothing beyond Docker is needed for the software twin.

---

## 8. Moving to hardware

No code changes. Per node: install on **Python ≥ 3.12** (Raspberry Pi OS Trixie or Ubuntu 24.04
— **not Bookworm**, which ships 3.11 and cannot install the stack), copy the demo key and cloud
certificate, run Mosquitto on each Pi, flash the ESP32s with the Pi's IP as `BROKER`, and pass
`--platform raspberry-pi-<model>` so the manifests record what they ran on.

Components: [plan §7a](phase8-hardware-federation.md). Verified software results:
[`results/phase8_software_twin.md`](../results/phase8_software_twin.md).
