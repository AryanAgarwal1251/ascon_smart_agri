# Phase 8 software twin — simulate the whole hardware setup before buying it

**Status (2026-09-19): implemented; the compose run needs Docker on your machine.** Everything
below runs as local processes too (`tests/test_software_twin.py::test_headless_twin_end_to_end`
does exactly that, in ~30 s, whenever `artifacts/phase4_cache.npz` is present).

The user was asked to simulate the entire hardware architecture in software first. No single
free online tool simulates a Linux Raspberry Pi and an ESP32 together, so the setup is split
into three layers, each free:

| Layer | Tool | What it proves | What it cannot prove |
| --- | --- | --- | --- |
| ESP32 sensor nodes | [Wokwi](https://wokwi.com) (browser, free) with `firmware/esp32_sensor/` | the real sketch, real Wi‑Fi/MQTT stack, virtual DHT22 + moisture probe | LAN reachability (free gateway → public broker only) |
| Pis + laptop + cloud | **Docker Compose** (`docker-compose.yml`, arm64 images) | the exact repo code as separate processes on a real (virtual) network; aarch64 install; sealed frames on the wire; TLS to the cloud | **Pi timing** — containers run far faster than a Pi 4 |
| Pi OS image | QEMU aarch64 `virt` + Ubuntu 24.04 (optional) | the OS-image install path | nothing the Docker layer doesn't, except the image itself |

Every manifest carries `platform` (`docker-arm64-simulation`, `pytest-headless-twin`,
`raspberry-pi-5`, …) so a simulation number is never quoted as a hardware number.

## 1. The sensor contract

The ESP32 is a *contract*, not a component. The Pi runtime cannot tell these apart:

```
topic    farm/<farm_id>/sensor/<device_id>
payload  {"deviceId": "soil01", "temperature": 24.8, "soilMoisture": 42.5, "seq": 17}
topic    farm/<farm_id>/scenario      "benign" | "attack" | {"deviceId": "...", "mode": "attack"}
```

- `firmware/esp32_sensor/esp32_sensor.ino` — the real node (Wokwi or hardware).
- `scripts/run_virtual_sensor.py` (`asa virtual-sensor`) — the same contract from Python.
- `telemetry/mqtt_source.py` — the Pi side: parses, orders, drops replays/malformed messages,
  tracks each device's scenario.

The scenario topic changes nothing the node sends. It tells the Pi's provenance adapter
(`telemetry/provenance.py`, `scenario=`) which **held-out** record pool to pair the device's
readings with — benign or attack — a selection over real CICIoT2023 records, never
synthesis. The G6 boundary is unchanged. (Plan §8 option A.)

## 2. The topology (`docker-compose.yml`)

```
init-secrets ──► keys/ (Ascon channel keys)  certs/ (self-signed demo TLS cert)
aggregator      asa aggregator          :7700   sealed weight rounds, weighted FedAvg
cloud-receiver  asa cloud-receiver      :8443   HTTPS, per-device replay guard
broker-farm1    mosquitto               :1883   farm 1's own broker
broker-farm2    mosquitto               :1883   farm 2's own broker
pi-1            asa client-node ‖ asa pi-runtime   farm1: soil01..03  → cloud / local alert
pi-2            asa client-node ‖ asa pi-runtime   farm2: soil04..06
sim-3           asa client-node                    third client (laptop) so K = 3
sensors-farm1   asa virtual-sensor      soil01..03, soil02 switched to attack at message 120
sensors-farm2   asa virtual-sensor      soil04..06, whole farm switched to attack at 120
```

Readings never traverse the laptop (each Pi has its own broker); weights cross only as
Ascon frames; benign readings cross only over TLS; malicious readings stop at the Pi.

```bash
docker compose up -d --build        # defaults: ROUNDS=2 EPOCHS=1 SEQ_CAP=5000
docker compose logs -f pi-1 pi-2    # the two interesting logs
docker compose down -v              # when the run is done
ROUNDS=5 SEQ_CAP=20000 MESSAGES=600 docker compose up --build
```

Requires `artifacts/phase4_cache.npz`. Manifests land in `artifacts/manifest_phase8_*.json`.
On an x86 laptop: `TWIN_PLATFORM=linux/amd64` (the aarch64 check is then lost).

## 3. The same thing without Docker

```bash
asa aggregator --generate-keys                     # keys/phase8.demo.key (gitignored)
asa aggregator --port 7700 --rounds 2 &
asa client-node --client-id pi-1 --client-index 0 --server 127.0.0.1:7700 --sequence-cap 5000 &
asa client-node --client-id pi-2 --client-index 1 --server 127.0.0.1:7700 --sequence-cap 5000 &
asa client-node --client-id sim-3 --client-index 2 --server 127.0.0.1:7700 --sequence-cap 5000
asa cloud-receiver --host 127.0.0.1 --port 8443 --edges pi-1 &      # plain HTTP without --cert
asa pi-runtime --client-id pi-1 --source sim --messages 240 --attack-after 120 \
    --cloud http://127.0.0.1:8443
```

With a local Mosquitto (`brew install mosquitto`), replace `--source sim` by
`--broker localhost:1883 --farm farm1` and run `asa virtual-sensor --farm farm1 …` — or a
Wokwi ESP32 against a public broker (`firmware/esp32_sensor/README.md`).

## 4. What was verified on 2026-09-19 (laptop, loopback, `platform=laptop-loopback-smoke`)

- Sealed federation: aggregator + pi-1 + pi-2 + sim-3, round 0 scaler + 1 weight round,
  136,410 B per sealed frame, 0 rejections.
- Runtime: 240 simulated readings on pi-1 with the scenario switched at 120 → 45 buffering
  (W = 16 × 3 devices), 76 benign → cloud accepted **76**, 119 malicious → **119** local
  alerts, **0** malicious readings at the cloud, receiver rejected 0. (Model: 1 round on
  3,000 sequences — the verdict quality is not the point of a smoke run.)

## 5. When you move to hardware

Nothing in the code changes. Per node: install (Python ≥ 3.12, Ubuntu 24.04 / Pi OS Trixie),
copy `keys/phase8.demo.key` and `keys/cloud.demo.pem`, run `mosquitto` on each Pi, flash the
ESP32s with the Pi's IP as `BROKER`, and pass `--platform raspberry-pi-<model>` so the
manifests say so. The per-round `round_seconds` in `manifest_phase8_client_<id>.json` are
then the real Pi feasibility numbers plan §7 asks for.
