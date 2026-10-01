# Breadboard wiring — one sensor node, wire by wire

Build this **six times** (`soil01`–`soil03` on farm 1, `soil04`–`soil06` on farm 2). The pin
assignments come from [`firmware/esp32_sensor/diagram.json`](../firmware/esp32_sensor/diagram.json)
and the sketch itself, so the firmware works against this layout unchanged.

---

## Before you start: the ESP32 is wider than a breadboard

The **ESP32 DevKitC V4** is 38 pins and about 25.5 mm wide. On a standard 830-point breadboard
it covers the centre channel and leaves only **one usable hole per pin row** on each side —
enough for this build, but cramped.

Two ways round it, pick either:

- **Two half breadboards pushed together**, ESP32 straddling the join → full access both sides.
  This is the comfortable option.
- **One full breadboard**, accept one hole per pin. Works for our three wires per side.

The diagrams below assume the single-board version, because we only need six wires total.

---

## Components per node

| # | Component | Pins it has |
| --- | --- | --- |
| 1 | ESP32 DevKitC V4 | 38 |
| 2 | DHT22 — **3-pin module** (recommended) | `VCC` · `DATA` · `GND` |
| 2b | DHT22 — bare AM2302 sensor (alternative) | `1=VCC` · `2=DATA` · `3=NC` · `4=GND` |
| 3 | Capacitive soil moisture sensor v1.2/v2.0 | `VCC` · `GND` · `AOUT` |

**Buy the 3-pin DHT22 module, not the bare sensor.** The module has the 10 kΩ pull-up resistor
already on its PCB. With the bare AM2302 you must add that resistor yourself (shown below).

**The soil sensor must be capacitive, not resistive.** Resistive probes corrode within weeks in
wet soil. Capacitive boards are marked "Capacitive Soil Moisture Sensor v1.2" or "v2.0".

---

## The six wires

| # | From | To | Colour | Carries |
| --- | --- | --- | --- | --- |
| 1 | ESP32 `3V3` | breadboard **+ rail** (red) | red | 3.3 V |
| 2 | ESP32 `GND` | breadboard **− rail** (blue/black) | black | ground |
| 3 | DHT22 `VCC` | **+ rail** | red | 3.3 V |
| 4 | DHT22 `GND` | **− rail** | black | ground |
| 5 | **DHT22 `DATA`** | **ESP32 `GPIO 15`** | green | temperature/humidity data |
| 6 | Soil `VCC` | **+ rail** | red | 3.3 V |
| 7 | Soil `GND` | **− rail** | black | ground |
| 8 | **Soil `AOUT`** | **ESP32 `GPIO 34`** | orange | analogue moisture 0–3.3 V |

Two signal wires (5 and 8) and the rest is power. That is the whole node.

---

## Breadboard layout

```
                         + RAIL  (red)   ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                         − RAIL  (blue)  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
                                           │ │          │ │        │  │
                      ┌────────────────────┘ │          │ │        │  │
                      │   ┌──────────────────┘          │ │        │  │
                      │   │                             │ │        │  │
       ┌──────────────┴───┴──────────────┐              │ │        │  │
       │   ESP32 DevKitC V4              │              │ │        │  │
       │                                 │              │ │        │  │
       │  3V3 ●──(1) red ──→ + RAIL      │              │ │        │  │
       │  GND ●──(2) black ─→ − RAIL     │              │ │        │  │
       │                                 │              │ │        │  │
       │   15 ●─────── (5) green ────────┼──────────────┘ │        │  │
       │                                 │                │        │  │
       │   34 ●─────── (8) orange ───────┼────────────────┼────────┘  │
       │                                 │                │           │
       └─────────────────────────────────┘                │           │
                                                          │           │
                   ┌──────────────────────┐               │           │
                   │   DHT22  (3-pin)     │               │           │
                   │  VCC ●──(3) red ─────┼──→ + RAIL     │           │
                   │ DATA ●───────────────┼───────────────┘           │
                   │  GND ●──(4) black ───┼──→ − RAIL                 │
                   └──────────────────────┘                           │
                                                                      │
                   ┌──────────────────────┐                           │
                   │ Capacitive soil      │                           │
                   │ moisture v1.2        │                           │
                   │  VCC ●──(6) red ─────┼──→ + RAIL                 │
                   │  GND ●──(7) black ───┼──→ − RAIL                 │
                   │ AOUT ●───────────────┼───────────────────────────┘
                   └──────────────────────┘
```

### If you bought the bare AM2302 instead of the module

Add one 10 kΩ resistor between `VCC` and `DATA`:

```
        + RAIL ───┬──────────────● VCC  (pin 1)
                  │
                 [10 kΩ]
                  │
                  ├──────────────● DATA (pin 2) ──── green ──→ ESP32 GPIO 15
                  │
                  ·                NC   (pin 3) ──── leave unconnected
        − RAIL ──────────────────● GND  (pin 4)
```

---

## Why these two pins specifically

| Pin | Why it, and not another |
| --- | --- |
| **GPIO 15** | Ordinary digital GPIO, free at boot, no conflict with flashing. Any free digital pin would do — but the firmware hard-codes 15 (`#define DHT_PIN 15`). |
| **GPIO 34** | **Must be an ADC1 pin.** GPIO 34 is input-only and on ADC1. **ADC2 stops working entirely while Wi-Fi is active on the ESP32** — and this node is always on Wi-Fi, so an ADC2 pin would silently read garbage. |

If you ever move the moisture sensor, keep it on **ADC1**: GPIO 32, 33, 34, 35, 36, 39.

---

## Power sanity checks

- Everything runs at **3.3 V**. Do **not** wire the sensors to `5V`/`VIN` — the ESP32's GPIOs
  are 3.3 V and GPIO 34 would see an out-of-range analogue level.
- The ESP32 has several `GND` pins; any of them works. The Wokwi diagram uses two different
  ones (`GND.1`, `GND.2`) purely for tidiness — a shared rail is equivalent.
- Power the board over **USB** while testing. Six nodes on a powered USB hub is convenient for
  flashing.

---

## Per-board firmware configuration

Each of the six boards needs four values changed in
[`esp32_sensor.ino`](../firmware/esp32_sensor/esp32_sensor.ino):

```c
const char* WIFI_SSID = "<your farm Wi-Fi>";
const char* WIFI_PASS = "<password>";
const char* BROKER    = "192.168.1.50";   // THAT farm's Raspberry Pi IP
const char* FARM      = "farm1";          // farm1 or farm2
const char* DEVICE    = "soil01";         // soil01..soil03 / soil04..soil06
```

| Board | FARM | DEVICE | BROKER |
| --- | --- | --- | --- |
| 1 | `farm1` | `soil01` | Pi #1 IP |
| 2 | `farm1` | `soil02` | Pi #1 IP |
| 3 | `farm1` | `soil03` | Pi #1 IP |
| 4 | `farm2` | `soil04` | Pi #2 IP |
| 5 | `farm2` | `soil05` | Pi #2 IP |
| 6 | `farm2` | `soil06` | Pi #2 IP |

Libraries to install in the Arduino IDE (also listed in
[`libraries.txt`](../firmware/esp32_sensor/libraries.txt)): `PubSubClient`,
`DHT sensor library`, `ArduinoJson`.

---

## Testing it without any hardware first

The board layout above has a **free browser simulation** — the project ships the Wokwi diagram:

1. Open <https://wokwi.com> → New project → ESP32
2. Paste [`esp32_sensor.ino`](../firmware/esp32_sensor/esp32_sensor.ino), and replace
   `diagram.json` with [this one](../firmware/esp32_sensor/diagram.json)
3. Add the three libraries in the Library Manager tab
4. Start it. The DHT22 slider sets temperature; the potentiometer stands in for the soil probe.

**The one limitation:** Wokwi's free gateway reaches the public internet, **not your LAN**, so a
Wokwi board cannot reach a Mosquitto broker running on your laptop or Pi. For a simulated
end-to-end run, point both sides at a public broker:

```bash
# in the sketch
const char* BROKER = "test.mosquitto.org";

# on the laptop
asa pi-runtime --client-id pi-1 --farm farm1 --broker test.mosquitto.org:1883 \
    --cloud https://127.0.0.1:8443 --ca-cert keys/cloud.demo.pem

# trigger the attack scenario from anywhere
mosquitto_pub -h test.mosquitto.org -t farm/farm1/scenario -m attack
```

A public broker is a simulation convenience only. On the real farm the broker runs **on the
Pi**, and readings never leave the farm on that hop.

---

## What the node actually sends

```
publish   farm/<farm>/sensor/<device>   every INTERVAL_MS
          { "deviceId": "soil01", "temperature": 24.9,
            "soilMoisture": 41.2, "seq": 137 }

subscribe farm/<farm>/scenario
          "attack" | "benign"
```

`soilMoisture` is `analogRead(34) * 100.0 / 4095.0`, i.e. a 0–100 % scale straight off the ADC.
`seq` increments per device and is what the Pi's replay guard checks.

Full topology and the rest of the architecture:
[`architecture-diagram.md`](architecture-diagram.md). Components to buy:
[plan §7a](phase8-hardware-federation.md).
