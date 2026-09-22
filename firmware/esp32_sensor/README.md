# ESP32 sensor node (Phase 8)

The sensor contract, as firmware. Same topics and JSON as `scripts/run_virtual_sensor.py`;
the Pi runtime (`asa pi-runtime`) cannot tell the two apart.

## Simulate on Wokwi (free, browser)

1. Go to <https://wokwi.com>, **New project → ESP32**.
2. Replace the sketch with `esp32_sensor.ino`, and `diagram.json` with this folder's.
3. Add the three libraries from `libraries.txt` (Library Manager tab).
4. Start. The DHT22 slider sets temperature/humidity; the potentiometer is soil moisture.
   Readings go to `farm/farm1/sensor/soil01` on `test.mosquitto.org` (Wokwi's free gateway
   reaches the internet, not your LAN).
5. On the laptop, point the Pi runtime at the same public broker:
   `asa pi-runtime --client-id pi-1 --farm farm1 --broker test.mosquitto.org:1883 ...`
6. Stage an attack from any MQTT client:
   `mosquitto_pub -h test.mosquitto.org -t farm/farm1/scenario -m attack`

Six nodes = six Wokwi tabs with `DEVICE` set to `soil01..soil03` (farm1) and `soil04..soil06`
(farm2). A public broker is a simulation convenience; it is not the farm-local property.

## On the hardware

Set `WIFI_SSID`/`WIFI_PASS` to the farm Wi‑Fi and `BROKER` to the Pi's IP (each Pi runs its
own Mosquitto). Flash with the Arduino IDE (ESP32 board package). Wiring as in
`diagram.json`: DHT22 data → GPIO 15, soil-moisture probe analogue out → GPIO 34.
