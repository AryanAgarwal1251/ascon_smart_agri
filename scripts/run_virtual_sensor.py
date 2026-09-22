"""Phase 8: a virtual sensor node --- the ESP32 contract from Python (software twin).

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_virtual_sensor.py \\
        --farm farm1 --devices soil01,soil02,soil03 --broker localhost:1883 \\
        --interval 0.5 --messages 300 --attack-after 150 --attack-device soil02

Publishes the same readings the ESP32 sketch (``firmware/esp32_sensor/``) publishes, to the
same topics, with the same ``seq`` counter, so the Pi runtime cannot tell them apart:

    farm/<farm>/sensor/<device>   {"deviceId","temperature","soilMoisture","seq"}
    farm/<farm>/scenario          "attack" | {"deviceId": ..., "mode": "attack"}

The readings come from ``telemetry.simulate.simulate_stream`` (Phase 5). ``--attack-after``
publishes a scenario switch once that many readings have gone out --- for the whole farm, or
for ``--attack-device`` only --- which is how a demonstration stages an attack without the
sensor sending anything different (plan §8 option A).
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import time

from configs.base import load_run_config

from ascon_smart_agri.telemetry.simulate import simulate_stream

print = functools.partial(builtins.print, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--farm", default="farm1")
    parser.add_argument("--devices", default="soil01,soil02,soil03")
    parser.add_argument("--broker", default="localhost:1883")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between readings")
    parser.add_argument("--messages", type=int, default=300, help="total across devices")
    parser.add_argument("--attack-after", type=int, default=0, help="0 = never switch")
    parser.add_argument("--attack-device", default="", help="empty = whole farm")
    parser.add_argument("--seed", type=int, default=-1, help="default: config telemetry seed")
    args = parser.parse_args()

    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    cfg = load_run_config(args.config)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    seed = args.seed if args.seed >= 0 else cfg.telemetry.seed
    host, _, port = args.broker.rpartition(":")

    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=f"virtual-{args.farm}")
    for attempt in range(60):  # the broker container may still be starting
        try:
            client.connect(host, int(port), keepalive=30)
            break
        except OSError as exc:
            if attempt == 59:
                raise SystemExit(f"broker {args.broker} unreachable: {exc}") from exc
            time.sleep(1.0)
    client.loop_start()
    print(f"[virtual-sensor] {devices} -> {args.broker} farm/{args.farm} | every {args.interval}s")

    switched = False
    for message in simulate_stream(
        devices,
        n_messages=args.messages,
        seed=seed,
        schema_version=cfg.telemetry.schema_version,
        temperature_range_c=tuple(cfg.telemetry.temperature_range_c),
        soil_moisture_range_pct=tuple(cfg.telemetry.soil_moisture_range_pct),
    ):
        if args.attack_after and not switched and message.stream_index >= args.attack_after:
            payload = (
                json.dumps({"deviceId": args.attack_device, "mode": "attack"})
                if args.attack_device
                else "attack"
            )
            client.publish(f"farm/{args.farm}/scenario", payload, qos=1)
            switched = True
            print(f"[virtual-sensor] scenario -> attack ({args.attack_device or 'whole farm'})")
        body = {**message.payload, "seq": message.counter}
        client.publish(f"farm/{args.farm}/sensor/{message.device_id}", json.dumps(body), qos=1)
        time.sleep(args.interval)

    client.loop_stop()
    client.disconnect()
    print(f"[virtual-sensor] published {args.messages} readings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
