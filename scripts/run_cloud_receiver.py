"""Phase 8: the demo cloud ingest endpoint (plan §1, §4). Runs on the laptop.

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_cloud_receiver.py \\
        --host 0.0.0.0 --port 8443 --cert keys/cloud.demo.pem --key keys/cloud.demo.key \\
        --edges pi-1,pi-2,sim-3

Accepts benign readings POSTed by each Pi's runtime over TLS (``routing/tls_path.py``),
enforces the per-device monotonic counter, and never sees a malicious reading --- those
stop at the Pi's alert sink. Runs until ``--seconds`` elapse (0 = until interrupted), then
writes ``artifacts/manifest_phase8_cloud_receiver.json`` with what it accepted and rejected.

``--generate-cert`` writes a self-signed DEMO certificate/key pair with ``openssl`` and exits.
The pair is gitignored (``*.pem``, ``*.key``); copy the ``.pem`` to each Pi as ``--ca-cert``.
Without ``--cert`` the receiver speaks plain HTTP, for tests and for the compose twin's
internal network; the hardware demo uses TLS.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import subprocess
import time
from pathlib import Path

from configs.base import load_run_config

from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.routing.tls_path import CloudReceiverServer

print = functools.partial(builtins.print, flush=True)


def generate_demo_cert(cert: Path, key: Path, *, hostname: str) -> None:
    """Self-signed DEMO certificate via the openssl CLI (never committed; III-J3)."""
    cert.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30",
            "-keyout", str(key), "-out", str(cert),
            "-subj", f"/CN={hostname}/O=ascon-smart-agri DEMO",
            "-addext", f"subjectAltName=DNS:{hostname},DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", default="", help="PEM certificate; omit for plain HTTP")
    parser.add_argument("--key", default="", help="PEM private key")
    parser.add_argument("--edges", default="pi-1,pi-2,sim-3", help="edge ids allowed to post")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = run until Ctrl-C")
    parser.add_argument("--generate-cert", action="store_true")
    parser.add_argument("--cert-hostname", default="cloud-receiver")
    parser.add_argument("--platform", default="unspecified")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    if args.generate_cert:
        cert, key = (
            Path(args.cert or "keys/cloud.demo.pem"),
            Path(args.key or "keys/cloud.demo.key"),
        )
        generate_demo_cert(cert, key, hostname=args.cert_hostname)
        print(f"[cert] wrote self-signed DEMO certificate {cert} + key {key} (not in git)")
        return 0

    edges = frozenset(e.strip() for e in args.edges.split(",") if e.strip())
    server = CloudReceiverServer(
        args.host,
        args.port,
        certfile=Path(args.cert) if args.cert else None,
        keyfile=Path(args.key) if args.key else None,
        known_edges=edges,
    )
    server.start()
    print(
        f"[cloud] {'TLS' if server.tls else 'plain HTTP'} receiver on {server.host}:{server.port}"
        f" | edges {sorted(edges)} | replay-guarded per device"
    )
    started = time.time()
    last = -1
    try:
        while args.seconds <= 0 or time.time() - started < args.seconds:
            time.sleep(1.0)
            if server.state.accepted != last:
                last = server.state.accepted
                print(f"  accepted {server.state.accepted} | rejected {server.state.rejected}")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name="phase8_cloud_receiver",
        seeds=[cfg.data.seed],
        library_versions=versions,
        hardware=hardware,
        config_snapshot=cfg.model_dump(mode="json"),
        results={
            "transport": "tls" if server.tls else "plain-http",
            "known_edges": sorted(edges),
            "accepted": server.state.accepted,
            "rejected": server.state.rejected,
            "rejections": server.state.rejections,
            "per_edge_accepted": {
                e: sum(1 for r in server.state.readings if r["edgeId"] == e) for e in sorted(edges)
            },
            "elapsed_seconds": round(time.time() - started, 1),
            "platform": args.platform,
        },
    )
    out = manifest.write(Path(cfg.output_dir))
    print(f"[cloud] done | accepted {server.state.accepted} | manifest {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
