"""The benign path over plain TLS (Phase 8, decided 2026-09-19; plan §1, §4).

Ascon is the project's security claim on the *weight channel* (``federated/transport.py``).
The Pi -> cloud hop carries benign sensor readings and uses ordinary TLS, one cipher per hop,
as any real deployment would. The cloud receives the reading, never a verdict; a malicious
reading is dropped and alerted locally. This module replaces ``routing/router.py`` +
``routing/cloud_sink.py`` (the Phase 6/7 Ascon-on-telemetry path, kept for its tests and
history) in the Phase 8 runtime.

What is preserved, structurally: **path disjointness (G1).** :class:`TlsVerdictRouter` is the
only object holding the cloud transport; :class:`~.alert_sink.AlertSink` is never given one.
``tests/test_tls_path.py`` asserts that a malicious-only run emits nothing to the transport.

What is preserved, behaviourally: the receiver's per-device monotonic-counter replay check
(:class:`~.replay_guard.ReplayGuard`) --- on this hop the counter is authenticated by TLS
rather than by an AEAD tag, which is sufficient once the transport is trusted end to end.

The receiver is a small ``http.server`` handler so the demo has no web-framework dependency;
it is a demo endpoint standing in for a cloud ingest API (the paper's A6), not a product.
"""

from __future__ import annotations

import json
import socket
import ssl
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .alert_sink import AlertSink
from .replay_guard import ReplayGuard


@dataclass(frozen=True)
class ReadingEnvelope:
    """What the benign path sends: the reading plus the identity/replay metadata."""

    edge_id: str
    device_id: str
    counter: int
    schema_version: str
    payload: dict[str, Any]

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "edgeId": self.edge_id,
                "deviceId": self.device_id,
                "counter": self.counter,
                "schemaVersion": self.schema_version,
                "payload": self.payload,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> ReadingEnvelope:
        body = json.loads(data.decode("utf-8"))
        return cls(
            edge_id=str(body["edgeId"]),
            device_id=str(body["deviceId"]),
            counter=int(body["counter"]),
            schema_version=str(body["schemaVersion"]),
            payload=dict(body["payload"]),
        )


@runtime_checkable
class PlainCloudTransport(Protocol):
    """The TLS-hop transport interface. The malicious path must never hold one of these."""

    def send_reading(self, envelope: ReadingEnvelope) -> None:
        """Transmit one benign reading toward the cloud receiver."""
        del envelope  # interface declaration


class TlsVerdictRouter:
    """Eq. (5) routing: benign -> cloud transport, malicious -> alert sink. Disjoint."""

    def __init__(self, cloud: PlainCloudTransport, alert: AlertSink) -> None:
        self._cloud = cloud  # lives ONLY here; never handed to `alert`
        self._alert = alert

    def route(self, *, verdict_benign: bool, envelope: ReadingEnvelope) -> None:
        if verdict_benign:
            self._cloud.send_reading(envelope)
        else:
            self._alert.raise_alert(
                envelope.edge_id,
                f"classifier verdict: malicious (device={envelope.device_id}, "
                f"counter={envelope.counter})",
            )


class InMemoryCloudTransport:
    """Test/demo transport: records what the benign path would have sent."""

    def __init__(self) -> None:
        self.sent: list[ReadingEnvelope] = []

    def send_reading(self, envelope: ReadingEnvelope) -> None:
        self.sent.append(envelope)


class HttpsCloudClient:
    """POSTs envelopes to the receiver over TLS (or plain HTTP when ``url`` says so)."""

    def __init__(self, url: str, *, ca_cert: Path | None = None, timeout_s: float = 10.0) -> None:
        self.url = url.rstrip("/") + "/readings"
        self.timeout_s = timeout_s
        self.sent = 0
        self.failed = 0
        self.last_error: str | None = None
        if ca_cert is not None:
            # Trust exactly the demo certificate, never the system store: a self-signed demo
            # cert is not a CA, and the demo must not silently accept any public endpoint.
            self._context: ssl.SSLContext | None = ssl.create_default_context(cafile=str(ca_cert))
        else:
            self._context = None

    def send_reading(self, envelope: ReadingEnvelope) -> None:
        request = urllib.request.Request(
            self.url,
            data=envelope.to_json(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_s, context=self._context
            ) as response:
                if response.status != 202:
                    raise urllib.error.HTTPError(
                        self.url, response.status, "unexpected status", response.headers, None
                    )
            self.sent += 1
        except (urllib.error.URLError, OSError) as exc:
            # Fire-and-forget from the router's point of view; the loss is logged, not raised.
            self.failed += 1
            self.last_error = str(exc)


@dataclass
class CloudReceiverState:
    """What the receiver has accepted and rejected; written to its manifest."""

    accepted: int = 0
    rejected: int = 0
    rejections: list[str] = field(default_factory=list)
    readings: list[dict[str, Any]] = field(default_factory=list)
    replay_guard: ReplayGuard = field(default_factory=ReplayGuard)
    known_edges: frozenset[str] = frozenset()
    lock: threading.Lock = field(default_factory=threading.Lock)

    def ingest(self, raw: bytes) -> tuple[int, str]:
        """Validate one POST body; returns ``(http_status, reason)``."""
        try:
            envelope = ReadingEnvelope.from_json(raw)
        except (ValueError, KeyError, TypeError) as exc:
            return self._reject(400, f"malformed envelope: {exc}")
        if self.known_edges and envelope.edge_id not in self.known_edges:
            return self._reject(403, f"unknown edge {envelope.edge_id!r}")
        with self.lock:
            fresh = self.replay_guard.check(envelope.edge_id, envelope.device_id, envelope.counter)
        if not fresh:
            return self._reject(
                409,
                f"replayed counter {envelope.counter} for {envelope.edge_id}/{envelope.device_id}",
            )
        with self.lock:
            self.accepted += 1
            self.readings.append(
                {
                    "edgeId": envelope.edge_id,
                    "deviceId": envelope.device_id,
                    "counter": envelope.counter,
                    "payload": envelope.payload,
                }
            )
        return 202, "accepted"

    def _reject(self, status: int, reason: str) -> tuple[int, str]:
        with self.lock:
            self.rejected += 1
            self.rejections.append(reason)
        return status, reason


def _make_handler(state: CloudReceiverState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/readings":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            status, reason = state.ingest(self.rfile.read(length))
            body = json.dumps({"status": reason}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/stats":
                self.send_error(404)
                return
            with state.lock:
                body = json.dumps({"accepted": state.accepted, "rejected": state.rejected}).encode(
                    "utf-8"
                )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args  # quiet; the receiver's own manifest is the record

    return Handler


class CloudReceiverServer:
    """The demo cloud ingest endpoint: HTTPS when a cert/key pair is given, HTTP otherwise."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        certfile: Path | None = None,
        keyfile: Path | None = None,
        known_edges: frozenset[str] = frozenset(),
    ) -> None:
        self.state = CloudReceiverState(known_edges=known_edges)
        self._server = ThreadingHTTPServer((host, port), _make_handler(self.state))
        self.tls = certfile is not None
        if certfile is not None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(certfile), str(keyfile) if keyfile else None)
            self._server.socket = context.wrap_socket(self._server.socket, server_side=True)
        address = self._server.server_address
        self.host, self.port = str(address[0]), int(address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        scheme = "https" if self.tls else "http"
        host = self.host if self.host != "0.0.0.0" else socket.gethostname()
        return f"{scheme}://{host}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="cloud-receiver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
