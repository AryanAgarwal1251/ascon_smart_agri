"""Networked aggregator and client nodes over the sealed weight channel (Phase 8, plan §2, §4).

This is the piece that turns Phase 4's in-process simulation into K separate processes on
separate machines --- two Raspberry Pis and a laptop in the demonstration --- without changing
what is computed. :class:`AggregatorNode` does exactly what :class:`~.server.FederatedServer`
does, and a loopback run of three :class:`ClientNode` processes reproduces the in-process
result bit-for-bit (``tests/test_node_loopback.py``). The only thing that differs is that
every byte crossing a socket is an Ascon-sealed frame from ``federated/transport.py``.

Wire protocol, one TCP connection per client per round:

    client -> aggregator:   u32 length || sealed "up" frame     (theta_k, n_k) for round r
    aggregator -> client:   u32 length || sealed "down" frame   global theta for round r

The aggregator answers only once every registered client's frame for round *r* has been
opened successfully, so a connection stays open across the barrier. A frame that fails to
open is dropped: the connection is closed with no reply, nothing is applied, and the reason is
logged locally (never sent to the peer). The client sees a closed socket, waits, and re-sends
the same round --- which is safe precisely because the sealed frame is bound to that round.

**Round 0 is the scaler exchange (Section III-F4), over the same channel.** Before any
weights move, each client seals its per-feature sufficient statistics ``(count, mean, M2)`` as
a two-tensor state with ``n_k = count``; the aggregator combines them with Chan's formula
(:func:`~.scaler_stats.combine_stats`) and seals ``(mean, std)`` back. No raw row leaves a
client, and the scaler frames get the same tamper/replay protection as the weights.

**The initial global state is derived, not transmitted.** Every node builds round 1's
starting parameters from the shared run seed exactly as :func:`~..eval.baselines.run_federation`
does (``torch.manual_seed(seed); build_detector(...)``), so there is no unauthenticated
bootstrap message and the loopback equivalence test has something exact to compare against.

Nothing here is a production federated-learning server: there is no client authentication
beyond key possession, no straggler policy beyond a timeout, and no protection against a
keyed client that lies. Those are named out of scope in the project and stubbed by omission.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from .._types import Array
from ..crypto.ascon_aead import NonceRegistry
from ..model.gru import build_detector
from .aggregation import unweighted_average, weighted_fedavg
from .client import FederatedClient
from .scaler_stats import FeatureStats, combine_stats
from .serialization import StateDict
from .transport import AggregatorChannel, ChannelKeys, ClientChannel, peek_client_id

SCALER_ROUND: Final = 0
_LEN_FMT: Final = ">I"
_MAX_MESSAGE: Final = 64 * 1024 * 1024  # 64 MiB: far above |theta| = 135 KB, below silly


class ChannelError(RuntimeError):
    """A node could not complete a round over the sealed channel."""


# ----------------------------------------------------------------------------- framing


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(min(1 << 16, n - len(chunks)))
        if not chunk:
            raise ConnectionError("peer closed the connection mid-message")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(sock: socket.socket, payload: bytes) -> None:
    """Send one length-prefixed message."""
    if len(payload) > _MAX_MESSAGE:
        raise ValueError(f"message of {len(payload)} bytes exceeds the {_MAX_MESSAGE} cap")
    sock.sendall(struct.pack(_LEN_FMT, len(payload)) + payload)


def recv_message(sock: socket.socket) -> bytes:
    """Receive one length-prefixed message."""
    (length,) = struct.unpack(_LEN_FMT, _recv_exact(sock, struct.calcsize(_LEN_FMT)))
    if length > _MAX_MESSAGE:
        raise ValueError(f"announced message of {length} bytes exceeds the {_MAX_MESSAGE} cap")
    return _recv_exact(sock, length)


# ------------------------------------------------------------------------ shared state


def initial_global_state(
    seed: int, n_features: int, hidden_size: int, n_classes: int
) -> dict[str, torch.Tensor]:
    """Round-1 starting parameters, derived from the run seed on every node identically."""
    torch.manual_seed(seed)
    model = build_detector(n_features, hidden_size, n_classes)
    return {name: t.detach().clone() for name, t in model.state_dict().items()}


def stats_to_state(stats: FeatureStats) -> dict[str, torch.Tensor]:
    """Sufficient statistics as a state dict, so the weight channel can carry them."""
    return {
        "mean": torch.as_tensor(np.asarray(stats.mean, dtype=np.float64)),
        "m2": torch.as_tensor(np.asarray(stats.m2, dtype=np.float64)),
    }


def state_to_stats(state: StateDict, count: int) -> FeatureStats:
    """Inverse of :func:`stats_to_state`; ``count`` is the frame's authenticated ``n_k``."""
    return FeatureStats(
        count=count, mean=state["mean"].numpy().astype(np.float64), m2=state["m2"].numpy()
    )


@dataclass
class RoundRecord:
    """What the aggregator logs per round, for the manifest."""

    round_index: int
    sequence_counts: dict[str, int]
    bytes_up: int
    bytes_down: int
    seconds: float
    rejections: int


# ---------------------------------------------------------------------------- aggregator


@dataclass
class _RoundState:
    index: int
    opened: dict[str, tuple[StateDict, int]] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    bytes_up: int = 0


class AggregatorNode:
    """The master-GRU side: opens K sealed updates per round, averages, seals the global back.

    ``rounds`` counts weight rounds; the scaler round 0 always precedes them. The node serves
    ``rounds + 1`` rounds in total and then stops accepting connections.
    """

    def __init__(
        self,
        client_ids: list[str],
        keys: dict[str, ChannelKeys],
        *,
        rounds: int,
        aggregation: str = "weighted",
        host: str = "127.0.0.1",
        port: int = 0,
        round_timeout_s: float = 3600.0,
    ) -> None:
        if not client_ids:
            raise ValueError("an aggregator needs at least one client")
        missing = [cid for cid in client_ids if cid not in keys]
        if missing:
            raise ValueError(f"no channel keys for clients {missing}")
        if aggregation not in {"weighted", "unweighted"}:
            raise ValueError(f"aggregation must be 'weighted' or 'unweighted', got {aggregation!r}")
        if rounds <= 0:
            raise ValueError(f"rounds must be positive, got {rounds}")

        self.client_ids = list(client_ids)  # aggregation order == this order, always
        self.rounds = rounds
        self.aggregation = aggregation
        self.round_timeout_s = round_timeout_s
        registry = NonceRegistry()  # one per process: zero nonce reuse across all downlinks
        self.channels = {
            cid: AggregatorChannel(cid, keys[cid], registry=registry) for cid in client_ids
        }

        self._server = socket.create_server((host, port), reuse_port=False)
        self._server.settimeout(0.5)
        self.host, self.port = self._server.getsockname()[:2]

        self._lock = threading.Condition()
        self._round = _RoundState(SCALER_ROUND)
        self._history: dict[int, dict[str, bytes]] = {}  # round -> per-client sealed reply
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.records: list[RoundRecord] = []
        self.global_state: StateDict | None = None
        self.scaler: tuple[Array, Array] | None = None
        self.rejections: list[tuple[str, str]] = []  # (client id or "?", reason)

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Serve in a background thread until every round is complete or :meth:`stop`."""
        self._thread = threading.Thread(target=self.serve_forever, name="aggregator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server.close()

    @property
    def finished(self) -> bool:
        return self._round.index > self.rounds

    def serve_forever(self) -> None:
        """Accept connections until the last round has been answered."""
        handlers: list[threading.Thread] = []
        while not self._stop.is_set() and not self.finished:
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            handler = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            handler.start()
            handlers.append(handler)
        for handler in handlers:
            handler.join(timeout=1.0)

    # -- one connection ---------------------------------------------------------------

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                frame = recv_message(conn)
            except (OSError, ValueError) as exc:
                self._note_rejection("?", f"unreadable message: {exc}")
                return

            client_id = peek_client_id(frame)
            if client_id is None or client_id not in self.channels:
                self._note_rejection(client_id or "?", "frame not addressed from a known client")
                return

            with self._lock:
                round_index = self._round.index
                if self.finished:
                    self._note_rejection(client_id, "federation already complete")
                    return
                opened = self.channels[client_id].open_update(frame, round_index)
                if opened is None:
                    reason = self.channels[client_id].last_rejection or "bottom"
                    self._note_rejection(client_id, reason)
                    return  # closed with no reply; the client will re-send this round
                self._round.opened[client_id] = opened
                self._round.bytes_up += len(frame)
                if len(self._round.opened) == len(self.client_ids):
                    self._complete_round()
                    self._lock.notify_all()
                else:
                    deadline = time.monotonic() + self.round_timeout_s
                    while round_index not in self._history:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._note_rejection(client_id, f"round {round_index} timed out")
                            return
                        self._lock.wait(timeout=remaining)
                reply = self._history[round_index][client_id]

            try:
                send_message(conn, reply)
            except OSError as exc:
                self._note_rejection(client_id, f"reply failed: {exc}")

    # -- round completion (called with the lock held) --------------------------------------

    def _complete_round(self) -> None:
        current = self._round
        states = [current.opened[cid][0] for cid in self.client_ids]
        counts = [current.opened[cid][1] for cid in self.client_ids]

        if current.index == SCALER_ROUND:
            mean, std = combine_stats(
                [state_to_stats(s, c) for s, c in zip(states, counts, strict=True)]
            )
            self.scaler = (mean, std)
            reply_state: StateDict = {
                "mean": torch.as_tensor(mean, dtype=torch.float64),
                "std": torch.as_tensor(std, dtype=torch.float64),
            }
        else:
            reply_state = (
                weighted_fedavg(states, counts)
                if self.aggregation == "weighted"
                else unweighted_average(states)
            )
            self.global_state = reply_state

        replies = {
            cid: self.channels[cid].seal_global(reply_state, current.index)
            for cid in self.client_ids
        }
        self._history[current.index] = replies
        self.records.append(
            RoundRecord(
                round_index=current.index,
                sequence_counts=dict(zip(self.client_ids, counts, strict=True)),
                bytes_up=current.bytes_up,
                bytes_down=sum(len(r) for r in replies.values()),
                seconds=time.monotonic() - current.started,
                rejections=sum(ch.frames_rejected for ch in self.channels.values()),
            )
        )
        self._round = _RoundState(current.index + 1)

    def _note_rejection(self, client_id: str, reason: str) -> None:
        with self._lock:
            self.rejections.append((client_id, reason))


# -------------------------------------------------------------------------------- client


class ClientNode:
    """The local-GRU side: trains locally, seals up, opens the global that comes back.

    ``make_client`` builds the :class:`FederatedClient` once the scaler is known (round 0's
    reply): the node hands it ``(mean, std)`` and it returns a client holding the scaled,
    windowed local data. Keeping the data pipeline behind that callable keeps this class
    about the channel and nothing else.
    """

    def __init__(
        self,
        client_id: str,
        keys: ChannelKeys,
        *,
        server: tuple[str, int],
        local_stats: FeatureStats,
        make_client: Callable[[Array, Array], FederatedClient],
        initial_state: StateDict,
        rounds: int,
        local_epochs: int,
        retries: int = 20,
        retry_delay_s: float = 0.5,
        connect_timeout_s: float = 3600.0,
    ) -> None:
        self.client_id = client_id
        self.channel = ClientChannel(client_id, keys)
        self.server = server
        self.local_stats = local_stats
        self.make_client = make_client
        self.initial_state = initial_state
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.connect_timeout_s = connect_timeout_s

        self.global_state: StateDict | None = None
        self.scaler: tuple[Array, Array] | None = None
        self.federated_client: FederatedClient | None = None
        self.bytes_up = 0
        self.bytes_down = 0
        self.round_seconds: list[float] = []

    def _exchange(self, frame: bytes) -> bytes:
        """Send one sealed frame and return the sealed reply; retries on a dropped connection."""
        last_error: Exception | None = None
        for _ in range(self.retries):
            try:
                with socket.create_connection(self.server, timeout=self.connect_timeout_s) as sock:
                    send_message(sock, frame)
                    return recv_message(sock)
            except (OSError, ValueError) as exc:
                last_error = exc
                time.sleep(self.retry_delay_s)
        raise ChannelError(
            f"{self.client_id}: no reply after {self.retries} attempts: {last_error}"
        )

    def _round_trip(self, state: StateDict, n_k: int, round_index: int) -> StateDict:
        started = time.monotonic()
        frame = self.channel.seal_update(state, n_k, round_index)
        self.bytes_up += len(frame)
        reply = self._exchange(frame)
        self.bytes_down += len(reply)
        opened = self.channel.open_global(reply, round_index)
        if opened is None:
            reason = self.channel.last_rejection
            raise ChannelError(f"{self.client_id}: round {round_index} reply rejected: {reason}")
        self.round_seconds.append(time.monotonic() - started)
        return opened

    def run(self) -> StateDict:
        """Round 0 (scaler), then ``rounds`` weight rounds; returns the final global state."""
        scaler_reply = self._round_trip(
            stats_to_state(self.local_stats), self.local_stats.count, SCALER_ROUND
        )
        mean = scaler_reply["mean"].numpy().astype(np.float64)
        std = scaler_reply["std"].numpy().astype(np.float64)
        self.scaler = (mean, std)
        self.federated_client = self.make_client(mean, std)

        global_state: StateDict = {n: t.clone() for n, t in self.initial_state.items()}
        for round_index in range(1, self.rounds + 1):
            state, n_k = self.federated_client.local_train(
                global_state, local_epochs=self.local_epochs
            )
            global_state = self._round_trip(state, n_k, round_index)
        self.global_state = global_state
        return global_state
