"""Ascon-sealed weight exchange between a local GRU and the aggregator (Phase 8, plan Section 3).

Every federated update that crosses the network is one *frame*:

    frame = MAGIC(4) || nonce(16) || L(ad):u16 || ad || ciphertext||tag

where the plaintext is the safetensors blob of ``federated/serialization.py`` (so ``n_k``, the
Eq. (21) weight, is under the tag too) and ``ad`` is a serialised
:class:`~ascon_smart_agri.crypto.ascon_aead.WeightAssociatedData`
``<client_id, round, direction, schema_version>``. The cipher is the KAT-gated
:class:`~ascon_smart_agri.crypto.ascon_aead.AsconAEAD128`; nothing here touches a primitive.

Why the channel is shaped this way:

* **One key per client per direction.** ``k_up`` is used to *encrypt* only by that client and
  ``k_down`` only by the aggregator, so every key has exactly one encrypting process and the
  per-process :class:`NonceRegistry` makes "zero nonce reuse per key" a structural fact rather
  than a cross-machine convention. Compromise of one Pi exposes one client's channel only.
* **The opener decides, and it decides once.** :meth:`WeightOpener.open` returns the state and
  ``n_k`` or ``None`` (bottom) --- for a bad tag, a malformed frame, a foreign client id, the
  wrong direction, the wrong schema, or a round that is not the one expected. A caller that
  gets ``None`` applies nothing; there is no partial acceptance and no exception path that a
  network loop could turn into a retry-with-tampered-data. ``last_rejection`` records why, for
  the node's log, never for the peer.
* **Replay is rejected by round, not by a window.** A frame carries the round it belongs to
  under the tag. The opener accepts a frame only for the round it is currently expecting and
  only if that round is strictly later than the last one it accepted, so a captured round-*r*
  frame is dead the moment round *r* completes, and a valid uplink frame presented as a
  downlink fails on ``direction`` before the tag is even checked.
* **Demo keys are labelled and live outside git.** :func:`load_keys` refuses a key file that
  does not declare itself a demo file, so a real deployment cannot silently run on the demo
  material (III-J3); ``*.demo.key`` and ``keys/`` are gitignored.

Out of scope, stubbed here by omission and not by accident: key agreement or rotation, client
authentication beyond possession of the key, and any defence against a *legitimately keyed*
client sending poisoned weights (Byzantine-robust aggregation is out of scope for the project).
"""

from __future__ import annotations

import json
import secrets
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..crypto.ascon_aead import AsconAEAD128, NonceRegistry, WeightAssociatedData
from .serialization import StateDict, deserialize_state, serialize_state

#: Wire schema of the sealed weight frame; bumped if the plaintext or AD layout changes.
SCHEMA_VERSION: Final = "w1"

_MAGIC: Final = b"ASAW"  # "ascon smart agri weights"; rejects frames from the wrong protocol
_NONCE_BYTES: Final = 16
_KEY_BYTES: Final = 16
_AD_LEN_FMT: Final = ">H"
_MIN_FRAME: Final = len(_MAGIC) + _NONCE_BYTES + struct.calcsize(_AD_LEN_FMT)

#: The label a key file must carry; anything else is refused (see :func:`load_keys`).
DEMO_KEY_LABEL: Final = "DEMO KEYS - generated for the Phase 8 demonstration; NOT for production"


@dataclass(frozen=True)
class ChannelKeys:
    """The two 128-bit keys of one client's channel: ``up`` (client -> aggregator), ``down``."""

    up: bytes
    down: bytes

    def __post_init__(self) -> None:
        for name, key in (("up", self.up), ("down", self.down)):
            if len(key) != _KEY_BYTES:
                raise ValueError(f"{name} key must be {_KEY_BYTES} bytes, got {len(key)}")
        if self.up == self.down:
            raise ValueError("up and down keys must differ (one encrypting process per key)")


def generate_demo_keys(client_ids: list[str]) -> dict[str, ChannelKeys]:
    """Draw fresh CSPRNG up/down keys for each client id."""
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("client ids must be unique")
    return {
        cid: ChannelKeys(up=secrets.token_bytes(_KEY_BYTES), down=secrets.token_bytes(_KEY_BYTES))
        for cid in client_ids
    }


def write_keys(path: Path, keys: dict[str, ChannelKeys]) -> None:
    """Write a labelled demo key file (JSON, hex-encoded). Keep it out of version control."""
    payload = {
        "label": DEMO_KEY_LABEL,
        "schema_version": SCHEMA_VERSION,
        "clients": {cid: {"up": k.up.hex(), "down": k.down.hex()} for cid, k in keys.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_keys(path: Path) -> dict[str, ChannelKeys]:
    """Load a key file written by :func:`write_keys`, refusing one without the demo label."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("label") != DEMO_KEY_LABEL:
        raise ValueError(
            f"{path} is not a labelled demo key file; production key management is out of scope"
        )
    clients = payload.get("clients")
    if not isinstance(clients, dict) or not clients:
        raise ValueError(f"{path} carries no client keys")
    return {
        str(cid): ChannelKeys(up=bytes.fromhex(entry["up"]), down=bytes.fromhex(entry["down"]))
        for cid, entry in clients.items()
    }


def _pack_frame(nonce: bytes, ad: bytes, ciphertext: bytes) -> bytes:
    if len(ad) > 0xFFFF:
        raise ValueError("associated data exceeds the u16 length prefix")
    return b"".join((_MAGIC, nonce, struct.pack(_AD_LEN_FMT, len(ad)), ad, ciphertext))


def _unpack_frame(frame: bytes) -> tuple[bytes, bytes, bytes] | None:
    """``(nonce, ad, ciphertext)`` or ``None`` if the frame is structurally malformed."""
    if len(frame) < _MIN_FRAME or not frame.startswith(_MAGIC):
        return None
    offset = len(_MAGIC)
    nonce = frame[offset : offset + _NONCE_BYTES]
    offset += _NONCE_BYTES
    (ad_len,) = struct.unpack(_AD_LEN_FMT, frame[offset : offset + 2])
    offset += 2
    if offset + ad_len > len(frame):
        return None
    ad = frame[offset : offset + ad_len]
    return nonce, ad, frame[offset + ad_len :]


class WeightSealer:
    """Seals ``(state, n_k)`` for one ``(client_id, direction)`` under one key."""

    def __init__(
        self,
        key: bytes,
        client_id: str,
        direction: str,
        *,
        registry: NonceRegistry | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self._cipher = AsconAEAD128(key)
        self._key = key
        self.client_id = client_id
        self.direction = direction
        self.schema_version = schema_version
        # Shared by every sealer in a process so the zero-reuse invariant holds across them.
        self.registry = registry if registry is not None else NonceRegistry()
        self.frames_sealed = 0

    def seal(self, state: StateDict, sequence_count: int, round_index: int) -> bytes:
        """Return the sealed frame for ``round_index``; a fresh nonce is drawn per call."""
        ad = WeightAssociatedData(
            client_id=self.client_id,
            round=round_index,
            direction=self.direction,
            schema_version=self.schema_version,
        ).to_bytes()
        plaintext = serialize_state(state, sequence_count)
        nonce = self.registry.issue(self._key)
        ciphertext = self._cipher.encrypt(nonce, ad, plaintext)
        self.frames_sealed += 1
        return _pack_frame(nonce, ad, ciphertext)


class WeightOpener:
    """Opens frames for one ``(client_id, direction)``; returns bottom on any doubt."""

    def __init__(
        self, key: bytes, client_id: str, direction: str, *, schema_version: str = SCHEMA_VERSION
    ) -> None:
        self._cipher = AsconAEAD128(key)
        self.client_id = client_id
        self.direction = direction
        self.schema_version = schema_version
        self.last_accepted_round: int | None = None
        self.last_rejection: str | None = None
        self.frames_rejected = 0

    def _reject(self, reason: str) -> None:
        """Record why a frame was dropped (for the node's log; never sent to the peer)."""
        self.last_rejection = reason
        self.frames_rejected += 1

    def open(self, frame: bytes, *, expected_round: int) -> tuple[StateDict, int] | None:
        """``(state, n_k)`` if the frame verifies for ``expected_round``, else ``None``.

        The checks run cheapest-first, but every failure is the same outcome: nothing is
        applied. The order is never observable to the peer.
        """
        parts = _unpack_frame(frame)
        if parts is None:
            self._reject("malformed frame")
            return None
        nonce, ad_bytes, ciphertext = parts

        try:
            ad = WeightAssociatedData.from_bytes(ad_bytes)
        except ValueError as exc:
            self._reject(f"malformed associated data: {exc}")
            return None

        if ad.client_id != self.client_id:
            self._reject(f"client id {ad.client_id!r} is not {self.client_id!r}")
            return None
        if ad.direction != self.direction:
            self._reject(f"direction {ad.direction!r} is not {self.direction!r}")
            return None
        if ad.schema_version != self.schema_version:
            self._reject(f"schema {ad.schema_version!r} is not {self.schema_version!r}")
            return None
        if ad.round != expected_round:
            self._reject(f"round {ad.round} is not the expected {expected_round}")
            return None
        if self.last_accepted_round is not None and ad.round <= self.last_accepted_round:
            self._reject(f"round {ad.round} already accepted (replay)")
            return None

        plaintext = self._cipher.decrypt(nonce, ad_bytes, ciphertext)
        if plaintext is None:
            self._reject("authentication failed (bottom)")
            return None

        try:
            state, sequence_count = deserialize_state(plaintext)
        except ValueError as exc:
            self._reject(f"undecodable plaintext: {exc}")
            return None

        self.last_accepted_round = ad.round
        self.last_rejection = None
        return state, sequence_count


class ClientChannel:
    """The client end: seals its updates under ``k_up``, opens global states under ``k_down``."""

    def __init__(
        self, client_id: str, keys: ChannelKeys, *, registry: NonceRegistry | None = None
    ) -> None:
        self.client_id = client_id
        self._sealer = WeightSealer(keys.up, client_id, "up", registry=registry)
        self._opener = WeightOpener(keys.down, client_id, "down")

    def seal_update(self, state: StateDict, sequence_count: int, round_index: int) -> bytes:
        """Seal this round's ``(theta_k, n_k)`` for the aggregator."""
        return self._sealer.seal(state, sequence_count, round_index)

    def open_global(self, frame: bytes, round_index: int) -> StateDict | None:
        """Open the aggregator's global state for ``round_index``, or ``None``."""
        opened = self._opener.open(frame, expected_round=round_index)
        return None if opened is None else opened[0]

    @property
    def last_rejection(self) -> str | None:
        return self._opener.last_rejection


class AggregatorChannel:
    """The aggregator end for ONE client: opens its updates, seals global states back to it."""

    def __init__(
        self, client_id: str, keys: ChannelKeys, *, registry: NonceRegistry | None = None
    ) -> None:
        self.client_id = client_id
        self._opener = WeightOpener(keys.up, client_id, "up")
        self._sealer = WeightSealer(keys.down, client_id, "down", registry=registry)

    def open_update(self, frame: bytes, round_index: int) -> tuple[StateDict, int] | None:
        """Open the client's ``(theta_k, n_k)`` for ``round_index``, or ``None``."""
        return self._opener.open(frame, expected_round=round_index)

    def seal_global(self, state: StateDict, round_index: int) -> bytes:
        """Seal the new global state for this client. ``n_k`` is 0: the global has no weight."""
        return self._sealer.seal(state, 0, round_index)

    @property
    def last_rejection(self) -> str | None:
        return self._opener.last_rejection

    @property
    def frames_rejected(self) -> int:
        return self._opener.frames_rejected


def peek_client_id(frame: bytes) -> str | None:
    """The ``client_id`` from a frame's (authenticated, unencrypted) AD, or ``None``.

    The aggregator needs it to pick the channel --- the same reason Eq. (27) keeps ``edge_id``
    in the clear. It is a routing hint only: the opener re-checks it under the tag.
    """
    parts = _unpack_frame(frame)
    if parts is None:
        return None
    try:
        return WeightAssociatedData.from_bytes(parts[1]).client_id
    except ValueError:
        return None


def frame_overhead_bytes(frame: bytes, plaintext_len: int) -> int:
    """Bytes the sealed frame adds over its plaintext (nonce + tag + AD + framing).

    Reported next to Eq. (29): for a 135 KB weight blob the expansion is ~0.02 %, against 33 %
    for a 96-byte telemetry reading --- the argument for putting Ascon on the weight channel.
    """
    return len(frame) - plaintext_len
