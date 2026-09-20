"""Receiver-side replay-window enforcement (Phase 6, Section III-G2, near Eq. 27).

Decision confirmed with the user before implementation (per Golden Rule 1 and
``docs/plans/phase6-replay-window.md``, which explicitly withheld authorization to write this
until the policy was chosen): **per-(edge_id, device_id) scope, strict-monotonic acceptance,
in-memory state.** Persistence across a receiver restart is not implemented -- the same class
of exclusion as production key management, which the paper itself lists as out of scope.

The counter check runs after the receiver has parsed and accepted the routing metadata. See
``routing/cloud_sink.py``'s ``MockCloudReceiver.send_plaintext`` for the enforced ordering: a
message rejected for any reason (malformed metadata, edge_id mismatch) must not advance this
state, or a rejected message could poison the window for a legitimate later one.

    NOTE (2026-09-20 deviation): the telemetry path is no longer Ascon-encrypted (see
    ``routing/cloud_sink.py``), so the counter is now carried as plaintext metadata rather than
    authenticated associated data. Replay detection still works as ordering enforcement, but it no
    longer rests on a cryptographic authenticity guarantee for the counter -- that guarantee moved
    to the federated weight channel (``federated/crypto.py``).

Deliberately its own small object, not folded into ``crypto/ascon_aead.py``: replay state does
not belong in the crypto core, whose job is authenticating a value, not interpreting it.
"""

from __future__ import annotations


class ReplayGuard:
    """Tracks the last-seen counter per (edge_id, device_id) pair; strict-monotonic acceptance."""

    def __init__(self) -> None:
        self._last_seen: dict[tuple[str, str], int] = {}

    def check(self, edge_id: str, device_id: str, counter: int) -> bool:
        """Return ``True`` (and record ``counter``) iff it is strictly greater than last seen.

        A first-seen ``(edge_id, device_id)`` pair is always accepted. Equal or lower counters
        are rejected as a replay or stale delivery -- state is NOT updated on rejection, so a
        replayed message can never advance the window.
        """
        if counter < 0:
            raise ValueError(f"counter must be non-negative, got {counter}")

        key = (edge_id, device_id)
        last = self._last_seen.get(key)
        if last is not None and counter <= last:
            return False
        self._last_seen[key] = counter
        return True

    def last_seen(self, edge_id: str, device_id: str) -> int | None:
        """The last accepted counter for this device, or ``None`` if never seen."""
        return self._last_seen.get((edge_id, device_id))
