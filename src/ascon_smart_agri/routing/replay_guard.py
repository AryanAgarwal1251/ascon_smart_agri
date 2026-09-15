"""Receiver-side replay-window enforcement (Phase 6, Section III-G2, near Eq. 27).

Decision confirmed with the user before implementation (per Golden Rule 1 and
``docs/plans/phase6-replay-window.md``, which explicitly withheld authorization to write this
until the policy was chosen): **per-(edge_id, device_id) scope, strict-monotonic acceptance,
in-memory state.** Persistence across a receiver restart is not implemented -- the same class
of exclusion as production key management, which the paper itself lists as out of scope.

MUST be checked only AFTER Ascon verification succeeds (Eq. 26). The counter is only
trustworthy because it is authenticated; checking it before verification would let an attacker
poison this state with forged counters from a message that never should have been accepted.
See ``routing/cloud_sink.py``'s ``MockCloudReceiver.send_encrypted`` for the enforced ordering.

Deliberately its own small object, not folded into ``crypto/ascon_aead.py``: the
``AssociatedData`` docstring there states replay state must not live in the crypto core, since
the core's job is authenticating the counter, not interpreting it.
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
