"""Verdict router --- implements the routing rule of Eq. (5) (Phase 6, Section III-A, gap G1).

The classifier verdict selects the data path:

    benign     -> encrypt with Ascon and hand to the CloudTransport (Pi_cloud)
    malicious  -> hand to the AlertSink (Pi_alert), which cannot reach the cloud

The two paths are disjoint by construction (see alert_sink.py). The router is the ONLY object
that holds the cloud transport; it never passes it to the alert sink.

Implementation notes:

* **``route()``'s scaffold signature took ``edge_id: str`` alone, which cannot build a valid
  Eq. (27) associated-data tuple** -- ``device_id``, ``counter``, and ``schema_version`` were
  simply missing, so the benign path could not actually encrypt anything correctly. Widened
  (flagged, Golden Rule 1) to take a full :class:`~ascon_smart_agri.crypto.ascon_aead.
  AssociatedData`, which already carries ``edge_id`` and everything else Eq. (27) needs, rather
  than adding three more scattered parameters.
* **One router instance holds one Ascon key** (``cipher: AsconAEAD128``), consistent with the
  original scaffold's constructor -- this models one edge gateway with one provisioned key, not
  a multi-tenant relay. Multi-edge key selection is the RECEIVER's job (``cloud_sink.py``'s
  ``keys`` mapping), not the sender's.
* **A ``NonceRegistry`` is owned by the router and persists across calls**, never rebuilt per
  message: nonce-reuse-freedom (III-G3) is a property of the whole key's lifetime, not of one
  call. It can be injected for testing (e.g. to pre-seed or inspect issued nonces); by default
  the router constructs its own.
"""

from __future__ import annotations

from ..crypto.ascon_aead import AsconAEAD128, AssociatedData, NonceRegistry
from .alert_sink import AlertSink
from .cloud_sink import CloudTransport


class VerdictRouter:
    """Routes each telemetry item to exactly one disjoint path based on the verdict."""

    def __init__(
        self,
        cloud: CloudTransport,
        alert: AlertSink,
        cipher: AsconAEAD128,
        *,
        nonce_registry: NonceRegistry | None = None,
    ) -> None:
        # The cloud transport lives ONLY here; it is never handed to `alert`.
        self._cloud = cloud
        self._alert = alert
        self._cipher = cipher
        self._nonces = nonce_registry if nonce_registry is not None else NonceRegistry()

    def route(
        self, *, verdict_benign: bool, associated_data: AssociatedData, payload: bytes
    ) -> None:
        """Send benign payloads (encrypted) to the cloud, malicious ones to the alert sink."""
        if verdict_benign:
            nonce = self._nonces.issue(self._cipher.key)
            ad_bytes = associated_data.to_bytes()
            ciphertext = self._cipher.encrypt(nonce, ad_bytes, payload)
            self._cloud.send_encrypted(associated_data.edge_id, nonce, ad_bytes, ciphertext)
        else:
            reason = (
                f"classifier verdict: malicious "
                f"(device={associated_data.device_id}, counter={associated_data.counter})"
            )
            self._alert.raise_alert(associated_data.edge_id, reason)
