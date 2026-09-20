"""Verdict router --- implements the routing rule of Eq. (5) (Phase 6, Section III-A, gap G1).

The classifier verdict selects the data path:

    benign     -> hand to the CloudTransport (Pi_cloud)
    malicious  -> hand to the AlertSink (Pi_alert), which cannot reach the cloud

The two paths are disjoint by construction (see alert_sink.py). The router is the ONLY object
that holds the cloud transport; it never passes it to the alert sink.

    IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER (flagged per CLAUDE.md golden rule 1, decided
    with the user; see ``docs/design_paper.md``'s "Implementation deviation" section and
    ``crypto/ascon_aead.py``'s module docstring). Eq. (5)'s benign branch originally read
    ``Enc_ke(...)`` before handing off to ``Pi_cloud`` -- this router no longer encrypts that
    payload. Cloud-payload confidentiality/integrity is now assumed to be provided by mechanisms
    outside this codebase; what remains from the original design, and what stays gated by
    ``tests/test_path_disjointness.py``, is the verdict-selects-path structure itself: a
    malicious verdict still cannot reach the cloud transport, benign still cannot reach the
    alert sink. Ascon-AEAD128 now protects Channel 3 instead (``federated/crypto.py``).

Implementation notes:

* **``route()`` still builds a full :class:`~ascon_smart_agri.crypto.ascon_aead.AssociatedData`**
  tuple (``edge_id``, ``device_id``, ``counter``, ``schema_version``) and passes its serialised
  bytes to the cloud sink as plain routing metadata -- this is no longer cryptographic
  associated data (nothing encrypts it), but ``cloud_sink.py`` still uses it for edge-identity
  matching and replay detection, which are independent of encryption.
* **The router no longer holds a cipher or a nonce registry** -- there is nothing left on this
  path for it to encrypt with.
"""

from __future__ import annotations

from ..crypto.ascon_aead import AssociatedData
from .alert_sink import AlertSink
from .cloud_sink import CloudTransport


class VerdictRouter:
    """Routes each telemetry item to exactly one disjoint path based on the verdict."""

    def __init__(self, cloud: CloudTransport, alert: AlertSink) -> None:
        # The cloud transport lives ONLY here; it is never handed to `alert`.
        self._cloud = cloud
        self._alert = alert

    def route(
        self, *, verdict_benign: bool, associated_data: AssociatedData, payload: bytes
    ) -> None:
        """Send benign payloads to the cloud, malicious ones to the alert sink."""
        if verdict_benign:
            metadata = associated_data.to_bytes()
            self._cloud.send_plaintext(associated_data.edge_id, metadata, payload)
        else:
            reason = (
                f"classifier verdict: malicious "
                f"(device={associated_data.device_id}, counter={associated_data.counter})"
            )
            self._alert.raise_alert(associated_data.edge_id, reason)
