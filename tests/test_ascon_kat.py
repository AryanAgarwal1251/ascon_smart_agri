"""GATING TEST (R5) --- Ascon-AEAD128 conformance to NIST SP 800-232 (Section III-G1).

The candidate library's conformance is UNVERIFIED until this passes against the official
known-answer test vectors. It gates all crypto integration: SP 800-232 Ascon-AEAD128 differs
from the pre-standard Ascon-128/128a (revised IVs, little-endian), and the PyPI ``ascon``
package implements only the old variants. The paper-consistent fallback (III-G1) is to bind the
official reference implementation --- NOT to hand-roll the primitive --- which is what the
vendored ``crypto/_vendor/pyascon`` backend does.

The KAT vector file is committed under ``tests/kat/`` and its provenance recorded there and in
the run manifest (``backend_provenance``).
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.crypto.ascon_aead import verify_kat_conformance


@pytest.mark.gating
def test_backend_matches_sp800_232_vectors() -> None:
    assert verify_kat_conformance() is True
