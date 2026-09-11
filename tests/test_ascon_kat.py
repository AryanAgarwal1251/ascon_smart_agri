"""GATING TEST (R5) --- Ascon-AEAD128 conformance to NIST SP 800-232 (Section III-G1).

The candidate library's conformance is UNVERIFIED until this passes against the official
known-answer test vectors. It gates all crypto integration: SP 800-232 Ascon-AEAD128 differs
from the pre-standard Ascon-128/128a (revised IVs, little-endian), and the widely installed
package implements the old variant. If this fails, the paper-consistent fallback (III-G1) is
to bind the official reference implementation --- NOT to hand-roll the primitive.

Activates in Phase 6. The KAT vector file must be committed under tests/ and its provenance
recorded in the run manifest.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.crypto.ascon_aead import verify_kat_conformance


@pytest.mark.gating
@pytest.mark.skip(reason="pending Phase 6: backend selection + KAT vectors not landed yet")
def test_backend_matches_sp800_232_vectors() -> None:
    assert verify_kat_conformance() is True
