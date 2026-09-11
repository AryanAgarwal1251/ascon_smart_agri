"""GATING TEST (R6) --- nonce discipline (Section III-G3).

Nonce reuse under a fixed key is catastrophic and unrecoverable for AEAD. The NonceRegistry
accumulates every nonce issued per key; issuing many messages under one key must produce zero
collisions. Activates in Phase 6.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.crypto.ascon_aead import NonceRegistry


@pytest.mark.gating
@pytest.mark.skip(reason="pending Phase 6: NonceRegistry.register not implemented yet")
def test_no_nonce_reuse_under_one_key() -> None:
    # Phase 6 will draw N CSPRNG nonces, register each under one key, and assert no collision
    # is raised; a deliberate re-registration must raise.
    registry = NonceRegistry()
    assert registry is not None
    raise AssertionError("implement in Phase 6")
