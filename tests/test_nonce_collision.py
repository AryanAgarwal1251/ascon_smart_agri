"""GATING TEST (R6) --- nonce discipline (Section III-G3).

Nonce reuse under a fixed key is catastrophic and unrecoverable for AEAD. The NonceRegistry
accumulates every nonce issued per key; issuing many messages under one key must produce zero
collisions, and a deliberate re-registration must raise.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.crypto.ascon_aead import NonceRegistry, NonceReuseError

_KEY_A = b"\x00" * 16
_KEY_B = b"\x11" * 16


@pytest.mark.gating
def test_no_nonce_reuse_under_one_key() -> None:
    registry = NonceRegistry()
    nonces = {registry.issue(_KEY_A) for _ in range(10_000)}
    assert len(nonces) == 10_000  # every CSPRNG-drawn nonce distinct
    assert all(len(n) == 16 for n in nonces)


@pytest.mark.gating
def test_deliberate_reuse_raises() -> None:
    registry = NonceRegistry()
    nonce = registry.issue(_KEY_A)
    with pytest.raises(NonceReuseError):
        registry.register(_KEY_A, nonce)


def test_same_nonce_under_different_keys_is_allowed() -> None:
    registry = NonceRegistry()
    nonce = registry.issue(_KEY_A)
    # Reuse across a *different* key is not a nonce-reuse violation; the registry is per key.
    registry.register(_KEY_B, nonce)
