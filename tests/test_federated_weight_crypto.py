"""Ascon-AEAD128 protection of the federated weight transport (Channel 3, federated/crypto.py).

Implementation deviation from the design paper (CLAUDE.md golden rule 1): this is where Ascon now
lives instead of on the telemetry path. Mirrors the style of ``test_ascon_ad_encoding.py`` (AD
round-trip/anti-ambiguity) and ``test_ascon_tamper.py`` ("reject, never accept") for the new
``WeightAssociatedData`` tuple and ``protect_state``/``unprotect_state`` pair.
"""

from __future__ import annotations

import pytest
import torch

from ascon_smart_agri.crypto.ascon_aead import AsconAEAD128, NonceRegistry
from ascon_smart_agri.federated.crypto import (
    WeightAssociatedData,
    WeightIntegrityError,
    protect_state,
    unprotect_state,
)

# ---------------------------------------------------------------- WeightAssociatedData wire format


def test_ad_round_trips() -> None:
    ad = WeightAssociatedData(client_id=2, round_index=17, direction="upload", schema_version="1")

    assert WeightAssociatedData.from_bytes(ad.to_bytes()) == ad


def test_ad_round_trips_for_broadcast_direction_and_edge_values() -> None:
    ad = WeightAssociatedData(client_id=0, round_index=0, direction="broadcast", schema_version="")

    assert WeightAssociatedData.from_bytes(ad.to_bytes()) == ad


def test_ad_direction_changes_the_wire_bytes() -> None:
    """A ciphertext for one leg of a round must not verify as the other leg's AD."""
    upload = WeightAssociatedData(1, 5, "upload", "1")
    broadcast = WeightAssociatedData(1, 5, "broadcast", "1")

    assert upload.to_bytes() != broadcast.to_bytes()


def test_ad_client_id_and_round_index_are_not_confusable() -> None:
    """<client_id=1, round=257> must not collide with <client_id=257, round=1>."""
    a = WeightAssociatedData(1, 257, "upload", "1")
    b = WeightAssociatedData(257, 1, "upload", "1")

    assert a.to_bytes() != b.to_bytes()


def test_ad_rejects_an_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        WeightAssociatedData(0, 0, "sideways", "1").to_bytes()  # type: ignore[arg-type]


def test_ad_from_bytes_rejects_unknown_fmt_version() -> None:
    ad = WeightAssociatedData(0, 0, "upload", "1")
    tampered = bytes([0xFF]) + ad.to_bytes()[1:]

    with pytest.raises(ValueError, match="fmt_version"):
        WeightAssociatedData.from_bytes(tampered)


# ---------------------------------------------------------------- protect_state / unprotect_state


def _cipher_and_registry(key: bytes = bytes(16)) -> tuple[AsconAEAD128, NonceRegistry]:
    return AsconAEAD128(key), NonceRegistry()


def test_protect_unprotect_round_trips_exactly() -> None:
    cipher, registry = _cipher_and_registry()
    state = {"w": torch.randn(4, 3), "b": torch.zeros(3)}
    ad = WeightAssociatedData(0, 1, "upload", "1")

    nonce, ad_bytes, ciphertext = protect_state(cipher, registry, state, 123, ad=ad)
    restored, n_k = unprotect_state(cipher, nonce, ad_bytes, ciphertext)

    assert n_k == 123
    assert set(restored) == {"w", "b"}
    torch.testing.assert_close(restored["w"], state["w"])
    torch.testing.assert_close(restored["b"], state["b"])


def test_protect_draws_a_fresh_nonce_each_call() -> None:
    cipher, registry = _cipher_and_registry()
    state = {"w": torch.ones(2)}

    ad_a = WeightAssociatedData(0, 0, "upload", "1")
    ad_b = WeightAssociatedData(0, 1, "upload", "1")
    nonce_a, _, _ = protect_state(cipher, registry, state, 1, ad=ad_a)
    nonce_b, _, _ = protect_state(cipher, registry, state, 1, ad=ad_b)

    assert nonce_a != nonce_b


def test_tampered_ciphertext_is_rejected_not_silently_accepted() -> None:
    cipher, registry = _cipher_and_registry()
    state = {"w": torch.ones(2)}
    ad = WeightAssociatedData(0, 1, "upload", "1")
    nonce, ad_bytes, ciphertext = protect_state(cipher, registry, state, 1, ad=ad)

    tampered = bytes([ciphertext[0] ^ 0x01]) + ciphertext[1:]

    with pytest.raises(WeightIntegrityError):
        unprotect_state(cipher, nonce, ad_bytes, tampered)


def test_swapped_client_id_in_ad_is_rejected() -> None:
    """A ciphertext genuinely produced for client 0 must not verify as client 1's update."""
    cipher, registry = _cipher_and_registry()
    state = {"w": torch.ones(2)}
    ad_for_client_0 = WeightAssociatedData(0, 1, "upload", "1")
    nonce, _, ciphertext = protect_state(cipher, registry, state, 1, ad=ad_for_client_0)

    wrong_ad_bytes = WeightAssociatedData(1, 1, "upload", "1").to_bytes()

    with pytest.raises(WeightIntegrityError):
        unprotect_state(cipher, nonce, wrong_ad_bytes, ciphertext)


def test_swapped_direction_in_ad_is_rejected() -> None:
    """A ciphertext for the broadcast leg must not verify as the upload leg's AD, or vice versa."""
    cipher, registry = _cipher_and_registry()
    state = {"w": torch.ones(2)}
    broadcast_ad = WeightAssociatedData(0, 1, "broadcast", "1")
    nonce, _, ciphertext = protect_state(cipher, registry, state, 0, ad=broadcast_ad)

    upload_ad_bytes = WeightAssociatedData(0, 1, "upload", "1").to_bytes()

    with pytest.raises(WeightIntegrityError):
        unprotect_state(cipher, nonce, upload_ad_bytes, ciphertext)


def test_nonce_reuse_across_many_simulated_rounds_never_collides() -> None:
    cipher, registry = _cipher_and_registry()
    state = {"w": torch.ones(1)}

    for round_index in range(200):
        ad = WeightAssociatedData(0, round_index, "upload", "1")
        protect_state(cipher, registry, state, 1, ad=ad)  # raises NonceReuseError on collision
