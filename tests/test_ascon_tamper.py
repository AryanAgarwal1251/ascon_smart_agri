"""Ascon tamper rejection (Section III-G, Eq. 26).

Decryption must return bottom (None), never plaintext, for any modified ciphertext OR modified
associated data. Both single-bit mutations are checked.
"""

from __future__ import annotations

from ascon_smart_agri.crypto.ascon_aead import AsconAEAD128

_KEY = b"\x00" * 16
_NONCE = bytes(range(16))
_AD = b"edge-1|dev-7|42|v1"
_PT = b"soil_moisture=0.31"


def _flip_first_bit(data: bytes) -> bytes:
    return bytes([data[0] ^ 0x01]) + data[1:]


def test_roundtrip_recovers_plaintext() -> None:
    cipher = AsconAEAD128(key=_KEY)
    ct = cipher.encrypt(_NONCE, _AD, _PT)
    assert cipher.decrypt(_NONCE, _AD, ct) == _PT


def test_single_bit_ciphertext_mutation_rejected() -> None:
    cipher = AsconAEAD128(key=_KEY)
    ct = cipher.encrypt(_NONCE, _AD, _PT)
    tampered = _flip_first_bit(ct)
    assert tampered != ct
    assert cipher.decrypt(_NONCE, _AD, tampered) is None


def test_single_bit_associated_data_mutation_rejected() -> None:
    cipher = AsconAEAD128(key=_KEY)
    ct = cipher.encrypt(_NONCE, _AD, _PT)
    tampered_ad = _flip_first_bit(_AD)
    assert tampered_ad != _AD
    assert cipher.decrypt(_NONCE, tampered_ad, ct) is None
