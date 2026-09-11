"""Ascon tamper rejection (Section III-G, Eq. 26).

Decryption must return bottom (None), never plaintext, for any modified ciphertext OR modified
associated data. Both single-bit mutations are checked. Activates in Phase 6.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.crypto.ascon_aead import AsconAEAD128


@pytest.mark.skip(reason="pending Phase 6: encrypt/decrypt not implemented yet")
def test_single_bit_ciphertext_mutation_rejected() -> None:
    cipher = AsconAEAD128(key=b"\x00" * 16)
    assert cipher is not None
    raise AssertionError("implement in Phase 6: flip one ciphertext bit -> decrypt returns None")


@pytest.mark.skip(reason="pending Phase 6: encrypt/decrypt not implemented yet")
def test_single_bit_associated_data_mutation_rejected() -> None:
    cipher = AsconAEAD128(key=b"\x00" * 16)
    assert cipher is not None
    raise AssertionError("implement in Phase 6: flip one AD bit -> decrypt returns None")
