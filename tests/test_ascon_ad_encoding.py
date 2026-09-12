"""Associated-data byte serialization (Section III-G, Eq. 27; see phase6-ad-serialization.md).

The AD tuple ``⟨edge_id, device_id, counter, schema_version⟩`` is serialized with an injective,
canonical length-prefixed (TLV-style) encoding so a reading can never authenticate against the
wrong device's metadata. These tests cover round-trip, the anti-ambiguity cases that motivate the
encoding, strict parsing, bounds, and integration with the AEAD tamper property.
"""

from __future__ import annotations

import struct

import pytest

from ascon_smart_agri.crypto.ascon_aead import AsconAEAD128, AssociatedData

_KEY = b"\x00" * 16
_NONCE = bytes(range(16))
_PT = b"soil_moisture=0.31"


def test_roundtrip_varied_fields() -> None:
    for ad in (
        AssociatedData("edge-1", "dev-7", 42, "v1"),
        AssociatedData("", "", 0, ""),  # empty strings
        AssociatedData("edge-北", "デバイス", 123, "スキーマ-v2"),  # non-ASCII UTF-8
        AssociatedData("e", "d", 2**64 - 1, "v"),  # counter at u64 max
    ):
        assert AssociatedData.from_bytes(ad.to_bytes()) == ad


def test_anti_ambiguity_field_boundary() -> None:
    # The motivating collision under naive concatenation must NOT occur.
    a = AssociatedData("ab", "c", 1, "v")
    b = AssociatedData("a", "bc", 1, "v")
    assert a.to_bytes() != b.to_bytes()


def test_anti_ambiguity_no_truncation_aliasing() -> None:
    # Ids that a fixed-width encoding would truncate to the same bytes stay distinct.
    a = AssociatedData("sensor-north-01", "dev", 1, "v")
    b = AssociatedData("sensor-north-02", "dev", 1, "v")
    assert a.to_bytes() != b.to_bytes()


def test_neighbouring_ad_fails_decryption() -> None:
    cipher = AsconAEAD128(key=_KEY)
    ad1 = AssociatedData("edge-1", "dev-7", 42, "v1").to_bytes()
    ad2 = AssociatedData("edge-1", "dev-8", 42, "v1").to_bytes()
    ct = cipher.encrypt(_NONCE, ad1, _PT)
    assert cipher.decrypt(_NONCE, ad1, ct) == _PT
    assert cipher.decrypt(_NONCE, ad2, ct) is None


def test_from_bytes_rejects_unknown_fmt_version() -> None:
    good = AssociatedData("e", "d", 1, "v").to_bytes()
    bad = bytes([0x02]) + good[1:]
    with pytest.raises(ValueError, match="fmt_version"):
        AssociatedData.from_bytes(bad)


def test_from_bytes_rejects_truncated_input() -> None:
    good = AssociatedData("edge-1", "dev-7", 1, "v1").to_bytes()
    with pytest.raises(ValueError, match="truncated"):
        AssociatedData.from_bytes(good[:-1])


def test_from_bytes_rejects_trailing_bytes() -> None:
    good = AssociatedData("edge-1", "dev-7", 1, "v1").to_bytes()
    with pytest.raises(ValueError, match="trailing"):
        AssociatedData.from_bytes(good + b"\x00")


def test_to_bytes_rejects_oversized_field() -> None:
    oversized = AssociatedData("x" * (0xFFFF + 1), "d", 1, "v")
    with pytest.raises(ValueError, match="exceeds u16 max"):
        oversized.to_bytes()


@pytest.mark.parametrize("counter", [-1, 2**64])
def test_to_bytes_rejects_out_of_range_counter(counter: int) -> None:
    with pytest.raises(ValueError, match="counter"):
        AssociatedData("e", "d", counter, "v").to_bytes()


def test_wire_layout_is_length_prefixed_big_endian() -> None:
    ad = AssociatedData("ab", "c", 42, "v1")
    expected = (
        struct.pack(">B", 0x01)
        + struct.pack(">H", 2)
        + b"ab"
        + struct.pack(">H", 1)
        + b"c"
        + struct.pack(">Q", 42)
        + struct.pack(">H", 2)
        + b"v1"
    )
    assert ad.to_bytes() == expected
