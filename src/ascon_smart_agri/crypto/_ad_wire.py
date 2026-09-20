"""Shared length-prefixed (TLV-style) wire encoding for associated-data tuples.

Both :class:`~ascon_smart_agri.crypto.ascon_aead.AssociatedData` (Eq. 27, the telemetry-metadata
tuple used on the routing path) and
:class:`~ascon_smart_agri.federated.crypto.WeightAssociatedData` (the client<->server weight
transport tuple) need the same injective, versioned framing so a tuple can never authenticate
against the wrong fields by string-boundary collision (e.g. ``"ab", "c"`` vs. ``"a", "bc"``
concatenating to the same bytes). Extracted here so both AD types share one tested packing
implementation instead of two independently-maintained copies of the same struct calls.
"""

from __future__ import annotations

import struct

U16_MAX = 0xFFFF
U64_MAX = 2**64 - 1


def pack_str(name: str, value: str) -> bytes:
    """Length-prefixed UTF-8 string: a big-endian u16 length followed by the encoded bytes."""
    encoded = value.encode("utf-8")
    if len(encoded) > U16_MAX:
        raise ValueError(f"{name} is {len(encoded)} UTF-8 bytes, exceeds u16 max {U16_MAX}")
    return struct.pack(">H", len(encoded)) + encoded


def pack_u64(name: str, value: int) -> bytes:
    """Fixed-width big-endian u64, range-checked."""
    if not 0 <= value <= U64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64), got {value}")
    return struct.pack(">Q", value)


class Cursor:
    """A forward-only, bounds-checked reader over AD bytes, used by each type's ``from_bytes``."""

    def __init__(self, data: bytes) -> None:
        self._view = memoryview(data)
        self._offset = 0

    def take(self, n: int, what: str) -> bytes:
        if self._offset + n > len(self._view):
            raise ValueError(f"truncated AD: need {n} bytes for {what} at offset {self._offset}")
        chunk = bytes(self._view[self._offset : self._offset + n])
        self._offset += n
        return chunk

    def take_u8(self, what: str) -> int:
        (value,) = struct.unpack(">B", self.take(1, what))
        return int(value)

    def take_u64(self, what: str) -> int:
        (value,) = struct.unpack(">Q", self.take(8, what))
        return int(value)

    def take_str(self, what: str) -> str:
        length = struct.unpack(">H", self.take(2, f"{what} length"))[0]
        return self.take(length, what).decode("utf-8")

    def assert_exhausted(self) -> None:
        if self._offset != len(self._view):
            raise ValueError(f"trailing bytes after AD: {len(self._view) - self._offset} extra")
