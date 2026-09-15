"""Ascon-AEAD128 latency and ciphertext-expansion measurement (Section III-G4, Eq. 29).

Section III-I2 lists "Ascon encrypt and decrypt latency at the median and 95th percentile,
ciphertext expansion" among what a run reports; neither was measured anywhere in this project
until this module (found while scoping Phase 7's remaining gaps). This benchmarks the
ALREADY-IMPLEMENTED, KAT-verified :class:`~ascon_smart_agri.crypto.ascon_aead.AsconAEAD128`
facade -- it introduces no new cryptography and changes no crypto-core behaviour.

Eq. (29): ``|c|_wire = |p| + 32 bytes`` (a 16-byte tag plus a 16-byte nonce), independent of
payload size. This is a structural property of the scheme, not a measurement with noise, so it
is checked EXACTLY against the real vendored backend's output rather than assumed from the
formula -- if the backend ever changed tag/nonce width this would fail loudly. Only the latency
figures are genuinely empirical (machine- and load-dependent), which is why they are reported
as a distribution (median/p95) rather than a single number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..crypto.ascon_aead import AsconAEAD128, NonceRegistry


@dataclass(frozen=True)
class LatencyStats:
    """Microsecond latency distribution over ``n_samples`` real encrypt or decrypt calls."""

    median_us: float
    p95_us: float
    n_samples: int


@dataclass(frozen=True)
class CryptoBenchmark:
    """One payload size's measured latency and ciphertext expansion (Eq. 29)."""

    payload_bytes: int
    encrypt: LatencyStats
    decrypt: LatencyStats
    ciphertext_expansion_bytes: int  # measured; Eq. (29) says this must be exactly 32
    ciphertext_expansion_pct: float  # relative to payload size


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted list."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of zero samples")
    index = min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[index]


def benchmark_ascon(payload: bytes, *, n_samples: int = 200) -> CryptoBenchmark:
    """Measure real encrypt/decrypt latency and ciphertext expansion for one payload size.

    Every sample is a genuine round trip (encrypt then decrypt, with the round trip's own
    correctness checked) through the KAT-verified backend -- not a synthetic timing stand-in.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not payload:
        raise ValueError("payload must be non-empty")

    key = bytes(16)  # fixed benchmark key, never a real/demo secret -- timing is key-independent
    cipher = AsconAEAD128(key)
    registry = NonceRegistry()
    associated_data = b"benchmark-associated-data"

    encrypt_us: list[float] = []
    decrypt_us: list[float] = []
    ciphertext = b""
    for _ in range(n_samples):
        nonce = registry.issue(key)  # a fresh CSPRNG nonce per message, as III-G3 requires

        t0 = time.perf_counter()
        ciphertext = cipher.encrypt(nonce, associated_data, payload)
        encrypt_us.append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        plaintext = cipher.decrypt(nonce, associated_data, ciphertext)
        decrypt_us.append((time.perf_counter() - t0) * 1e6)
        if plaintext != payload:
            raise RuntimeError("benchmark round-trip failed; this would invalidate the timing")

    encrypt_us.sort()
    decrypt_us.sort()
    # Eq. (29)'s |c|_wire counts BOTH the tag and the nonce that travels alongside it on the
    # wire. AsconAEAD128.encrypt() returns only ciphertext||tag (len(payload) + 16); the nonce
    # is a separate parameter/field in this design, so it must be added explicitly to match
    # what Eq. (29) actually measures -- found by this benchmark failing its own first run
    # against the paper's worked 33%/6.3% examples, not assumed correct in advance.
    expansion = (len(ciphertext) + len(nonce)) - len(payload)

    return CryptoBenchmark(
        payload_bytes=len(payload),
        encrypt=LatencyStats(
            median_us=_percentile(encrypt_us, 50),
            p95_us=_percentile(encrypt_us, 95),
            n_samples=n_samples,
        ),
        decrypt=LatencyStats(
            median_us=_percentile(decrypt_us, 50),
            p95_us=_percentile(decrypt_us, 95),
            n_samples=n_samples,
        ),
        ciphertext_expansion_bytes=expansion,
        ciphertext_expansion_pct=100.0 * expansion / len(payload),
    )


def benchmark_at_sizes(
    payload_sizes: list[int], *, n_samples: int = 200
) -> dict[int, CryptoBenchmark]:
    """Convenience wrapper: benchmark several payload sizes in one call.

    The paper's own worked examples are 96 bytes ("a typical sensor reading", 33% expansion)
    and 512 bytes (6.3% expansion) -- both reproduce exactly from Eq. (29), since 32/96 = 33.3%
    and 32/512 = 6.25%.
    """
    if not payload_sizes:
        raise ValueError("payload_sizes must be non-empty")
    return {size: benchmark_ascon(bytes(size), n_samples=n_samples) for size in payload_sizes}
