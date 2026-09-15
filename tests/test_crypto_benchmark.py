"""Unit tests for Ascon latency/ciphertext-expansion measurement (Section III-G4, Eq. 29)."""

from __future__ import annotations

import pytest

from ascon_smart_agri.eval.crypto_benchmark import benchmark_ascon, benchmark_at_sizes


def test_ciphertext_expansion_is_exactly_32_bytes() -> None:
    # Eq. (29): |c|_wire = |p| + 32 -- a structural property, checked exactly.
    result = benchmark_ascon(b"x" * 100, n_samples=5)

    assert result.ciphertext_expansion_bytes == 32


def test_expansion_percentage_matches_the_papers_worked_examples() -> None:
    # Section III-G4: "a typical 96-byte sensor reading" -> 33% expansion; 512 bytes -> 6.3%.
    at_96 = benchmark_ascon(bytes(96), n_samples=5)
    at_512 = benchmark_ascon(bytes(512), n_samples=5)

    assert at_96.ciphertext_expansion_pct == pytest.approx(33.33, abs=0.1)
    assert at_512.ciphertext_expansion_pct == pytest.approx(6.25, abs=0.1)


def test_expansion_is_independent_of_payload_size() -> None:
    small = benchmark_ascon(bytes(10), n_samples=3)
    large = benchmark_ascon(bytes(10_000), n_samples=3)

    assert small.ciphertext_expansion_bytes == large.ciphertext_expansion_bytes == 32


def test_latency_stats_report_a_sane_ordering() -> None:
    result = benchmark_ascon(bytes(96), n_samples=50)

    assert result.encrypt.median_us <= result.encrypt.p95_us
    assert result.decrypt.median_us <= result.decrypt.p95_us
    assert result.encrypt.n_samples == 50
    assert result.encrypt.median_us > 0  # real measured time, not a zero stand-in


def test_every_round_trip_actually_verifies() -> None:
    """If decrypt ever failed to recover the original payload, benchmark_ascon must raise
    rather than silently report a timing for a broken round trip."""
    result = benchmark_ascon(b"agricultural sensor reading", n_samples=30)

    assert result.payload_bytes == len(b"agricultural sensor reading")


def test_benchmark_at_sizes_covers_every_requested_size() -> None:
    results = benchmark_at_sizes([64, 96, 512], n_samples=3)

    assert set(results) == {64, 96, 512}
    assert all(r.ciphertext_expansion_bytes == 32 for r in results.values())


def test_rejects_degenerate_arguments() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        benchmark_ascon(b"x", n_samples=0)
    with pytest.raises(ValueError, match="payload"):
        benchmark_ascon(b"", n_samples=1)
    with pytest.raises(ValueError, match="payload_sizes"):
        benchmark_at_sizes([], n_samples=1)
