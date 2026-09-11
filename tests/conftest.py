"""Shared pytest fixtures.

The ``gating`` marker (see pyproject.toml) tags the blocking correctness/security gates from
Section III-J2: the leakage test (R3), the nonce-collision test (R6), the path-disjointness
test (G1), and the Ascon KAT conformance test (R5). A failure in any of these blocks the
phases downstream of it.
"""

from __future__ import annotations

import pytest
from configs.base import RunConfig


@pytest.fixture
def default_config() -> RunConfig:
    """A default, fully-validated run configuration."""
    return RunConfig()
