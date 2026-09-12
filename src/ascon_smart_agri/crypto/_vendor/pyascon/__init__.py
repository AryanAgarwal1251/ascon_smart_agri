"""pyascon --- the official Ascon reference implementation (NIST SP 800-232).

This is the paper-consistent backend for Phase 6 (design paper III-G1, CLAUDE.md "Open
decision"): the PyPI ``ascon`` candidate implements only the pre-standard Ascon v1.2 variants
(``Ascon-128``/``Ascon-128a``) and does not expose ``Ascon-AEAD128`` at all, so it cannot pass
the SP 800-232 KAT. Rather than hand-roll the primitive (forbidden by III-G1), the official
reference implementation is vendored and gated behind ``tests/test_ascon_kat.py``.

Provenance (see ``PROVENANCE.md`` for full detail):
    upstream:   https://github.com/meichlseder/pyascon
    author:     Maria Eichlseder (one of the four original Ascon designers)
    commit:     ed24e54abf9507d26fa49b46a56091570c7e743e
    file:       ascon.py (byte-identical; sha256 recorded in PROVENANCE.md)
    license:    CC0 1.0 Universal (public domain dedication) --- see LICENSE

``ascon.py`` is kept verbatim so it can be re-verified by diffing against the pinned upstream
commit. Do not edit it. The KAT-gated facade lives in ``crypto/ascon_aead.py``.
"""

from .ascon import ascon_decrypt, ascon_encrypt

__all__ = ["ascon_decrypt", "ascon_encrypt"]
