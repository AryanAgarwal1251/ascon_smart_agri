#!/usr/bin/env python
"""Turn every .puml file in paper/diagrams/ into a plantuml.com render link.

PlantUML's public server takes the diagram source compressed into the URL itself, so a link
renders without installing anything. The encoding is raw DEFLATE followed by PlantUML's own
base64 variant, whose alphabet differs from standard base64 -- which is why a normal
``base64.b64encode`` produces a link that silently fails.

Usage::

    ./.venv/bin/python scripts/plantuml_links.py          # print links
    ./.venv/bin/python scripts/plantuml_links.py --fetch  # also download the PNGs
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

DIAGRAMS = Path("paper/diagrams")
SERVER = "https://www.plantuml.com/plantuml"
_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def _encode6(value: int) -> str:
    return _ALPHABET[value & 0x3F]


def _append3(b1: int, b2: int, b3: int) -> str:
    return (
        _encode6(b1 >> 2)
        + _encode6(((b1 & 0x3) << 4) | (b2 >> 4))
        + _encode6(((b2 & 0xF) << 2) | (b3 >> 6))
        + _encode6(b3 & 0x3F)
    )


def plantuml_encode(text: str) -> str:
    """Raw-DEFLATE the source, then apply PlantUML's base64 variant."""
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    data = compressor.compress(text.encode("utf-8")) + compressor.flush()
    out = []
    for i in range(0, len(data), 3):
        chunk = data[i : i + 3]
        b1 = chunk[0]
        b2 = chunk[1] if len(chunk) > 1 else 0
        b3 = chunk[2] if len(chunk) > 2 else 0
        out.append(_append3(b1, b2, b3))
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download rendered PNGs alongside")
    args = parser.parse_args()

    sources = sorted(DIAGRAMS.glob("*.puml"))
    if not sources:
        print(f"no .puml files under {DIAGRAMS}")
        return 1

    for src in sources:
        encoded = plantuml_encode(src.read_text(encoding="utf-8"))
        print(f"\n{src.name}")
        for fmt in ("png", "svg"):
            print(f"  {fmt:<4} {SERVER}/{fmt}/{encoded}")
        if args.fetch:
            import urllib.request

            target = src.with_suffix(".png")
            urllib.request.urlretrieve(f"{SERVER}/png/{encoded}", target)
            print(f"  saved {target} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
