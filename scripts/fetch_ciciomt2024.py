"""Download the CICIoMT2024 WiFi/MQTT CSVs we actually use, resumably (Phase 9 plan §1).

    PYTHONPATH=. ./.venv/bin/python scripts/fetch_ciciomt2024.py \\
        --train-url "<URL of the .../attacks/CSV/train listing>" \\
        --test-url  "<URL of the .../attacks/CSV/test listing>"  \\
        --cookie "<the Cookie header your logged-in browser sends>"  [--dry-run] [--all]

UNB's "Browse dataset" page is a plain directory listing with one "Download" link per file,
behind a registration session: anonymous requests get ``403 Registration required``. Copy the
session from the browser you registered in --- DevTools > Network > click the listing request
> Request Headers > ``Cookie:`` --- and pass the whole value to ``--cookie``. It is used only
for cicresearch.ca and never written anywhere.
This script reads that page, keeps the files the pipeline needs, and downloads each one
with ``curl``, restarting a file whenever the transfer stalls or drops --- the connection this
was written for is slow and the server does not support resuming.

Which files, and why (see docs/plans/phase9-multi-dataset.md):

* every un-numbered file (Benign, ARP_Spoofing, MQTT-*, Recon-*): the small classes, all rows;
* of each numbered TCP/IP flood (``TCP_IP-DDoS-ICMP1..8``, ...), **part 1 only**: the
  subsampler caps every class at ``per_class_cap`` rows and reads parts in order, so parts 2+
  would never be read. ``--all`` fetches everything instead.

A file counts as present only if it is complete (ends with a newline); ``--skip-existing``
re-fetches a truncated one. Runs unattended: a stalled or dropped transfer restarts the file.

Files land under ``data/ciciomt2024/WiFI_and_MQTT/attacks/CSV/{train,test}/`` with their
original names (the label is the filename). The characterisation report then describes the
downloaded parts, not the full corpus; the README's charts give the full per-class totals.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

_HREF = re.compile(r'href="([^"]*download\.php\?file=[^"]+)"', re.IGNORECASE)
_NUMBERED = re.compile(
    r"^(?P<stem>.+?)(?P<part>\d+)_(?P<split>train|test)\.pcap\.csv$", re.IGNORECASE
)


def select_files(names: list[str], *, everything: bool = False) -> list[str]:
    """The subset to fetch: all un-numbered files, plus part 1 of each numbered family."""
    keep: list[str] = []
    for name in sorted(set(names)):
        match = _NUMBERED.match(name)
        if everything or match is None or match["part"] == "1":
            keep.append(name)
    return keep


def listing_files(listing_url: str, cookie: str = "") -> dict[str, str]:
    """``{filename: absolute download url}`` from a UNB directory-listing page.

    Read with ``curl`` rather than ``urllib``: the python.org macOS build ships without root
    certificates and fails TLS verification on cicresearch.ca, while curl uses the system's.
    """
    html = subprocess.run(
        ["curl", "-sSL", "--max-time", "60", *_cookie_args(cookie), listing_url],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "Registration required" in html:
        raise SystemExit(
            "the listing needs your registered session: pass --cookie (see the module docstring)"
        )
    files: dict[str, str] = {}
    for href in _HREF.findall(html):
        url = urllib.parse.urljoin(listing_url, href)
        # download.php?file=WiFI_and_MQTT%2Fattacks%2FCSV%2Ftrain%2FBenign_train.pcap.csv
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        name = Path(query.get("file", [""])[0]).name
        if name.lower().endswith(".pcap.csv"):
            files[name] = url
    if not files:
        raise SystemExit(f"no *.pcap.csv links found at {listing_url}")
    return files


def _cookie_args(cookie: str) -> list[str]:
    return ["-b", cookie] if cookie else []


def is_complete(path: Path) -> bool:
    """A CSV the server finished sending ends with a newline; a cut-off one ends mid-row."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(-1, 2)
        return fh.read(1) == b"\n"


def fetch(url: str, dest: Path, cookie: str = "") -> bool:
    """Download ``url`` to ``dest`` whole, restarting on any failure; never resumes.

    The server's ``download.php`` does not honour byte ranges: on the first run a resumed
    transfer was reported complete at 1.8 MB of a 37 MB file, cut mid-row. So each attempt
    writes to a temp file, is aborted if it stalls (< 1 KB/s for 30 s, or nothing for 20 s
    while connecting), and counts only if the result ends with a newline.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 31):
        tmp.unlink(missing_ok=True)
        result = subprocess.run(
            ["curl", "-L", "--connect-timeout", "20", "--speed-time", "30",
             "--speed-limit", "1000", *_cookie_args(cookie), "-o", str(tmp), url],
        )  # fmt: skip
        head = tmp.open("rb").read(64) if tmp.exists() else b""
        if head.startswith(b"Registration required") or head.lstrip().startswith(b"<"):
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"{dest.name}: server sent a page, not the CSV -- is --cookie current?"
            )
        if result.returncode == 0 and is_complete(tmp):
            tmp.replace(dest)
            return True
        print(f"     incomplete (curl exit {result.returncode}); restarting ({attempt}/30)")
        time.sleep(min(60, 2 * attempt))  # an outage, not a bad file: back off, then retry
    tmp.unlink(missing_ok=True)
    print(f"     {dest.name}: gave up after 30 attempts; continuing with the next file")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-url", required=True)
    parser.add_argument("--test-url", required=True)
    parser.add_argument("--root", default="data/ciciomt2024/WiFI_and_MQTT/attacks/CSV")
    parser.add_argument("--all", action="store_true", help="every part, not just part 1")
    parser.add_argument("--dry-run", action="store_true", help="list, download nothing")
    parser.add_argument("--cookie", default="", help="the browser's Cookie header for the site")
    parser.add_argument("--skip-existing", action="store_true", help="do not touch files present")
    args = parser.parse_args()

    total = 0
    failed: list[str] = []
    for split, url in (("train", args.train_url), ("test", args.test_url)):
        files = listing_files(url, args.cookie)
        wanted = select_files(list(files), everything=args.all)
        print(f"[{split}] {len(wanted)} of {len(files)} files selected")
        for name in wanted:
            dest = Path(args.root) / split / name
            total += 1
            complete = is_complete(dest)
            if args.dry_run:
                note = "  (present)" if complete else ("  (INCOMPLETE)" if dest.exists() else "")
                print(f"  {name}{note}")
                continue
            if args.skip_existing and complete:
                continue
            print(f"  -> {dest}")
            if not fetch(files[name], dest, args.cookie):
                failed.append(name)
    print(f"done: {total} files{' (dry run)' if args.dry_run else ''}")
    if failed:
        print(f"FAILED ({len(failed)}), re-run with --skip-existing: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
