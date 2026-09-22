"""Per-packet rows -> CICIoT2023-style packet-window features (Phase 9 plan, Edge-IIoTset).

CICIoT2023 and CICIoMT2024 rows are already *windows*: UNB's extractor walks each capture in
packet order and emits one row of statistics per fixed-size group of packets. Edge-IIoTset
rows are single packets with Wireshark field names. To put the three corpora in one feature
language this module re-creates the window step over packet rows: consecutive packets, in
file order, in groups of ``window_packets``, **never across a label change and never across
a source file** --- the same two rules Section III-D imposes on sequence windows, one level
down.

What is reproduced (column -> definition over the packets of one window). The definitions
follow the CICIoMT2024 README's feature table, which documents the same UNB extractor that
produced CICIoT2023 and is the closest thing to a published specification of it:

    Tot sum         total packet length            Min / Max / AVG / Std / Variance  of lengths
    Tot size        (avg.) length of the packet    Number       total packets in the window
    IAT             interval mean between packets  Rate         packets / second in the window
    Header_Length   MEAN transport header length   Time_To_Live mean TTL
    Protocol Type   MODE of the protocols found    TCP / UDP    average no. of such packets
    syn/ack/psh/fin_flag_number   PROPORTION of packets with the flag set
    ack_count / syn_count / fin_count             COUNT of flag occurrences

The window size and the exact time base are still not published, so this remains an
approximation, and the only honest validation is the leave-one-dataset-out evaluation of the
Phase 9 plan --- a corpus aggregated here is never mixed into a training set without that
number being reported next to it.

A packet corpus supplies only the columns it has. :class:`PacketColumns` names which source
column feeds each definition; anything left ``None`` simply produces no canonical column, and
``datasets.harmonise_columns`` reports it as missing so feature selection runs over the
intersection. **Nothing is imputed and nothing is invented.**

Edge-IIoTset specifics live in :func:`edge_iiotset_packets`; they were written from the
published field list and MUST be checked against the characterisation report of the real
file before use (``frame.time`` format, whether ``tcp.len`` is the only length available,
which flag columns exist). That function is the one place to fix when the file disagrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PacketColumns:
    """Which source column supplies each ingredient. ``None`` = this corpus lacks it."""

    length: str  # packet length in bytes; required
    time_s: str | None = None  # packet timestamp in seconds (float); None = no IAT / Rate
    header_length: str | None = None
    ttl: str | None = None
    protocol_number: str | None = None  # IP protocol number: 6 TCP, 17 UDP, 1 ICMP
    is_tcp: str | None = None  # 0/1
    is_udp: str | None = None  # 0/1
    syn: str | None = None  # 0/1 flag columns
    ack: str | None = None
    psh: str | None = None
    fin: str | None = None


def _window_ids(labels: pd.Series[str], sources: pd.Series[str], window_packets: int) -> np.ndarray:
    """Window id per packet: a new window every ``window_packets`` packets, and always at a
    label or source-file change, so no window mixes labels or captures."""
    label_arr = labels.to_numpy().astype(str)
    source_arr = sources.to_numpy().astype(str)
    change = np.ones(len(label_arr), dtype=bool)
    change[1:] = (label_arr[1:] != label_arr[:-1]) | (source_arr[1:] != source_arr[:-1])
    run_id = np.cumsum(change) - 1
    # position within the run
    run_start = np.flatnonzero(change)
    pos = np.arange(len(label_arr)) - run_start[run_id]
    ids: np.ndarray = run_id * (len(label_arr) + 1) + pos // window_packets
    return ids


def aggregate_packet_windows(
    packets: pd.DataFrame,
    cols: PacketColumns,
    *,
    window_packets: int,
    label_col: str = "label",
    source_col: str = "source_file",
    drop_partial: bool = True,
) -> pd.DataFrame:
    """Aggregate packet rows (already in capture order) into canonical window rows.

    The output carries ``label`` and ``source_file`` so it drops straight into the existing
    dedup -> split -> select pipeline. Windows shorter than ``window_packets`` (the tail of a
    run) are dropped by default: a partial window's statistics are not comparable.
    """
    if window_packets <= 0:
        raise ValueError(f"window_packets must be positive, got {window_packets}")
    for required in (cols.length, label_col, source_col):
        if required not in packets.columns:
            raise ValueError(f"packets frame lacks required column {required!r}")

    frame = packets.reset_index(drop=True)
    wid = _window_ids(frame[label_col].astype(str), frame[source_col].astype(str), window_packets)
    groups = frame.groupby(wid, sort=True)

    length = frame[cols.length].astype(np.float64)
    out = pd.DataFrame(
        {
            "Number": groups.size(),
            "Tot sum": length.groupby(wid).sum(),
            "Min": length.groupby(wid).min(),
            "Max": length.groupby(wid).max(),
            "AVG": length.groupby(wid).mean(),
            "Std": length.groupby(wid).std(ddof=0).fillna(0.0),
            "Variance": length.groupby(wid).var(ddof=0).fillna(0.0),
            "Tot size": length.groupby(wid).mean(),
            "label": groups[label_col].first().astype(str),
            "source_file": groups[source_col].first().astype(str),
        }
    )
    if cols.time_s is not None:
        time_s = frame[cols.time_s].astype(np.float64)
        out["IAT"] = time_s.groupby(wid).apply(
            lambda t: float(np.diff(t).mean()) if len(t) > 1 else 0.0
        )
        out["Rate"] = time_s.groupby(wid).apply(_rate)

    def _mean(column: str | None, name: str) -> None:
        if column is not None:
            out[name] = frame[column].astype(np.float64).groupby(wid).mean()

    def _sum(column: str | None, name: str) -> None:
        if column is not None:
            out[name] = frame[column].astype(np.float64).groupby(wid).sum()

    def _mode(column: str | None, name: str) -> None:
        if column is not None:
            out[name] = (
                frame[column].astype(np.float64).groupby(wid).agg(lambda v: float(v.mode().iloc[0]))
            )

    _mean(cols.header_length, "Header_Length")  # README: mean of the transport header lengths
    _mean(cols.ttl, "Time_To_Live")
    _mode(cols.protocol_number, "Protocol Type")  # README: mode of protocols in the window
    _mean(cols.is_tcp, "TCP")
    _mean(cols.is_udp, "UDP")
    _mean(cols.syn, "syn_flag_number")
    _mean(cols.ack, "ack_flag_number")
    _mean(cols.psh, "psh_flag_number")
    _mean(cols.fin, "fin_flag_number")
    _sum(cols.ack, "ack_count")
    _sum(cols.syn, "syn_count")
    _sum(cols.fin, "fin_count")

    if drop_partial:
        out = out.loc[out["Number"] == window_packets]
    return out.reset_index(drop=True)


# ------------------------------------------------------------------------- Edge-IIoTset


#: Edge-IIoTset TCP flag columns, per the published field list. VERIFY against the real file.
_EDGE_TCP_FLAGS = "tcp.flags"  # hex string, e.g. "0x00000018"; PSH is bit 0x08
_EDGE_SYN = "tcp.connection.syn"
_EDGE_FIN = "tcp.connection.fin"
_EDGE_ACK = "tcp.flags.ack"
_EDGE_TCP_LEN = "tcp.len"
# A UDP packet is one with a UDP stream id or port. Reconciled against the real file
# (2026-09-21): ``udp.port`` is the placeholder 0 on every DDoS_UDP row while ``udp.stream`` is
# set on all of them; ``Normal``'s UDP packets carry both, ``MITM``'s carry only the port.
_EDGE_UDP_MARKS = ("udp.stream", "udp.port")
_EDGE_ICMP_MARK = "icmp.checksum"  # set on every DDoS_ICMP row and on nothing else but ICMP
#: Every source column :func:`edge_iiotset_packets` reads -- what a loader needs from the file.
EDGE_IIOTSET_SOURCE_COLUMNS: tuple[str, ...] = (
    _EDGE_TCP_LEN,
    _EDGE_TCP_FLAGS,
    _EDGE_SYN,
    _EDGE_FIN,
    _EDGE_ACK,
    *_EDGE_UDP_MARKS,
    _EDGE_ICMP_MARK,
)


def edge_iiotset_packets(raw: pd.DataFrame) -> tuple[pd.DataFrame, PacketColumns]:
    """Derive the packet ingredients from Edge-IIoTset's Wireshark columns.

    Reconciled against the real ``DNN-EdgeIIoT-dataset.csv`` on 2026-09-21 (see
    ``artifacts/phase1_characterization_report_edge_iiotset.json``): the file has no frame
    length, TTL or header length, so ``length`` is the TCP payload length (``tcp.len``) and
    those three canonical columns are simply absent. **No time base either**: ``frame.time``
    is a timestamp for 13 classes but the literal ``"6.0"`` on every ``DDoS_UDP`` and ``MITM``
    row, so ``IAT`` and ``Rate`` would be undefined for exactly those classes and any fill
    value would be a class marker (user decision 2026-09-21: drop the two features for this
    corpus rather than the two classes). Returns a frame with derived ``_pkt_*`` columns
    appended plus the :class:`PacketColumns` naming them. Raises if an expected source
    column is missing, which is the signal to reconcile this function with the report.
    """
    needed = [_EDGE_TCP_LEN, _EDGE_TCP_FLAGS, _EDGE_SYN, _EDGE_FIN, _EDGE_ACK]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Edge-IIoTset frame lacks expected columns {missing}; reconcile")

    frame = raw.copy()
    frame["_pkt_length"] = pd.to_numeric(frame[_EDGE_TCP_LEN], errors="coerce").fillna(0.0)

    # ``map`` with ``na_action`` so a missing flag (a normalised placeholder) stays missing
    # instead of reaching the parser as ``float('nan')`` -- pandas >= 3 keeps NA through
    # ``astype(str)``; older versions spelled it "nan", which the parser rejected anyway.
    flags = pd.to_numeric(
        frame[_EDGE_TCP_FLAGS].astype(str).str.strip().map(_hex_or_nan, na_action="ignore"),
        errors="coerce",
    ).fillna(0.0)
    frame["_pkt_psh"] = ((flags.astype(np.int64) & 0x08) > 0).astype(np.float64)
    for src, dst in ((_EDGE_SYN, "_pkt_syn"), (_EDGE_FIN, "_pkt_fin"), (_EDGE_ACK, "_pkt_ack")):
        frame[dst] = (pd.to_numeric(frame[src], errors="coerce").fillna(0.0) > 0).astype(np.float64)
    frame["_pkt_is_tcp"] = (flags > 0).astype(np.float64)
    udp_marks = [c for c in _EDGE_UDP_MARKS if c in frame.columns]
    if udp_marks:
        frame["_pkt_is_udp"] = frame[udp_marks].notna().any(axis=1).astype(np.float64)
        is_udp: str | None = "_pkt_is_udp"
    else:
        is_udp = None
    # IP protocol number from which transport layer left fields in the row: 6 TCP, 17 UDP,
    # 1 ICMP; 0 for a packet with none of them (ARP, the bulk of MITM). ``Protocol Type`` is
    # the window's mode of these, as the UNB extractor defines it.
    protocol: str | None = None
    if is_udp is not None and _EDGE_ICMP_MARK in frame.columns:
        is_icmp = frame[_EDGE_ICMP_MARK].notna().to_numpy()
        number = np.zeros(len(frame), dtype=np.float64)
        number[is_icmp] = 1.0
        number[frame["_pkt_is_udp"].to_numpy() > 0] = 17.0
        number[frame["_pkt_is_tcp"].to_numpy() > 0] = 6.0
        frame["_pkt_protocol"] = number
        protocol = "_pkt_protocol"

    cols = PacketColumns(
        length="_pkt_length",
        protocol_number=protocol,
        is_tcp="_pkt_is_tcp",
        is_udp=is_udp,
        syn="_pkt_syn",
        ack="_pkt_ack",
        psh="_pkt_psh",
        fin="_pkt_fin",
    )
    return frame, cols


def _rate(t: pd.Series[float]) -> float:
    """Packets per second over the window's span; 0 for a single packet or zero span."""
    span = float(t.iloc[-1] - t.iloc[0]) if len(t) > 1 else 0.0
    return len(t) / span if span > 0 else 0.0


def _hex_or_nan(text: str) -> float:
    try:
        return float(int(text, 16)) if text.lower().startswith("0x") else float(text)
    except ValueError:
        return float("nan")
