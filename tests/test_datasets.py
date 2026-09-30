"""Multi-dataset registry, harmonisation and packet-window aggregation (Phase 9 plan).

Everything here runs on synthetic frames shaped like the published documentation of each
corpus; the real files must still be characterised and reconciled (see data/datasets.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ascon_smart_agri.data.datasets import (
    CANONICAL_COLUMNS,
    CICIOMT2024,
    EDGE_IIOTSET,
    REGISTRY,
    SELECTED_COLUMNS,
    TRAINING_CORPORA,
    canonical_name,
    discover_parts,
    get_dataset,
    harmonise_columns,
    label_from_ciciomt2024_filename,
    normalise_placeholders,
    placeholder_collisions,
    selected_columns_available,
)
from ascon_smart_agri.data.packet_windows import (
    PacketColumns,
    aggregate_packet_windows,
    edge_iiotset_packets,
)
from ascon_smart_agri.data.taxonomy import (
    CICIOMT2024_LEAF_TO_FAMILY,
    CLASS_NAMES,
    EDGE_IIOTSET_LEAF_TO_FAMILY,
    to_class_index,
    to_family,
)

# ------------------------------------------------------------------------------ registry


def test_registry_lists_the_three_corpora() -> None:
    assert set(REGISTRY) == {"ciciot2023", "ciciomt2024", "edge_iiotset"}
    with pytest.raises(ValueError, match="unknown dataset"):
        get_dataset("bot-iot")


def test_training_corpora_are_ciciot2023_alone_each_exclusion_carrying_its_reason() -> None:
    # Edge-IIoTset excluded 2026-09-22, CICIoMT2024 2026-10-01. Both stay registered so they
    # remain characterisable and usable as ablations via an explicit --corpora flag; what the
    # exclusion removes is their standing as a silent default. Every exclusion must carry a
    # reason string, so the registry cannot disagree with docs/dataset-selection.md without
    # this test failing.
    assert TRAINING_CORPORA == ("ciciot2023",)
    assert EDGE_IIOTSET.not_for_training
    assert get_dataset("ciciomt2024").not_for_training
    assert all(not get_dataset(n).not_for_training for n in TRAINING_CORPORA)


def test_selected_columns_are_canonical() -> None:
    assert set(SELECTED_COLUMNS) <= set(CANONICAL_COLUMNS)
    assert len(SELECTED_COLUMNS) == 16


@pytest.mark.parametrize(
    ("filename", "leaf"),
    [
        ("Benign_train.pcap.csv", "Benign"),
        ("Benign_test.pcap.csv", "Benign"),
        ("ARP_Spoofing_train.pcap.csv", "ARP_Spoofing"),
        ("MQTT-DDoS-Connect_Flood_test.pcap.csv", "MQTT-DDoS-Connect_Flood"),
        ("TCP_IP-DDoS-ICMP2_train.pcap.csv", "TCP_IP-DDoS-ICMP"),
        ("TCP_IP-DoS-UDP4_test.pcap.csv", "TCP_IP-DoS-UDP"),
        ("DDoS-ICMP3_train.pcap.csv", "DDoS-ICMP"),
        ("Recon-VulScan_train.pcap.csv", "Recon-VulScan"),
    ],
)
def test_ciciomt2024_label_from_filename(filename: str, leaf: str) -> None:
    assert label_from_ciciomt2024_filename(Path(filename)) == leaf
    assert leaf in CICIOMT2024_LEAF_TO_FAMILY


# ------------------------------------------------------------------------------ taxonomy


def test_every_new_leaf_maps_into_the_fixed_eight_families() -> None:
    for mapping in (CICIOMT2024_LEAF_TO_FAMILY, EDGE_IIOTSET_LEAF_TO_FAMILY):
        assert set(mapping.values()) <= set(CLASS_NAMES)
        assert "Benign" in mapping.values()
    # 18 attacks + benign, with the 8 TCP/IP floods accepted under both known spellings.
    assert len(CICIOMT2024_LEAF_TO_FAMILY) == 27
    assert len(EDGE_IIOTSET_LEAF_TO_FAMILY) == 15  # 14 attacks + normal


def test_to_family_with_a_corpus_map_and_unknown_leaf() -> None:
    labels = pd.Series(["Normal", "DDoS_UDP", "Password", "Ransomware"])
    families = to_family(labels, EDGE_IIOTSET_LEAF_TO_FAMILY)
    assert list(families) == ["Benign", "DDoS", "BruteForce", "WebBased"]
    assert list(to_class_index(labels, EDGE_IIOTSET_LEAF_TO_FAMILY)) == [0, 1, 6, 7]
    with pytest.raises(ValueError, match="not present in the taxonomy"):
        to_family(pd.Series(["Normal", "Quantum"]), EDGE_IIOTSET_LEAF_TO_FAMILY)


def test_default_taxonomy_is_still_ciciot2023() -> None:
    assert list(to_family(pd.Series(["BenignTraffic", "Mirai-udpplain"]))) == ["Benign", "Mirai"]


# ------------------------------------------------------------------- harmonisation


#: The CICIoMT2024 README's feature table, in its own spelling (39 rows).
_IOMT_README_COLUMNS = (
    "Header Length", "Time-To-Live", "Rate", "fin flag number", "syn flag number",
    "rst flag number", "psh flag number", "ack flag number", "ece flag number",
    "cwr flag number", "syn count", "ack count", "fin count", "rst count", "IGMP", "HTTPS",
    "HTTP", "Telnet", "DNS", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv",
    "LLC", "Tot Sum", "Min", "Max", "AVG", "Std", "Tot Size", "IAT", "Number", "Variance",
    "Protocol Type",
)  # fmt: skip


def test_ciciomt2024_readme_table_is_the_canonical_vocabulary_verbatim() -> None:
    mapped = [canonical_name(c) for c in _IOMT_README_COLUMNS]
    assert None not in mapped
    assert sorted(mapped) == sorted(CANONICAL_COLUMNS)  # type: ignore[type-var]


#: The header every real CICIoMT2024 WiFi/MQTT CSV carries (verified 2026-09-20, 29 files).
_IOMT_CSV_HEADER = (
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate", "fin_flag_number",
    "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
    "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
)  # fmt: skip


def test_ciciomt2024_real_header_harmonises_to_the_full_selected_set() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({c: rng.random(5) for c in _IOMT_CSV_HEADER})
    result = harmonise_columns(frame, CICIOMT2024)
    assert "Time_To_Live" in result.frame.columns and "Duration" not in result.frame.columns
    assert result.missing_canonical == ()
    assert set(result.unmapped) == {"Srate", "Drate", "Magnitue", "Radius", "Covariance", "Weight"}
    assert selected_columns_available(result) == SELECTED_COLUMNS


def test_ciciomt2024_readme_spelling_also_harmonises() -> None:
    rng = np.random.default_rng(0)
    for spelling in (_IOMT_README_COLUMNS, CANONICAL_COLUMNS):
        frame = pd.DataFrame({c: rng.random(5) for c in spelling})
        result = harmonise_columns(frame, CICIOMT2024)
        assert result.missing_canonical == ()
        assert result.unmapped == ()


def test_ciciomt2024_discovery_skips_profiling_bluetooth_and_pcap(tmp_path: Path) -> None:
    for rel in (
        "WiFI_and_MQTT/attacks/CSV/train/Benign_train.pcap.csv",
        "WiFI_and_MQTT/attacks/CSV/test/DoS-SYN1_test.pcap.csv",
        "WiFI_and_MQTT/profiling/CSV/device.csv",
        "WiFI_and_MQTT/attacks/PCAP/x.csv",
        "Bluetooth/attacks/CSV/train/Benign_train.pcap.csv",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("a\n1\n")
    found = [str(p.relative_to(tmp_path)) for p in discover_parts(CICIOMT2024, tmp_path)]
    assert found == [
        "WiFI_and_MQTT/attacks/CSV/test/DoS-SYN1_test.pcap.csv",
        "WiFI_and_MQTT/attacks/CSV/train/Benign_train.pcap.csv",
    ]


def test_edge_iiotset_placeholder_artefact_is_neutralised() -> None:
    # The documented artefact: "0" in the normal branch, "0.0" in the attack branch.
    raw = pd.DataFrame(
        {
            "http.request.method": ["0", "GET", "0.0", "0.0"],
            "mqtt.protoname": ["0", "0", "0.0", "MQTT"],
            "tcp.len": [10, 20, 30, 40],
            "Attack_type": ["Normal", "Normal", "DDoS_TCP", "DDoS_TCP"],
        }
    )
    assert placeholder_collisions(raw) == {
        "http.request.method": ["0", "0.0"],
        "mqtt.protoname": ["0", "0.0"],
    }
    clean = normalise_placeholders(raw)
    assert placeholder_collisions(clean) == {}
    # Absent is absent in both branches; nothing about the label survives in the spelling.
    assert clean["http.request.method"].isna().tolist() == [True, False, True, True]
    assert clean["mqtt.protoname"].isna().tolist() == [True, True, True, False]
    assert clean["tcp.len"].tolist() == [10, 20, 30, 40]  # numeric untouched


def test_edge_iiotset_harmonisation_drops_identifiers_and_reports_gaps() -> None:
    raw = pd.DataFrame(
        {
            "frame.time": ["2021-01-01 00:00:00"],
            "ip.src_host": ["192.168.0.1"],
            "tcp.len": [10],
            "http.request.method": ["0"],
            "Attack_label": [0],
            "Attack_type": ["Normal"],
        }
    )
    result = harmonise_columns(raw, EDGE_IIOTSET)
    assert set(result.dropped) == {"frame.time", "ip.src_host", "Attack_label"}
    assert "Attack_type" in result.frame.columns
    # Nothing canonical exists at packet granularity before aggregation; all 39 are missing.
    assert len(result.missing_canonical) == len(CANONICAL_COLUMNS)


# ----------------------------------------------------------------- packet windows


def _packets() -> pd.DataFrame:
    # 7 packets: 4 Normal then 3 DDoS_TCP, one source file. Window of 2 packets.
    return pd.DataFrame(
        {
            "t": [0.0, 0.1, 0.3, 0.6, 1.0, 1.1, 1.3],
            "len": [100, 300, 200, 200, 50, 50, 70],
            "hdr": [20, 20, 20, 40, 20, 20, 20],
            "ttl": [64, 64, 128, 64, 32, 32, 32],
            "syn": [1, 0, 0, 0, 1, 1, 1],
            "ack": [0, 1, 1, 1, 0, 0, 0],
            "label": ["Normal"] * 4 + ["DDoS_TCP"] * 3,
            "source_file": ["a.csv"] * 7,
        }
    )


def test_aggregate_windows_never_cross_a_label_change_and_drop_partials() -> None:
    cols = PacketColumns(
        time_s="t", length="len", header_length="hdr", ttl="ttl", syn="syn", ack="ack"
    )
    out = aggregate_packet_windows(_packets(), cols, window_packets=2)
    # Normal: packets (0,1), (2,3). DDoS: (4,5); packet 6 is a partial window -> dropped.
    assert out["label"].tolist() == ["Normal", "Normal", "DDoS_TCP"]
    assert out["Number"].tolist() == [2, 2, 2]
    assert out["Tot sum"].tolist() == [400, 400, 100]
    assert out["Min"].tolist() == [100, 200, 50]
    assert out["Max"].tolist() == [300, 200, 50]
    assert out["Tot size"].tolist() == [200, 200, 50]
    assert out["Variance"].tolist() == [10000.0, 0.0, 0.0]
    np.testing.assert_allclose(out["IAT"], [0.1, 0.3, 0.1])
    assert out["Header_Length"].tolist() == [20, 30, 20]  # README: MEAN header length
    np.testing.assert_allclose(out["Rate"], [2 / 0.1, 2 / 0.3, 2 / 0.1])  # packets / second
    assert out["Time_To_Live"].tolist() == [64, 96, 32]
    assert out["syn_flag_number"].tolist() == [0.5, 0.0, 1.0]
    assert out["ack_count"].tolist() == [1, 2, 0]
    assert "TCP" not in out.columns  # not supplied -> not invented
    assert set(out.columns) & set(CANONICAL_COLUMNS) <= set(CANONICAL_COLUMNS)


def test_protocol_type_is_the_window_mode() -> None:
    packets = _packets()
    packets["proto"] = [6, 17, 17, 6, 1, 1, 1]
    cols = PacketColumns(time_s="t", length="len", protocol_number="proto")
    out = aggregate_packet_windows(packets, cols, window_packets=2)
    assert out["Protocol Type"].tolist() == [6.0, 6.0, 1.0]  # ties -> smallest, as pandas mode


def test_aggregate_windows_split_on_source_file_change() -> None:
    packets = _packets()
    packets["label"] = "Normal"
    packets.loc[3:, "source_file"] = "b.csv"
    out = aggregate_packet_windows(
        packets, PacketColumns(time_s="t", length="len"), window_packets=2
    )
    # a.csv has 3 packets -> one full window; b.csv has 4 -> two. Never (2,3) across files.
    assert out["source_file"].tolist() == ["a.csv", "b.csv", "b.csv"]
    assert out["Tot sum"].tolist() == [400, 250, 120]


def test_aggregate_windows_without_a_time_base_emit_no_time_features() -> None:
    # A corpus with no usable timestamps (Edge-IIoTset) supplies no IAT / Rate at all, rather
    # than a NaN or a 0 that would mark the windows it could not time.
    packets = _packets()
    packets["label"] = "Normal"
    out = aggregate_packet_windows(packets, PacketColumns(length="len"), window_packets=2)
    assert "IAT" not in out.columns and "Rate" not in out.columns
    assert out["Tot sum"].tolist() == [400, 400, 100]  # 7 packets -> 3 full windows
    harmonised = harmonise_columns(out, EDGE_IIOTSET)
    assert "IAT" in harmonised.missing_canonical
    assert "IAT" not in selected_columns_available(harmonised)


def test_edge_iiotset_packet_derivation_from_documented_columns() -> None:
    raw = pd.DataFrame(
        {
            "frame.time": ["2021-01-01 00:00:00.000", "2021-01-01 00:00:00.500"],
            "tcp.len": ["10", "0.0"],
            "tcp.flags": ["0x00000018", "0x00000002"],  # PSH|ACK, SYN
            "tcp.connection.syn": [0, 1],
            "tcp.connection.fin": [0, 0],
            "tcp.flags.ack": [1, 0],
            "udp.port": [np.nan, np.nan],
            "label": ["Normal", "Normal"],
            "source_file": ["x", "x"],
        }
    )
    frame, cols = edge_iiotset_packets(raw)
    assert frame["_pkt_length"].tolist() == [10.0, 0.0]
    assert frame["_pkt_psh"].tolist() == [1.0, 0.0]
    assert frame["_pkt_syn"].tolist() == [0.0, 1.0]
    assert frame["_pkt_ack"].tolist() == [1.0, 0.0]
    assert frame["_pkt_is_tcp"].tolist() == [1.0, 1.0]
    assert frame["_pkt_is_udp"].tolist() == [0.0, 0.0]
    assert cols.time_s is None  # frame.time is "6.0" for two classes: no time base, no IAT
    out = aggregate_packet_windows(frame, cols, window_packets=2)
    assert out["psh_flag_number"].tolist() == [0.5]
    assert out["TCP"].tolist() == [1.0]
    assert "Time_To_Live" not in out.columns  # Edge-IIoTset has no TTL; not invented
    assert "IAT" not in out.columns
    available = selected_columns_available(harmonise_columns(out, EDGE_IIOTSET))
    assert "Time_To_Live" not in available and "IAT" not in available
    assert "Tot sum" in available


def test_edge_iiotset_packet_derivation_after_placeholder_normalisation() -> None:
    # The real path: the raw frame is read as text, ``normalise_placeholders`` turns every
    # "0"/"0.0" into a missing value, and the derivation must treat a missing ``tcp.flags``
    # (a non-TCP packet) as "no flags", not crash on the NA reaching the hex parser.
    raw = pd.DataFrame(
        {
            "frame.time": ["2021-01-01 00:00:00.000", "2021-01-01 00:00:00.500"],
            "tcp.len": ["0", "12"],
            "tcp.flags": ["0", "0x00000018"],  # UDP packet has the placeholder, then PSH|ACK
            "tcp.connection.syn": ["0.0", "0"],
            "tcp.connection.fin": ["0", "0"],
            "tcp.flags.ack": ["0", "1"],
            "udp.port": ["0", "0"],  # DDoS_UDP rows: the port is the placeholder ...
            "udp.stream": ["7", "0"],  # ... and the stream id is what marks the packet as UDP
            "icmp.checksum": ["0", "0.0"],
            "label": ["Normal", "Normal"],
            "source_file": ["x", "x"],
        }
    )
    frame, cols = edge_iiotset_packets(normalise_placeholders(raw))
    assert frame["_pkt_is_tcp"].tolist() == [0.0, 1.0]
    assert frame["_pkt_is_udp"].tolist() == [1.0, 0.0]
    assert frame["_pkt_psh"].tolist() == [0.0, 1.0]
    assert frame["_pkt_length"].tolist() == [0.0, 12.0]
    assert frame["_pkt_protocol"].tolist() == [17.0, 6.0]
    out = aggregate_packet_windows(frame, cols, window_packets=2)
    assert out["Protocol Type"].tolist() == [6.0]  # mode of {17, 6}: ties -> smallest


def test_edge_iiotset_protocol_number_covers_icmp_and_non_ip() -> None:
    raw = pd.DataFrame(
        {
            "tcp.len": ["0", "0"],
            "tcp.flags": ["0", "0"],
            "tcp.connection.syn": ["0", "0"],
            "tcp.connection.fin": ["0", "0"],
            "tcp.flags.ack": ["0", "0"],
            "udp.port": ["0", "0"],
            "udp.stream": ["0", "0"],
            "icmp.checksum": ["0x1234", "0"],  # an ICMP packet, then an ARP packet (MITM)
            "label": ["DDoS_ICMP", "DDoS_ICMP"],
            "source_file": ["x", "x"],
        }
    )
    frame, _ = edge_iiotset_packets(normalise_placeholders(raw))
    assert frame["_pkt_protocol"].tolist() == [1.0, 0.0]
    assert frame["_pkt_is_tcp"].tolist() == [0.0, 0.0]


def test_edge_iiotset_packet_derivation_fails_loudly_on_unexpected_layout() -> None:
    with pytest.raises(ValueError, match="reconcile"):
        edge_iiotset_packets(pd.DataFrame({"tcp.len": [1]}))


# ------------------------------------------------------------------ fetch selection


def test_fetch_selection_keeps_small_classes_and_part_one_of_floods() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from fetch_ciciomt2024 import select_files

    listing = [
        "Benign_train.pcap.csv",
        "ARP_Spoofing_train.pcap.csv",
        "MQTT-Malformed_Data_train.pcap.csv",
        "Recon-VulScan_train.pcap.csv",
        *[f"TCP_IP-DDoS-ICMP{i}_train.pcap.csv" for i in range(1, 9)],
        *[f"TCP_IP-DoS-UDP{i}_train.pcap.csv" for i in range(1, 5)],
    ]
    kept = select_files(listing)
    assert kept == [
        "ARP_Spoofing_train.pcap.csv",
        "Benign_train.pcap.csv",
        "MQTT-Malformed_Data_train.pcap.csv",
        "Recon-VulScan_train.pcap.csv",
        "TCP_IP-DDoS-ICMP1_train.pcap.csv",
        "TCP_IP-DoS-UDP1_train.pcap.csv",
    ]
    assert len(select_files(listing, everything=True)) == len(listing)
    # The listing's spelling resolves to leaves the taxonomy knows.
    for name in kept:
        assert label_from_ciciomt2024_filename(Path(name)) in CICIOMT2024_LEAF_TO_FAMILY
