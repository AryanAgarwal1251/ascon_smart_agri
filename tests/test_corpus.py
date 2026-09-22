"""``data/corpus.py``: every registered corpus comes out in one shape (Phase 9 plan, §3 step 2).

Synthetic files under ``tmp_path`` shaped like the real ones: CICIoMT2024's ``<leaf>_train``/
``<leaf>_test`` sharded CSVs with the README header, and an Edge-IIoTset file with the
Wireshark columns the packet derivation reads plus the ``Attack_type`` label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ascon_smart_agri.data.corpus import load_corpus
from ascon_smart_agri.data.datasets import CANONICAL_COLUMNS, CICIOMT2024, EDGE_IIOTSET
from ascon_smart_agri.data.subsample import stratified_capped_subsample
from ascon_smart_agri.data.taxonomy import CLASS_NAMES, to_class_index

_MT_HEADER = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
    "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG",
    "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]  # fmt: skip


def _write_mt(root: Path, name: str, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.random((rows, len(_MT_HEADER))), columns=_MT_HEADER)
    path = root / "WiFI_and_MQTT" / "attacks" / "CSV" / ("train" if "train" in name else "test")
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / name, index=False)


def test_ciciomt2024_loads_through_the_capped_subsample_with_its_own_labels(
    tmp_path: Path,
) -> None:
    _write_mt(tmp_path, "Benign_train.pcap.csv", 30, 0)
    _write_mt(tmp_path, "Benign_test.pcap.csv", 10, 1)
    _write_mt(tmp_path, "TCP_IP-DDoS-ICMP1_train.pcap.csv", 40, 2)
    _write_mt(tmp_path, "TCP_IP-DDoS-ICMP2_train.pcap.csv", 40, 3)
    _write_mt(tmp_path, "Recon-Ping_Sweep_train.pcap.csv", 5, 4)

    frame, per_leaf = load_corpus(
        CICIOMT2024, tmp_path, per_class_cap=25, chunk_size=7, window_packets=10, seed=0
    )
    # Shards and the UNB train/test files pool under one leaf; the cap applies per leaf.
    assert per_leaf == {"Benign": 25, "Recon-Ping_Sweep": 5, "TCP_IP-DDoS-ICMP": 25}
    assert len(frame) == 55
    assert list(frame.columns[-2:]) == ["label", "source_file"]
    assert all(c in CANONICAL_COLUMNS for c in frame.columns[:-2])
    assert "Duration" not in frame.columns and "Time_To_Live" in frame.columns  # alias applied
    # Leaves map into the fixed eight families through the corpus's own taxonomy.
    families = to_class_index(frame["label"], CICIOMT2024.leaf_to_family)
    assert set(families.map(lambda i: CLASS_NAMES[i])) == {"Benign", "DDoS", "Reconnaissance"}


def test_subsampler_default_label_rule_is_unchanged(tmp_path: Path) -> None:
    # The CICIoT2023 rule: strip the shard number; no train/test suffix to worry about.
    for name in ("DDoS-ICMP_Flood1.csv", "DDoS-ICMP_Flood2.csv", "Benign.csv"):
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(tmp_path / name, index=False)
    frame, counts = stratified_capped_subsample(
        tmp_path, target=6, per_class_cap=10, chunk_size=10, seed=0
    )
    assert counts == {"Benign": 2, "DDoS-ICMP_Flood": 4}
    assert len(frame) == 6


def test_edge_iiotset_loads_as_capped_windows(tmp_path: Path) -> None:
    n = 60
    raw = pd.DataFrame(
        {
            "frame.time": ["6.0"] * n,  # the real file's broken time base: must not matter
            "ip.src_host": ["192.168.0.1"] * n,
            "tcp.len": ["0"] * 30 + ["12"] * 30,
            "tcp.flags": ["0"] * 30 + ["0x00000018"] * 30,
            "tcp.connection.syn": ["0"] * n,
            "tcp.connection.fin": ["0"] * n,
            "tcp.flags.ack": ["0"] * 30 + ["1"] * 30,
            "udp.port": ["0"] * n,
            "udp.stream": ["3"] * 30 + ["0"] * 30,
            "icmp.checksum": ["0"] * n,
            "http.file_data": ["0.0"] * n,  # a text column with the other placeholder spelling
            "Attack_type": ["DDoS_UDP"] * 30 + ["Normal"] * 30,
        }
    )
    raw.to_csv(tmp_path / "DNN-EdgeIIoT-dataset.csv", index=False)

    frame, per_leaf = load_corpus(
        EDGE_IIOTSET, tmp_path, per_class_cap=2, chunk_size=10, window_packets=10, seed=0
    )
    # 30 packets per label -> 3 windows each -> capped to 2 each.
    assert per_leaf == {"DDoS_UDP": 2, "Normal": 2}
    assert len(frame) == 4
    assert "IAT" not in frame.columns and "Number" in frame.columns
    by_label = frame.groupby("label")[["UDP", "TCP", "Protocol Type"]].mean()
    assert by_label.loc["DDoS_UDP"].tolist() == [1.0, 0.0, 17.0]
    assert by_label.loc["Normal"].tolist() == [0.0, 1.0, 6.0]
    assert frame["source_file"].unique().tolist() == ["DNN-EdgeIIoT-dataset.csv"]


def test_granularity_mismatch_is_an_error(tmp_path: Path) -> None:
    from ascon_smart_agri.data.corpus import load_packet_corpus, load_window_corpus

    with pytest.raises(ValueError, match="not a packet corpus"):
        load_packet_corpus(CICIOMT2024, tmp_path, per_class_cap=1, window_packets=1, seed=0)
    with pytest.raises(ValueError, match="not a window corpus"):
        load_window_corpus(EDGE_IIOTSET, tmp_path, per_class_cap=1, chunk_size=1, seed=0)
