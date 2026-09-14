"""Unit tests for the leaf-label -> attack-family taxonomy (Section III-B3).

The mapping is load-bearing for every per-class metric in Section III-I, so its shape (34 leaves,
8 classes, benign at index 0) is asserted here rather than assumed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ascon_smart_agri.data.taxonomy import (
    BENIGN_CLASS_INDEX,
    CLASS_NAMES,
    FAMILY_TO_INDEX,
    LEAF_TO_FAMILY,
    to_class_index,
    to_family,
)


def test_there_are_exactly_eight_classes() -> None:
    # C = 8 (benign + seven attack families), per Section III-B3 and ModelConfig's validator.
    assert len(CLASS_NAMES) == 8
    assert len(set(CLASS_NAMES)) == 8


def test_benign_is_class_index_zero() -> None:
    # Eq. (5)'s binary projection is exactly "y > 0", which depends on this.
    assert CLASS_NAMES[0] == "Benign"
    assert BENIGN_CLASS_INDEX == 0
    assert FAMILY_TO_INDEX["Benign"] == 0


def test_taxonomy_covers_thirty_four_leaves() -> None:
    # 33 leaf attack classes + benign, per Section III-B3.
    assert len(LEAF_TO_FAMILY) == 34
    attack_leaves = [leaf for leaf, fam in LEAF_TO_FAMILY.items() if fam != "Benign"]
    assert len(attack_leaves) == 33


def test_every_family_is_used_and_every_target_is_a_known_class() -> None:
    used = set(LEAF_TO_FAMILY.values())
    assert used == set(CLASS_NAMES)  # no empty family, no stray family name


def test_family_sizes_match_the_confirmed_partition() -> None:
    sizes = pd.Series(list(LEAF_TO_FAMILY.values())).value_counts().to_dict()
    assert sizes == {
        "DDoS": 12,
        "WebBased": 6,
        "Reconnaissance": 5,
        "DoS": 4,
        "Mirai": 3,
        "Spoofing": 2,
        "Benign": 1,
        "BruteForce": 1,
    }


def test_ddos_and_dos_are_separate_families() -> None:
    # Table I merges them into one row; Section III-B3's C=8 requires them split (see the
    # module docstring). Guard the reconciliation so it cannot be silently undone.
    assert LEAF_TO_FAMILY["DDoS-SYN_Flood"] == "DDoS"
    assert LEAF_TO_FAMILY["DoS-SYN_Flood"] == "DoS"
    assert FAMILY_TO_INDEX["DDoS"] != FAMILY_TO_INDEX["DoS"]


def test_to_family_maps_leaves_to_families() -> None:
    labels = pd.Series(["BenignTraffic", "DDoS-ICMP_Flood", "XSS", "Recon-PingSweep"])

    families = to_family(labels)

    assert list(families) == ["Benign", "DDoS", "WebBased", "Reconnaissance"]


def test_to_class_index_maps_leaves_to_fixed_indices() -> None:
    labels = pd.Series(["BenignTraffic", "DDoS-ICMP_Flood", "Mirai-udpplain"])

    indices = to_class_index(labels)

    assert list(indices) == [0, FAMILY_TO_INDEX["DDoS"], FAMILY_TO_INDEX["Mirai"]]
    assert indices.dtype == "int64"


def test_unknown_labels_raise_rather_than_falling_back() -> None:
    labels = pd.Series(["BenignTraffic", "SomeNewAttack2027"])

    with pytest.raises(ValueError, match="not present in the CICIoT2023 taxonomy"):
        to_family(labels)


def test_binary_projection_is_y_greater_than_zero() -> None:
    labels = pd.Series(["BenignTraffic", "DDoS-TCP_Flood", "BenignTraffic", "SqlInjection"])

    is_attack = to_class_index(labels) > BENIGN_CLASS_INDEX

    assert list(is_attack) == [False, True, False, True]
