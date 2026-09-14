"""Leaf-label to attack-family taxonomy (Phase 2/3, Section III-B3).

The primary task is C = 8 classes: benign plus seven attack families. CICIoT2023 ships 33 leaf
attack classes plus benign; at IR ~ 5751 the rarest leaves cannot support a defensible macro-F1
estimate, and the family is in any case the actionable granularity for an operator (III-B3).
This module is the single place that mapping lives, so no other module invents its own.

**Reconciled discrepancy (flagged per Golden Rule 1, confirmed with the user).** The paper
states the taxonomy twice and the two statements disagree. Section III-B3's prose says "benign
plus seven attack families" (C = 8), and ``configs/base.py`` hard-validates ``n_classes == 8``
while Eq. (19)'s 33,800-parameter count assumes C = 8. But Table I lists only **six** family
rows, because it merges DDoS and DoS into a single ``DDoS / DoS`` row. Splitting that merged row
into two families is the only reading that satisfies the prose, the validator and Eq. (19)
simultaneously, and it is what :data:`CLASS_NAMES` encodes. The alternative -- following Table I
literally at C = 7 -- was put to the user and rejected, since it would contradict III-B3 and
require changing both the validator and the reference parameter count.

Two placements inside that partition are judgement calls rather than deductions, and both follow
the canonical CICIoT2023 grouping: ``VulnerabilityScan`` is reconnaissance (it is a scan, and
Table I glosses reconnaissance as "deployment mapped; precursor to targeted compromise") and
``Backdoor_Malware`` is web-based (there is no eighth family for it to occupy).

Class index order is fixed and load-bearing: **benign is index 0**, so Eq. (5)'s binary
projection of the eight-class output is exactly ``y > 0`` and never depends on a lookup. The
seven attack families then follow Table I's own row order, with the merged row split in place.
"""

from __future__ import annotations

import pandas as pd

#: The C = 8 classes in fixed index order. Index 0 is benign (see the module docstring).
CLASS_NAMES: tuple[str, ...] = (
    "Benign",
    "DDoS",
    "DoS",
    "Mirai",
    "Reconnaissance",
    "Spoofing",
    "BruteForce",
    "WebBased",
)

BENIGN_CLASS_INDEX = 0

#: Every CICIoT2023 leaf label produced by ``data/subsample.py`` -> its family.
LEAF_TO_FAMILY: dict[str, str] = {
    # Benign (1 leaf)
    "BenignTraffic": "Benign",
    # DDoS (12 leaves)
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    # DoS (4 leaves)
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    # Mirai (3 leaves)
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Reconnaissance (5 leaves)
    "Recon-HostDiscovery": "Reconnaissance",
    "Recon-OSScan": "Reconnaissance",
    "Recon-PingSweep": "Reconnaissance",
    "Recon-PortScan": "Reconnaissance",
    "VulnerabilityScan": "Reconnaissance",
    # Spoofing (2 leaves)
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    # Brute force (1 leaf)
    "DictionaryBruteForce": "BruteForce",
    # Web-based (6 leaves)
    "Backdoor_Malware": "WebBased",
    "BrowserHijacking": "WebBased",
    "CommandInjection": "WebBased",
    "SqlInjection": "WebBased",
    "Uploading_Attack": "WebBased",
    "XSS": "WebBased",
}

#: family name -> its index in :data:`CLASS_NAMES`.
FAMILY_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def to_family(labels: pd.Series[str]) -> pd.Series[str]:
    """Map leaf labels to family names, raising on any label not in :data:`LEAF_TO_FAMILY`.

    Unknown labels are an error, never a silent fallback class: a leaf this module has not been
    told about would otherwise be absorbed into some family and corrupt the per-class metrics
    Section III-I depends on.
    """
    unknown = sorted(set(labels.astype(str)) - set(LEAF_TO_FAMILY))
    if unknown:
        raise ValueError(f"labels not present in the CICIoT2023 taxonomy: {unknown}")
    return labels.astype(str).map(LEAF_TO_FAMILY)


def to_class_index(labels: pd.Series[str]) -> pd.Series[int]:
    """Map leaf labels straight to the fixed class index used by the detector's logits."""
    return to_family(labels).map(FAMILY_TO_INDEX).astype("int64")
