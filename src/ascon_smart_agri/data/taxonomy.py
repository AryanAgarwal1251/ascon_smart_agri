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

# ---------------------------------------------------------------------------------------
# Multi-dataset generalisation (2026-09-19 redirect; see data/datasets.py). The C = 8 family
# partition is kept fixed so every corpus trains the same head; each new corpus's leaves are
# mapped INTO it. Leaf names below come from the published documentation and must be checked
# against the ``label_counts`` of that corpus's characterisation report before training.
# ---------------------------------------------------------------------------------------

#: CICIoMT2024 (WiFI_and_MQTT), leaves derived from filenames by
#: ``datasets.label_from_ciciomt2024_filename``. 18 attacks + benign, per the dataset README
#: (categories DDoS, DoS, Recon, MQTT, Spoofing). The TCP/IP floods appear as
#: ``TCP_IP-DDoS-ICMP`` in filenames and as ``DDoS-ICMP`` in the README's charts; both
#: spellings are mapped so whichever the CSVs use resolves.
#: Judgement call, flagged: ``MQTT-Malformed_Data`` is an application-layer protocol-abuse
#: attack with no family of its own here; it goes to WebBased with the other application-layer
#: attacks (the same reasoning that placed Backdoor_Malware there), not to DoS.
CICIOMT2024_LEAF_TO_FAMILY: dict[str, str] = {
    "Benign": "Benign",
    "ARP_Spoofing": "Spoofing",
    "MQTT-DDoS-Connect_Flood": "DDoS",
    "MQTT-DDoS-Publish_Flood": "DDoS",
    "MQTT-DoS-Connect_Flood": "DoS",
    "MQTT-DoS-Publish_Flood": "DoS",
    "MQTT-Malformed_Data": "WebBased",
    "Recon-OS_Scan": "Reconnaissance",
    "Recon-Ping_Sweep": "Reconnaissance",
    "Recon-Port_Scan": "Reconnaissance",
    "Recon-VulScan": "Reconnaissance",
    "TCP_IP-DDoS-ICMP": "DDoS",
    "TCP_IP-DDoS-SYN": "DDoS",
    "TCP_IP-DDoS-TCP": "DDoS",
    "TCP_IP-DDoS-UDP": "DDoS",
    "TCP_IP-DoS-ICMP": "DoS",
    "TCP_IP-DoS-SYN": "DoS",
    "TCP_IP-DoS-TCP": "DoS",
    "TCP_IP-DoS-UDP": "DoS",
    "DDoS-ICMP": "DDoS",
    "DDoS-SYN": "DDoS",
    "DDoS-TCP": "DDoS",
    "DDoS-UDP": "DDoS",
    "DoS-ICMP": "DoS",
    "DoS-SYN": "DoS",
    "DoS-TCP": "DoS",
    "DoS-UDP": "DoS",
}

#: Edge-IIoTset ``Attack_type`` values. 14 attacks + Normal.
#: Judgement calls, flagged: ``Ransomware`` is malware and follows Backdoor into WebBased
#: (there is no malware family); ``Fingerprinting`` is a scan and is reconnaissance;
#: ``Password`` is a brute-force login attack; ``MITM`` is ARP/DNS spoofing by mechanism.
EDGE_IIOTSET_LEAF_TO_FAMILY: dict[str, str] = {
    "Normal": "Benign",
    "DDoS_HTTP": "DDoS",
    "DDoS_ICMP": "DDoS",
    "DDoS_TCP": "DDoS",
    "DDoS_UDP": "DDoS",
    "Fingerprinting": "Reconnaissance",
    "Port_Scanning": "Reconnaissance",
    "Vulnerability_scanner": "Reconnaissance",
    "MITM": "Spoofing",
    "Password": "BruteForce",
    "Backdoor": "WebBased",
    "Ransomware": "WebBased",
    "SQL_injection": "WebBased",
    "Uploading": "WebBased",
    "XSS": "WebBased",
}


def to_family(
    labels: pd.Series[str], leaf_to_family: dict[str, str] | None = None
) -> pd.Series[str]:
    """Map leaf labels to family names, raising on any label not in the map.

    ``leaf_to_family`` defaults to CICIoT2023's :data:`LEAF_TO_FAMILY`; pass a corpus's own map
    (``DatasetSpec.leaf_to_family``) for the others. Unknown labels are an error, never a silent
    fallback class: a leaf this module has not been told about would otherwise be absorbed into
    some family and corrupt the per-class metrics Section III-I depends on.
    """
    mapping = LEAF_TO_FAMILY if leaf_to_family is None else leaf_to_family
    unknown = sorted(set(labels.astype(str)) - set(mapping))
    if unknown:
        raise ValueError(f"labels not present in the taxonomy: {unknown}")
    families = labels.astype(str).map(mapping)
    bad = sorted(set(families) - set(CLASS_NAMES))
    if bad:
        raise ValueError(f"taxonomy maps onto families outside CLASS_NAMES: {bad}")
    return families


def to_class_index(
    labels: pd.Series[str], leaf_to_family: dict[str, str] | None = None
) -> pd.Series[int]:
    """Map leaf labels straight to the fixed class index used by the detector's logits."""
    return to_family(labels, leaf_to_family).map(FAMILY_TO_INDEX).astype("int64")
