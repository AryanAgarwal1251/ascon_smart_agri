"""Dataset registry and schema harmonisation for multi-dataset training (Phase 9 plan).

The "CICIoT2023 only" rule was lifted on 2026-09-19: the detector must generalise across two
to three IoT intrusion corpora so it does not fail on attacks that are in scope but absent
from CICIoT2023. The hard part is not the number of datasets but the *feature language*: the
GRU consumes the 16 CICIoT2023 columns selected in Phase 2, produced by UNB's DPKT
packet-window extractor. Another corpus only helps if its rows can be expressed in that
language. This module is the single place that translation lives.

Three corpora are registered:

* **ciciot2023** --- the reference. Its raw-distribution column vocabulary (39 features, from
  ``artifacts/phase1_characterization_report_raw.json``) IS the canonical schema.
* **ciciomt2024** --- same lab, same extractor. **Verified on the real files (2026-09-20):**
  every CSV carries one 45-column header = the 39 canonical columns with TTL spelled
  ``Duration`` (its values are 64/128: a TTL) plus six extras that CICIoT2023's raw
  distribution dropped (``Srate``, ``Drate``, ``Magnitue``, ``Radius``, ``Covariance``,
  ``Weight``). The README's 39-row feature table describes the *same* extractor in its own
  spelling; the CSV header is the Kaggle-era layout. Net effect: a full drop-in --- all 16
  selected features present after the ``Duration -> Time_To_Live`` alias; the extras are
  reported as ``unmapped`` and never used. Files: ``WiFI_and_MQTT/attacks/CSV/{train,test}``,
  label in the filename; ``profiling/`` is benign device-fingerprinting traffic, not used.
* **edge_iiotset** --- the most agriculture-relevant corpus (soil-moisture, temperature, pH,
  water-level sensors over MQTT) but a different schema: one row per *packet* with Wireshark
  field names, and the label in an ``Attack_type`` column. Its rows must be aggregated into
  packet windows (``data/packet_windows.py``) before they speak the canonical language, and
  only a subset of the canonical columns is recoverable. It also carries a documented leakage
  artefact --- see :func:`normalise_placeholders`. **Registered but not a training corpus**
  (``not_for_training``, decided 2026-09-22 on the first full Phase 9 run): the windows
  rebuilt from its packet fields come out near-constant (99 % TCP, 93 % ACK-flagged), 81 %
  of them are exact duplicates, and the model trained *on* them scores macro-F1 0.12 with
  FPR 0.93 on its own test split -- while being the only corpus that cannot supply
  ``Header_Length``, ``IAT`` and ``Time_To_Live``, Phase 2's top-ranked features. It stays
  in the registry so ``asa characterize --dataset edge_iiotset`` and the placeholder check
  keep working; :data:`TRAINING_CORPORA` is what the Phase 9 driver trains on.

**Every column alias and leaf label below for the two new corpora was written from the
published documentation, not from the files.** The first thing to do when a corpus arrives is
``asa characterize --dataset <name>``: the report lists the real columns and label vocabulary,
and :func:`harmonise_columns` / :func:`taxonomy.to_family` fail loudly on anything the
registry did not anticipate. Do not train on a corpus whose characterisation has not been
reconciled with this file.

Leakage discipline is per corpus and unchanged: characterise -> dedup BEFORE split -> the R3
gate (``tests/test_leakage.py``), then windows only within one source file.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from .taxonomy import (
    CICIOMT2024_LEAF_TO_FAMILY,
    EDGE_IIOTSET_LEAF_TO_FAMILY,
    LEAF_TO_FAMILY,
)

_NAME_SEPARATORS = re.compile(r"[\s_\-]+")

#: The canonical feature vocabulary: CICIoT2023's raw-distribution columns, verbatim.
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
)

_CANONICAL_BY_KEY: Final[dict[str, str]] = {
    _NAME_SEPARATORS.sub("", c).lower(): c for c in CANONICAL_COLUMNS
}

#: The F = 16 columns Phase 2 selected (artifacts/manifest_phase4_default.json), in rank order.
SELECTED_COLUMNS: Final[tuple[str, ...]] = (
    "Tot sum",
    "Header_Length",
    "Tot size",
    "Max",
    "Variance",
    "IAT",
    "TCP",
    "ack_count",
    "Time_To_Live",
    "syn_flag_number",
    "ack_flag_number",
    "UDP",
    "Protocol Type",
    "psh_flag_number",
    "fin_flag_number",
    "Min",
)


def canonical_name(column: str) -> str | None:
    """The canonical column whose name matches ``column`` up to case and separators.

    UNB's READMEs write "Header Length", "Time-To-Live", "Tot Sum"; the CICIoT2023 raw CSVs
    write ``Header_Length``, ``Time_To_Live``, ``Tot sum``. Same feature, three spellings.
    Returns ``None`` for a name that matches no canonical column.
    """
    return _CANONICAL_BY_KEY.get(_NAME_SEPARATORS.sub("", column).lower())


_CSV_SUFFIX = re.compile(r"(?:\.pcap)?\.csv$", re.IGNORECASE)
_TRAILING_DIGITS = re.compile(r"\d+$")
_SPLIT_SUFFIX = re.compile(r"[_-](train|test)$", re.IGNORECASE)

#: Spellings that all mean "field absent" in a CSV written from a Wireshark/Zeek export. The
#: Edge-IIoTset build wrote "0" in its normal-traffic branch and "0.0" in its attack branch,
#: so a one-hot of the raw strings encodes file provenance and scores 1.0000 on its own
#: (arXiv 2608.15761). Mapping every spelling to one missing value removes that channel.
PLACEHOLDER_SPELLINGS: Final[frozenset[str]] = frozenset({"0", "0.0", "", "nan", "NaN", "-"})


def label_from_ciciot2023_filename(path: Path) -> str:
    """``DDoS-ICMP_Flood3.pcap.csv`` -> ``DDoS-ICMP_Flood`` (mirrors ``data/subsample.py``)."""
    stem = _CSV_SUFFIX.sub("", path.name)
    stem = _TRAILING_DIGITS.sub("", stem)
    return stem.rstrip("-_")


def label_from_ciciomt2024_filename(path: Path) -> str:
    """``TCP_IP-DDoS-ICMP2_train.pcap.csv`` -> ``TCP_IP-DDoS-ICMP``; ``Benign_test.pcap.csv``
    -> ``Benign``.

    UNB ships a pre-made train/test split; the split token is stripped because this project
    pools everything and re-splits after dedup (R3). Any label the taxonomy does not know is
    a loud error downstream, never a silent new class.
    """
    stem = _CSV_SUFFIX.sub("", path.name)
    stem = _SPLIT_SUFFIX.sub("", stem)
    stem = _TRAILING_DIGITS.sub("", stem)
    return stem.rstrip("-_")


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the pipeline needs to know to read one corpus into the canonical language."""

    name: str
    default_root: Path
    citation: str
    #: "window": one row is already an extractor window (CICIoT2023 family).
    #: "packet": one row is a packet; ``data/packet_windows.py`` must aggregate first.
    granularity: Literal["window", "packet"]
    label_source: Literal["filename", "column"]
    leaf_to_family: dict[str, str]
    label_column: str | None = None
    label_from_filename: Callable[[Path], str] | None = None
    #: Glob relative to the root that finds the feature CSVs (default: every CSV underneath).
    csv_glob: str = "**/*.csv"
    #: Path components (case-insensitive) under which CSVs are NOT part of the benchmark.
    exclude_dirs: tuple[str, ...] = ()
    #: dataset column name -> canonical column name, where the spelling differs.
    column_aliases: dict[str, str] = field(default_factory=dict)
    #: Columns that are identifiers or free text and must never become features.
    drop_columns: tuple[str, ...] = ()
    #: Set when the corpus is known to need :func:`normalise_placeholders`.
    normalise_placeholders: bool = False
    #: Non-empty = the corpus is registered (characterisable, harmonisable) but excluded from
    #: training, and this is why. The Phase 9 driver refuses to train on such a corpus unless
    #: it is named explicitly, so the exclusion is a recorded decision, not a silent default.
    not_for_training: str = ""
    notes: str = ""

    def label_of(self, path: Path, frame: pd.DataFrame) -> pd.Series[str]:
        """The leaf label of every row of ``frame`` read from ``path``."""
        if self.label_source == "filename":
            assert self.label_from_filename is not None
            return pd.Series(self.label_from_filename(path), index=frame.index, dtype=str)
        assert self.label_column is not None
        if self.label_column not in frame.columns:
            raise ValueError(f"{path}: expected label column {self.label_column!r}")
        return frame[self.label_column].astype(str)


CICIOT2023: Final = DatasetSpec(
    name="ciciot2023",
    default_root=Path("data/ciciot2023_raw"),
    citation="Neto et al., CICIoT2023, Sensors 23(13), 2023",
    granularity="window",
    label_source="filename",
    label_from_filename=label_from_ciciot2023_filename,
    leaf_to_family=LEAF_TO_FAMILY,
    notes="Reference corpus; its raw column vocabulary is CANONICAL_COLUMNS.",
)

CICIOMT2024: Final = DatasetSpec(
    name="ciciomt2024",
    default_root=Path("data/ciciomt2024"),
    citation="Dadkhah et al., CICIoMT2024, Internet of Things 28, 2024",
    granularity="window",
    label_source="filename",
    label_from_filename=label_from_ciciomt2024_filename,
    leaf_to_family=CICIOMT2024_LEAF_TO_FAMILY,
    # WiFI_and_MQTT/attacks/CSV/{train,test}. Bluetooth has a different (BLE) feature set and
    # profiling/ is benign device-fingerprinting traffic; neither is part of the benchmark.
    exclude_dirs=("bluetooth", "profiling", "pcap"),
    # Verified against the real header: "Duration" holds the TTL (values 64/128), exactly as
    # in the Kaggle-era CICIoT2023 layout this CSV format comes from.
    column_aliases={"Duration": "Time_To_Live"},
    notes=(
        "Same DPKT extractor as CICIoT2023. Real CSV header = 45 columns: the 39 canonical "
        "(TTL as 'Duration') + Srate, Drate, Magnitue, Radius, Covariance, Weight (unmapped, "
        "unused). Labels come from filenames; UNB's train/test split is pooled and re-split "
        "after dedup (R3)."
    ),
)

EDGE_IIOTSET: Final = DatasetSpec(
    name="edge_iiotset",
    default_root=Path("data/edge_iiotset"),
    citation="Ferrag et al., Edge-IIoTset, IEEE Access 10, 2022",
    granularity="packet",
    label_source="column",
    label_column="Attack_type",
    leaf_to_family=EDGE_IIOTSET_LEAF_TO_FAMILY,
    csv_glob="**/DNN-EdgeIIoT-dataset.csv",  # the full set; ML-EdgeIIoT is a balanced subset
    drop_columns=(
        # identifiers, addresses, timestamps and free text: provenance, never behaviour
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "arp.src.proto_ipv4",
        "arp.dst.proto_ipv4",
        "http.file_data",
        "http.request.full_uri",
        "http.request.uri.query",
        "http.referer",
        "tcp.options",
        "tcp.payload",
        "tcp.srcport",
        "tcp.dstport",
        "udp.port",
        "mqtt.msg",
        "mqtt.topic",
        "dns.qry.name",
        "Attack_label",  # binary twin of Attack_type; the family map is derived from the latter
    ),
    normalise_placeholders=True,
    not_for_training=(
        "Packet-field reconstruction of the window features is near-constant (99 % TCP, "
        "93 % ACK) and 81 % duplicate; in-distribution macro-F1 0.12 / FPR 0.93 on the "
        "2026-09-21 run (artifacts/manifest_phase9_generalised.json); and it alone forces "
        "Header_Length, IAT and Time_To_Live out of the feature set. Excluded 2026-09-22."
    ),
    notes=(
        "One row per packet (Wireshark fields). Must go through data/packet_windows.py. "
        "Carries the '0' vs '0.0' provenance artefact (arXiv 2608.15761); "
        "normalise_placeholders neutralises it and the characterisation report must confirm."
    ),
)

REGISTRY: Final[dict[str, DatasetSpec]] = {
    spec.name: spec for spec in (CICIOT2023, CICIOMT2024, EDGE_IIOTSET)
}

#: The corpora the generalised detector is trained on: every registered corpus without a
#: ``not_for_training`` reason. Leave-one-dataset-out runs over these.
TRAINING_CORPORA: Final[tuple[str, ...]] = tuple(
    name for name, spec in REGISTRY.items() if not spec.not_for_training
)


def get_dataset(name: str) -> DatasetSpec:
    """Look a corpus up by name, listing the known ones on a miss."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}") from None


def discover_parts(spec: DatasetSpec, root: Path | None = None) -> list[Path]:
    """The CSV parts of a corpus under ``root`` (default: the spec's root), sorted."""
    base = root if root is not None else spec.default_root
    excluded = {d.lower() for d in spec.exclude_dirs}
    parts = sorted(
        p
        for p in base.glob(spec.csv_glob)
        if not excluded & {part.lower() for part in p.relative_to(base).parts[:-1]}
    )
    if not parts:
        raise FileNotFoundError(f"no files matching {spec.csv_glob!r} under {base}")
    return parts


def _is_text(series: pd.Series[Any]) -> bool:
    """Object or string dtype (pandas >= 3 reads text as ``str``, older versions as ``object``)."""
    return bool(series.dtype == object or pd.api.types.is_string_dtype(series))


def normalise_placeholders(frame: pd.DataFrame) -> pd.DataFrame:
    """Map every "absent field" spelling in object columns to one missing value.

    Returns a copy. Numeric columns are untouched: a genuine 0 in a count column is data.
    """
    out = frame.copy()
    for col in out.columns:
        if _is_text(out[col]):
            series = out[col].astype(str).str.strip()
            absent = series.isin(PLACEHOLDER_SPELLINGS).to_numpy()
            out[col] = series.mask(absent, other=None)
    return out


def placeholder_spellings(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Which placeholder spellings each text column contains (possibly just one, or none).

    The per-chunk ingredient of :func:`placeholder_collisions`: a streaming scan unions these
    over every chunk of every file and judges the collision at the end, because the two
    spellings of an artefact that encodes file provenance typically live in *different* files
    and never share a chunk.
    """
    found: dict[str, set[str]] = {}
    for col in frame.columns:
        if not _is_text(frame[col]):
            continue
        present = set(frame[col].dropna().astype(str).str.strip().unique()) & PLACEHOLDER_SPELLINGS
        if present:
            found[str(col)] = present
    return found


def placeholder_collisions(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Object columns in which more than one placeholder spelling occurs, and which ones.

    This is the check that would have caught the Edge-IIoTset artefact before training: a
    column whose "absent" value is spelled two ways is encoding where its rows came from.
    Reported by ``asa characterize`` for every corpus (over the whole corpus, not per chunk).
    """
    return {
        col: sorted(present)
        for col, present in placeholder_spellings(frame).items()
        if len(present) > 1
    }


@dataclass(frozen=True)
class Harmonised:
    """A frame in the canonical language plus what could not be translated."""

    frame: pd.DataFrame
    missing_canonical: tuple[str, ...]  # canonical columns this corpus cannot provide
    unmapped: tuple[str, ...]  # corpus columns kept as-is because no alias matched
    dropped: tuple[str, ...]


def harmonise_columns(frame: pd.DataFrame, spec: DatasetSpec) -> Harmonised:
    """Rename aliased columns, drop identifiers, and report the canonical gaps.

    Never invents a column: a canonical feature the corpus cannot supply is *missing* and is
    reported, so feature selection is re-run over the intersection rather than over NaNs.
    """
    dropped = tuple(c for c in spec.drop_columns if c in frame.columns)
    out = frame.drop(columns=list(dropped))
    renames = {src: dst for src, dst in spec.column_aliases.items() if src in out.columns}
    for col in out.columns:
        if col not in renames and col not in CANONICAL_COLUMNS:
            canonical = canonical_name(str(col))
            if canonical is not None and canonical not in out.columns:
                renames[str(col)] = canonical
    out = out.rename(columns=renames)
    if spec.normalise_placeholders:
        out = normalise_placeholders(out)
    present = set(out.columns)
    missing = tuple(c for c in CANONICAL_COLUMNS if c not in present)
    unmapped = tuple(
        c for c in out.columns if c not in CANONICAL_COLUMNS and c not in {"label", "source_file"}
    )
    return Harmonised(frame=out, missing_canonical=missing, unmapped=unmapped, dropped=dropped)


def selected_columns_available(harmonised: Harmonised) -> tuple[str, ...]:
    """Which of Phase 2's 16 selected columns this corpus provides after harmonisation."""
    return tuple(c for c in SELECTED_COLUMNS if c in harmonised.frame.columns)
