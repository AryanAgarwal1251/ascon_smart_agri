"""Unit tests for Phase 2 stratified, per-class-capped subsampling (Section III-B1, risk R1).

Uses small synthetic CSV parts written to ``tmp_path`` -- one directory tree per test, shaped
like the real UNB raw distribution (one directory per attack type, numbered ``*.pcap.csv``
parts within it) -- rather than the real multi-gigabyte corpus.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ascon_smart_agri.data.subsample import stratified_capped_subsample


def _write_csv(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("v\n" + "\n".join(str(v) for v in values) + "\n")


def test_label_comes_from_filename_not_directory(tmp_path: Path) -> None:
    # Mirrors the real raw distribution: the "Benign_Final" directory holds parts named
    # "BenignTraffic*.pcap.csv" -- the label must be the latter, not the former.
    _write_csv(tmp_path / "Benign_Final" / "BenignTraffic.pcap.csv", [1, 2])
    _write_csv(tmp_path / "Benign_Final" / "BenignTraffic1.pcap.csv", [3, 4])

    frame, counts = stratified_capped_subsample(
        tmp_path, target=4, per_class_cap=100, chunk_size=100, seed=0
    )

    assert set(frame["label"]) == {"BenignTraffic"}
    assert counts == {"BenignTraffic": 4}


def test_parts_are_read_in_natural_numeric_order_not_lexicographic(tmp_path: Path) -> None:
    # Lexicographic order would visit "X10" before "X2" (since '1' < '2'); natural order must
    # visit the unnumbered part, then "X2", then "X10".
    _write_csv(tmp_path / "X" / "X.pcap.csv", [0, 1])
    _write_csv(tmp_path / "X" / "X2.pcap.csv", [20, 21])
    _write_csv(tmp_path / "X" / "X10.pcap.csv", [100, 101])

    # Cap reached exactly after the first two parts in *numeric* order (2 + 2 = 4); the third
    # part must never be read under the natural ordering.
    frame, counts = stratified_capped_subsample(
        tmp_path, target=4, per_class_cap=4, chunk_size=100, seed=0
    )

    assert counts == {"X": 4}
    assert set(frame["v"]) == {0, 1, 20, 21}


def test_cap_stops_before_the_next_whole_part_but_trims_the_current_one(tmp_path: Path) -> None:
    # The first part alone (5 rows) already exceeds a cap of 3: the part is read in full (never
    # split mid-read), the second part is never opened, and the class is trimmed down to the cap.
    _write_csv(tmp_path / "Y" / "Y.pcap.csv", [1, 2, 3, 4, 5])
    _write_csv(tmp_path / "Y" / "Y1.pcap.csv", [99, 98, 97])

    frame, counts = stratified_capped_subsample(
        tmp_path, target=3, per_class_cap=3, chunk_size=100, seed=0
    )

    assert counts == {"Y": 3}
    assert (frame["source_file"] == str(Path("Y") / "Y.pcap.csv")).all()
    assert set(frame["v"]).issubset({1, 2, 3, 4, 5})


def test_rare_class_below_cap_keeps_every_row(tmp_path: Path) -> None:
    _write_csv(tmp_path / "Rare" / "Rare.pcap.csv", [1, 2, 3])

    frame, counts = stratified_capped_subsample(
        tmp_path, target=3, per_class_cap=1_000, chunk_size=100, seed=0
    )

    assert counts == {"Rare": 3}
    assert sorted(frame["v"]) == [1, 2, 3]


def test_subsample_is_deterministic_given_a_seed(tmp_path: Path) -> None:
    _write_csv(tmp_path / "Z" / "Z.pcap.csv", list(range(20)))

    frame1, counts1 = stratified_capped_subsample(
        tmp_path, target=10, per_class_cap=10, chunk_size=100, seed=7
    )
    frame2, counts2 = stratified_capped_subsample(
        tmp_path, target=10, per_class_cap=10, chunk_size=100, seed=7
    )

    pd.testing.assert_frame_equal(frame1, frame2)
    assert counts1 == counts2


def test_subsample_warns_when_total_deviates_far_from_target(tmp_path: Path) -> None:
    _write_csv(tmp_path / "Small" / "Small.pcap.csv", [1, 2])

    with pytest.warns(UserWarning, match="deviates from target"):
        stratified_capped_subsample(
            tmp_path, target=1_000_000, per_class_cap=100, chunk_size=100, seed=0
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"per_class_cap": 0}, "per_class_cap"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"target": 0}, "target"),
    ],
)
def test_subsample_rejects_non_positive_arguments(
    tmp_path: Path, kwargs: dict[str, int], match: str
) -> None:
    _write_csv(tmp_path / "A" / "A.pcap.csv", [1])
    base = {"target": 1, "per_class_cap": 1, "chunk_size": 1, "seed": 0}
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        stratified_capped_subsample(tmp_path, **base)  # type: ignore[arg-type]


def test_subsample_raises_when_no_csv_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        stratified_capped_subsample(tmp_path, target=1, per_class_cap=1, chunk_size=1, seed=0)
