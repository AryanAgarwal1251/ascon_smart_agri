"""Unit tests for Phase 1 dataset characterisation (Section III-B).

Uses small synthetic CSV parts written to ``tmp_path`` rather than the real multi-gigabyte
CICIoT2023 corpus, so these run fast and need no dataset on disk. They exercise the same
streaming code path (``pandas.read_csv(chunksize=...)``) with a tiny chunk size to force
multiple chunks despite the tiny fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ascon_smart_agri.data.characterize import characterize_dataset


def _write_csv(path: Path, rows: list[str], header: str) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_characterize_pools_multiple_csv_parts(tmp_path: Path) -> None:
    # Two "parts" (as the real corpus, and this project's pre-split mirror, both are) that must
    # be pooled into one corpus before any statistic is computed.
    part_a = tmp_path / "a.csv"
    part_b = tmp_path / "b.csv"
    header = "f1,f2,label"
    _write_csv(part_a, ["1,10,benign", "2,20,attack", "1,10,benign"], header)  # 1 dup within a
    _write_csv(part_b, ["3,30,attack", "1,10,benign"], header)  # dup of a row in a

    report = characterize_dataset(tmp_path, chunk_size=2)

    assert report.n_records == 5
    assert report.columns["f1"] == "int64"
    assert report.columns["f2"] == "int64"
    # pandas' string dtype label varies by version ("object" vs. pyarrow-backed "str"); only the
    # numeric columns' exact dtype matters here.
    assert set(report.columns) == {"f1", "f2", "label"}
    assert report.label_counts == {"benign": 3, "attack": 2}
    assert report.imbalance_ratio == pytest.approx(3 / 2)
    # Three rows are identical ("1,10,benign"): 2 duplicates beyond the first occurrence.
    assert report.exact_duplicate_count == 2


def test_characterize_zero_variance_and_nulls(tmp_path: Path) -> None:
    header = "constant,varying,sparse,label"
    _write_csv(
        tmp_path / "only.csv",
        ["7,1,,x", "7,2,5,y", "7,3,,x"],
        header,
    )

    report = characterize_dataset(tmp_path, chunk_size=100)

    # "constant" never varies; "sparse" has only one non-null value (5), so it is trivially
    # zero-variance among its observed values too -- nulls are reported separately below.
    assert report.zero_variance_columns == ["constant", "sparse"]
    assert report.null_counts["sparse"] == 2
    assert report.null_counts["constant"] == 0


def test_characterize_infinite_values_counted_and_excluded_from_correlation(tmp_path: Path) -> None:
    header = "rate,other,label"
    # One +inf row (a near-zero-duration flow dividing out to infinity, as happens with the
    # real CICIoT2023 "Rate" feature) must be counted, not silently folded into the correlation.
    _write_csv(
        tmp_path / "inf.csv",
        ["1,10,a", "2,20,a", "inf,30,a"],
        header,
    )

    report = characterize_dataset(tmp_path, chunk_size=100)

    assert report.infinite_counts["rate"] == 1
    assert report.infinite_counts.get("other", 0) == 0
    # Only the two finite rows feed the correlation accumulation -> perfect correlation, not NaN.
    assert report.correlation_matrix["rate"]["other"] == pytest.approx(1.0)


def test_characterize_correlation_matrix_perfect_and_inverse(tmp_path: Path) -> None:
    header = "x,y,z,label"
    # y = 2x (perfect positive correlation), z = -x (perfect negative correlation).
    rows = [f"{i},{2 * i},{-i},lbl" for i in range(1, 6)]
    _write_csv(tmp_path / "corr.csv", rows, header)

    report = characterize_dataset(tmp_path, chunk_size=2)

    assert report.correlation_matrix["x"]["x"] == pytest.approx(1.0)
    assert report.correlation_matrix["x"]["y"] == pytest.approx(1.0)
    assert report.correlation_matrix["x"]["z"] == pytest.approx(-1.0)
    assert report.correlation_matrix["y"]["z"] == pytest.approx(-1.0)


def test_characterize_raises_on_mismatched_columns(tmp_path: Path) -> None:
    _write_csv(tmp_path / "a.csv", ["1,x"], "f1,label")
    _write_csv(tmp_path / "b.csv", ["1,2,x"], "f1,f2,label")

    with pytest.raises(ValueError, match="column layout"):
        characterize_dataset(tmp_path, chunk_size=100)


def test_characterize_raises_when_no_csv_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        characterize_dataset(tmp_path, chunk_size=100)
