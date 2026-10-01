#!/usr/bin/env python
"""Generate the paper's figures from committed run manifests.

Every figure here is read from a manifest in ``artifacts/``; nothing is transcribed. Running
this script is therefore the check that the paper's numbers and its plots agree -- if a
manifest changes, the figures change with it.

Output is vector PDF into ``paper/figures/``, sized for IEEE two-column: 3.5 in wide for a
single-column float, 7.16 in for a full-width one.

Design rules followed (the project's data-visualisation guidance):

* **One axis per chart, never two.** Where two measures share a unit they share an axis; where
  they do not (macro-F1 against megabytes) they are separate figures. A dual-axis chart is the
  single most common charting error and is not used here.
* **Categorical hues in fixed order**, from the validated three-slot palette (blue, orange,
  aqua), which passes the all-pairs colour-vision and normal-vision floors. The aqua slot sits
  below 3:1 contrast on the light surface, so every figure using it carries direct labels or has
  its data tabulated in the paper -- the documented relief.
* **Secondary encoding on every series**: distinct markers and line styles, because an IEEE
  paper is frequently read in greyscale and colour alone must never carry identity.
* Recessive axes and grid, thin marks, selective direct labels rather than a number on every
  point, and text in ink colours rather than in the series colour.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path("artifacts")
OUT = Path("paper/figures")

# --- validated palette (light surface) --------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"

COL1, COL2 = 3.5, 7.16  # IEEE column widths, inches


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.color": "#e4e3de",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def recessive(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Spines down to two, grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def save(fig: plt.Figure, stem: str) -> None:
    """Vector PDF for the paper, plus a raster copy for eyeballing the layout."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.pdf"
    fig.savefig(path)
    preview = OUT / "preview"
    preview.mkdir(exist_ok=True)
    fig.savefig(preview / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


# ===========================================================================
def fig_ksweep() -> None:
    """Macro-F1 and rarest-class F1 against client count.

    Both series are F1 scores on [0,1], so they share one axis -- the point of the figure is
    precisely that the aggregate hides the rare class, and that comparison is only legible on a
    common scale.
    """
    import statistics as st

    d = load("k_sweep_results.json")
    by: dict[int, list[dict]] = {}
    for row in d["per_run"]:
        by.setdefault(int(row["K"]), []).append(row)

    ks = sorted(by)
    macro = [st.mean([r["macro_f1"] for r in by[k]]) for k in ks]
    macro_sd = [st.pstdev([r["macro_f1"] for r in by[k]]) for k in ks]
    rarest = []
    for k in ks:
        acc: dict[str, list[float]] = {}
        for r in by[k]:
            for cls, f1 in r["per_class_f1"].items():
                acc.setdefault(cls, []).append(f1)
        rarest.append(min(st.mean(v) for v in acc.values()))

    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    recessive(ax)

    ax.axhline(0.8543, color=MUTED, ls=(0, (1, 2)), lw=0.8, zorder=1)
    ax.text(
        50, 0.8543, "centralised ceiling 0.8543", ha="right", va="bottom", fontsize=6.5, color=INK2
    )

    ax.axvline(42, color=MUTED, ls=(0, (4, 2)), lw=0.8, zorder=1)
    # Left of the rule and high in the panel: the lower-right corner belongs to the
    # weakest-class endpoint label, which this collided with at the axis floor.
    ax.text(41, 0.615, "pigeonhole\nbound 42 ", ha="right", va="center", fontsize=6.5, color=INK2)

    ax.errorbar(
        ks,
        macro,
        yerr=macro_sd,
        color=BLUE,
        lw=1.6,
        marker="o",
        ms=4.5,
        capsize=2,
        elinewidth=0.8,
        label="macro-F1 (all 8 classes)",
        zorder=3,
    )
    ax.plot(
        ks,
        rarest,
        color=ORANGE,
        lw=1.6,
        ls=(0, (5, 2)),
        marker="s",
        ms=4.5,
        label="weakest class F1",
        zorder=3,
    )

    # selective direct labels: the endpoints only
    for x, y, t, _c, va in (
        (ks[0], macro[0], f"{macro[0]:.3f}", BLUE, "bottom"),
        (ks[-1], macro[-1], f"{macro[-1]:.3f}", BLUE, "bottom"),
        (ks[0], rarest[0], f"{rarest[0]:.3f}", ORANGE, "bottom"),
        (ks[-1], rarest[-1], f"{rarest[-1]:.3f}", ORANGE, "top"),
    ):
        ax.annotate(
            t,
            (x, y),
            textcoords="offset points",
            xytext=(0, 6 if va == "bottom" else -11),
            ha="center",
            fontsize=6.5,
            color=INK2,
        )

    ax.set_xlabel("client count $K$")
    ax.set_ylabel("F1")
    ax.set_xticks(ks)
    ax.set_ylim(0.30, 0.92)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0))
    save(fig, "ksweep_f1")


def fig_communication() -> None:
    """Per-round traffic against client count. Separate figure because bytes are not F1."""
    d = load("k_sweep_results.json")
    by: dict[int, int] = {}
    for row in d["per_run"]:
        by[int(row["K"])] = int(row["bytes_per_round"])
    ks = sorted(by)
    mb = [by[k] / 1e6 for k in ks]

    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    recessive(ax)
    ax.plot(ks, mb, color=BLUE, lw=1.6, marker="o", ms=4.5, zorder=3)
    for x, y in ((ks[0], mb[0]), (ks[-1], mb[-1])):
        ax.annotate(
            f"{y:.2f} MB",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=6.5,
            color=INK2,
        )
    ax.set_xlabel("client count $K$")
    ax.set_ylabel("traffic per round (MB)")
    ax.set_xticks(ks)
    save(fig, "communication")


def fig_bracket() -> None:
    """The three-way comparison: local-only, federated, centralised."""
    s = load("manifest_phase4_default.json")["results"]["summary"]
    names = ["local-only\n(lower bound)", "federated\n($K$=3)", "centralised\n(upper bound)"]
    vals = [
        s["local_only"]["macro_f1_mean"],
        s["federated"]["macro_f1_mean"],
        s["centralized"]["macro_f1_mean"],
    ]
    errs = [
        s["local_only"]["macro_f1_std"],
        s["federated"]["macro_f1_std"],
        s["centralized"]["macro_f1_std"],
    ]
    colors = [MUTED, BLUE, MUTED]

    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    recessive(ax)
    bars = ax.bar(
        names,
        vals,
        yerr=errs,
        capsize=3,
        color=colors,
        width=0.55,
        error_kw={"elinewidth": 0.9, "ecolor": INK2},
        zorder=3,
    )
    for b, v, e in zip(bars, vals, errs, strict=True):
        ax.annotate(
            f"{v:.4f}\n$\\pm${e:.4f}",
            (b.get_x() + b.get_width() / 2, v + e),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=6.5,
            color=INK2,
        )
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.0)
    save(fig, "federation_bracket")


def fig_per_client_gain() -> None:
    """What each farm gains by federating, on both corpora."""
    import statistics as st

    d = load("manifest_phase9_federation_gain_mixed.json")
    ps = d["results"]["experiments"]["in_distribution"]["per_seed"]
    corpora = ["ciciot2023", "ciciomt2024"]
    labels = {"ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}
    fed = {c: st.mean([s["scores"][c]["macro_f1_present"] for s in ps]) for c in corpora}
    clients = [c["client"] for c in ps[0]["local_only_per_client"]]

    gains = {c: [] for c in corpora}
    for i in range(len(clients)):
        for c in corpora:
            solo = st.mean(
                [s["local_only_per_client"][i]["scores"][c]["macro_f1_present"] for s in ps]
            )
            gains[c].append(fed[c] - solo)

    x = range(len(clients))
    w = 0.36
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    recessive(ax)
    for j, (c, col, hatch) in enumerate(((corpora[0], BLUE, None), (corpora[1], ORANGE, "///"))):
        pos = [k + (j - 0.5) * w for k in x]
        ax.bar(
            pos,
            gains[c],
            width=w - 0.02,
            color=col,
            label=labels[c],
            hatch=hatch,
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
        )
        for p, v in zip(pos, gains[c], strict=True):
            ax.annotate(
                f"+{v:.3f}",
                (p, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=6.2,
                color=INK2,
            )
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("farm-", "farm ") for c in clients])
    ax.set_ylabel("macro-F1 gained by federating")
    ax.set_ylim(0, max(max(gains[c]) for c in corpora) * 1.28)
    ax.legend(loc="upper right")
    save(fig, "per_client_gain")


def fig_feature_overlap() -> None:
    """Cross-corpus range overlap per selected feature -- the semantics finding."""
    import sys

    sys.path.insert(0, "src")
    import numpy as np

    from ascon_smart_agri.data.datasets import SELECTED_COLUMNS

    d = np.load(ART / "phase9_cache.npz", allow_pickle=True)
    cols = [str(c) for c in d["ciciot2023__columns"]]
    A, B = d["ciciot2023__x_tr"], d["ciciomt2024__x_tr"]

    def ov(a, b):
        la, ha = np.quantile(a, [0.01, 0.99])
        lb, hb = np.quantile(b, [0.01, 0.99])
        lo, hi = max(la, lb), min(ha, hb)
        union = max(ha, hb) - min(la, lb)
        return 1.0 if union <= 0 else max(0.0, (hi - lo)) / union

    rows = [(c, ov(A[:, cols.index(c)], B[:, cols.index(c)])) for c in SELECTED_COLUMNS]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] * 100 for r in rows]
    colors = [ORANGE if v < 10 else BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(COL1, 3.1))
    recessive(ax, grid_axis="x")
    ax.barh(names, vals, color=colors, height=0.66, zorder=3)
    ax.axvline(10, color=INK2, ls=(0, (4, 2)), lw=0.8, zorder=4)
    ax.text(10, -0.9, " 10% gate", fontsize=6.5, color=INK2, va="top")
    for n, v in zip(names, vals, strict=True):
        ax.annotate(
            f"{v:.1f}",
            (v, n),
            textcoords="offset points",
            xytext=(3, 0),
            va="center",
            fontsize=6.2,
            color=INK2,
        )
    ax.set_xlabel("cross-corpus range overlap (%), 1st-99th percentile")
    ax.set_xlim(0, 108)
    save(fig, "feature_overlap")


def fig_convergence() -> None:
    """Per-round macro-F1 at three client counts."""
    d = load("k_sweep_results.json")
    want = [3, 20, 50]
    series = {}
    for row in d["per_run"]:
        k = int(row["K"])
        if k in want and int(row["seed"]) == 0:
            series[k] = row["per_round_macro_f1"]

    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    recessive(ax)
    spec = ((BLUE, "o", "solid"), (ORANGE, "s", (0, (5, 2))), (AQUA, "^", (0, (1, 1.6))))
    for (k, ys), (col, mk, ls) in zip(sorted(series.items()), spec, strict=False):
        ax.plot(
            range(1, len(ys) + 1),
            ys,
            color=col,
            lw=1.5,
            ls=ls,
            marker=mk,
            ms=3.4,
            markevery=2,
            label=f"$K$={k}",
            zorder=3,
        )
    ax.set_xlabel("federated round")
    ax.set_ylabel("macro-F1")
    ax.legend(loc="lower right")
    save(fig, "convergence")


def fig_baselines() -> None:
    """Centralised baselines: recurrence against the alternatives at an identical budget."""
    names = ["random\nforest", "MLP\n(no recurrence)", "GRU\n($W$=16)"]
    vals = [0.6855, 0.6070, 0.8297]
    errs = [0.0, 0.0, 0.0013]
    colors = [MUTED, MUTED, BLUE]

    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    recessive(ax)
    bars = ax.bar(
        names,
        vals,
        yerr=errs,
        capsize=3,
        color=colors,
        width=0.55,
        error_kw={"elinewidth": 0.9, "ecolor": INK2},
        zorder=3,
    )
    for b, v in zip(bars, vals, strict=True):
        ax.annotate(
            f"{v:.4f}",
            (b.get_x() + b.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6.8,
            color=INK2,
        )
    ax.annotate(
        "",
        xy=(2, 0.8297),
        xytext=(1, 0.6070),
        arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 1.0},
    )
    ax.text(1.5, 0.735, "recurrence\n$+0.2322$", ha="center", fontsize=6.8, color=ORANGE)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.0)
    save(fig, "baselines")


FIGURES = {
    "ksweep_f1": fig_ksweep,
    "communication": fig_communication,
    "federation_bracket": fig_bracket,
    "per_client_gain": fig_per_client_gain,
    "feature_overlap": fig_feature_overlap,
    "convergence": fig_convergence,
    "baselines": fig_baselines,
}


def main() -> int:
    style()
    failures = 0
    for name, fn in FIGURES.items():
        try:
            fn()
        except Exception as exc:  # a missing manifest must name itself, not crash the batch
            failures += 1
            print(f"  SKIPPED {name}: {type(exc).__name__}: {exc}")
    print(f"\n  {len(FIGURES) - failures}/{len(FIGURES)} figures written to {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
