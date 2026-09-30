#!/usr/bin/env python
"""Agriculture-native centralised check --- the gate on a domain-specific rebuild.

The user fixed **smart agriculture as the project's deployment domain** (2026-09-29). That
decides which corpora are admissible, because the standing domain-coherence principle says a
detector for agriculture federates over farms: CICIoMT2024 is a *medical* testbed, and
Edge-IIoTset is the only registered corpus captured on agricultural equipment (soil-moisture,
temperature, pH and water-level sensors over MQTT, plus Modbus TCP).

Edge-IIoTset is registered ``not_for_training`` because the 2026-09-21 run scored macro-F1 0.12
on it. The 2026-09-23 characterisation re-read established that this figure measured **our
packet-field reconstruction of CICIoT2023's window features**, not the corpus: the raw file has
815 exact duplicates in 2,219,201 records (0.04 %), not the 81 % the reconstruction produced.
This script is the half-day centralised check that entry scoped. It trains the GRU on
Edge-IIoTset's **own** columns and reports what the corpus can actually support. It does not
change the registry, and it does not federate: it answers one question, by measurement, before
any pivot is built on top of it.

What it does, following the project's existing discipline unchanged:

* **Native numeric features (26), not reconstructed windows.** The 63 native columns minus the
  spec's ``drop_columns``, minus 4 zero-variance columns, minus 5 text columns, minus 9
  identifier columns (below), leaves 26: 8 MQTT fields, 2 Modbus TCP fields, and TCP/DNS/ARP/
  ICMP/HTTP flags, lengths and timings. ``udp.time_delta`` is the native analogue of ``IAT``
  and ``tcp.len`` of ``Tot size``, so the behavioural shape matches the canonical set.
* **The five text columns stay out, and two of them are the reason.** All five carry the
  documented ``"0"``/``"0.0"`` provenance artefact (arXiv 2608.15761), which alone would justify
  excluding them. Worse, a census of the whole file shows ``http.request.version`` holds injected
  attack payloads (``-al&_PHPLIB[libdir]=http://cirt.net/rfiinc.txt?? HTTP/1.1``) and
  ``dns.qry.name.len`` holds DNS names rather than lengths --- a column-shift in the corpus build.
  Encoding either would let the model read the attack class off the payload text. A numeric-only
  feature set excludes the artefact structurally rather than by cleaning.
* **Nine identifier columns dropped** (``tcp.seq``, ``tcp.ack``, ``tcp.ack_raw``,
  ``tcp.checksum``, ``icmp.checksum``, ``icmp.seq_le``, ``icmp.transmit_timestamp``,
  ``udp.stream``, ``mbtcp.trans_id``): raw sequence numbers, checksums, a Wireshark-assigned
  per-capture stream index and a Modbus transaction id. Each is a transport nonce or a
  capture-local counter, and the spec's own rule is "identifiers, addresses, timestamps and free
  text: provenance, never behaviour". Their *deltas* are behavioural and their absolute values
  are not; recovering the deltas is future work, not this check.
* **Per-label contiguous prefix cap**, not a random trim. The file is sorted by label into
  exactly 15 runs, so taking each label's first ``per_class_cap`` rows preserves the adjacency
  windowing depends on. A seeded random trim at this cap would delete ~22 of every 23 ``Normal``
  rows and leave no contiguous run of length W at all --- zero windows, not a smaller sample.
  This mirrors ``data/subsample.py``'s "read whole parts in fixed order until the cap" mechanic.
* **Dedup BEFORE split (R3)**, block split, and ``contiguity_segments`` so no window spans a
  block removed by the split. The realised train/test hash overlap is asserted to be 0 and
  recorded in the manifest.
* **Two taxonomies, both reported.** ``family`` maps the 15 leaves onto the project's 8 canonical
  families, which is what makes the number comparable to CICIoT2023's 0.83; ``native`` keeps all
  15 classes, which is the agriculture-relevant target, because the family map collapses
  Ransomware, Backdoor, SQL_injection, Uploading and XSS into one "WebBased" class and so
  discards exactly the attack coverage this corpus was wanted for.
* **>= 3 seeds, mean +/- std, never accuracy alone**, with FPR from the benign-vs-attack
  projection reported first-class.

Run::

    ./.venv/bin/python scripts/run_agri_native.py --epochs 15 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ascon_smart_agri.data.dedup import deduplicate, record_hash
from ascon_smart_agri.data.scaling import apply_scaler, fit_scaler
from ascon_smart_agri.data.split import make_blocks, stratified_block_split
from ascon_smart_agri.data.taxonomy import (
    CLASS_NAMES,
    EDGE_IIOTSET_LEAF_TO_FAMILY,
)
from ascon_smart_agri.eval.metrics import binary_metrics, multiclass_metrics
from ascon_smart_agri.model.train import predict, train_centralized
from ascon_smart_agri.sequences.windowing import (
    build_windows,
    contiguity_segments,
)

#: The 26 behavioural columns of Edge-IIoTset's native schema. Order is fixed so the manifest,
#: the scaler and any later checkpoint agree on what column i means.
AGRI_NATIVE_FEATURES: tuple[str, ...] = (
    # --- MQTT: the protocol farm automation runs on (8) ---
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.topic_len",
    "mqtt.ver",
    # --- Modbus TCP: the other industrial-control protocol in the capture (2) ---
    "mbtcp.len",
    "mbtcp.unit_id",
    # --- TCP behaviour: flags, connection state, payload size (7) ---
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    # --- timing: the native analogue of the canonical set's IAT (1) ---
    "udp.time_delta",
    # --- DNS (4) ---
    "dns.qry.qu",
    "dns.retransmission",
    "dns.retransmit_request",
    "dns.retransmit_request_in",
    # --- HTTP (2) ---
    "http.content_length",
    "http.response",
    # --- ARP (2) ---
    "arp.opcode",
    "arp.hw.size",
)

#: Kept out of :data:`AGRI_NATIVE_FEATURES` on purpose, with the reason, so the manifest records
#: a decision rather than an omission.
EXCLUDED_COLUMNS: dict[str, str] = {
    "tcp.seq": "raw TCP sequence number: transport nonce, capture-local",
    "tcp.ack": "raw TCP ack number: transport nonce, capture-local",
    "tcp.ack_raw": "raw TCP ack number: transport nonce, capture-local",
    "tcp.checksum": "checksum: nonce, no behavioural content",
    "icmp.checksum": "checksum: nonce, no behavioural content",
    "icmp.seq_le": "ICMP sequence number: transport nonce",
    "icmp.transmit_timestamp": "timestamp: provenance, never behaviour",
    "udp.stream": "Wireshark-assigned per-capture stream index: pure provenance",
    "mbtcp.trans_id": "Modbus transaction id: capture-local counter",
    "icmp.unused": "zero variance over the whole corpus",
    "http.tls_port": "zero variance over the whole corpus",
    "dns.qry.type": "zero variance over the whole corpus",
    "mqtt.msg_decoded_as": "zero variance over the whole corpus",
    "http.request.method": "text; carries the '0'/'0.0' provenance artefact",
    "http.request.version": "text; holds injected attack payloads (reads the class off content)",
    "dns.qry.name.len": "text; holds DNS names, not lengths (corpus column shift)",
    "mqtt.conack.flags": "text; carries the '0'/'0.0' provenance artefact",
    "mqtt.protoname": "text; carries the '0'/'0.0' provenance artefact",
}

CSV_NAME = "DNN-EdgeIIoT-dataset.csv"
BENIGN_LEAF = "Normal"


def load_capped(root: Path, per_class_cap: int, chunk_size: int) -> pd.DataFrame:
    """Read the corpus, keeping each label's first ``per_class_cap`` rows in file order."""
    path = root / CSV_NAME
    usecols = [*AGRI_NATIVE_FEATURES, "Attack_type"]
    kept: list[pd.DataFrame] = []
    taken: dict[str, int] = {}

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk = chunk.copy()
        chunk["Attack_type"] = chunk["Attack_type"].astype(str).str.strip()
        pieces: list[pd.DataFrame] = []
        for leaf, group in chunk.groupby("Attack_type", sort=False):
            room = per_class_cap - taken.get(leaf, 0)
            if room <= 0:
                continue
            piece = group.iloc[:room]  # prefix, order preserved: adjacency survives
            taken[leaf] = taken.get(leaf, 0) + len(piece)
            pieces.append(piece)
        if pieces:
            kept.append(pd.concat(pieces).sort_index())
        if taken and all(v >= per_class_cap for v in taken.values()) and len(taken) >= 15:
            break

    frame = pd.concat(kept).sort_index().reset_index(drop=True)
    for column in AGRI_NATIVE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["label"] = frame["Attack_type"]
    frame["source_file"] = CSV_NAME
    return frame.drop(columns=["Attack_type"])


def encode(labels: pd.Series, taxonomy: str) -> tuple[np.ndarray, list[str], int]:
    """Map leaf labels to class indices under ``family`` (8) or ``native`` (15)."""
    if taxonomy == "family":
        names = list(CLASS_NAMES)
        mapped = labels.map(EDGE_IIOTSET_LEAF_TO_FAMILY)
        benign = names.index("Benign")
    else:
        names = sorted(labels.unique())
        mapped = labels
        benign = names.index(BENIGN_LEAF)
    index = {name: i for i, name in enumerate(names)}
    return mapped.map(index).to_numpy(dtype=np.int64), names, benign


def windows_for(
    frame: pd.DataFrame, codes: np.ndarray, mask: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Windows over one split's rows, never spanning a block the split removed."""
    part = frame.loc[mask]
    segments = contiguity_segments(
        part["source_file"].to_numpy(), part.index.to_numpy().astype(np.int64)
    )
    features = part[list(AGRI_NATIVE_FEATURES)].to_numpy(dtype=np.float64)
    return build_windows(features, codes[mask], segments, window)


def window_hashes(sequences: np.ndarray) -> np.ndarray:
    """A stable 64-bit content hash per ``(W, F)`` window, for the R3 gate.

    FNV-1a over each window's raw bytes, vectorised one column at a time. Hashing the
    equivalent ``(N, W*F)`` DataFrame through ``pandas.util.hash_pandas_object`` is correct but
    takes minutes at W=16, F=26; this takes seconds and the gate is run three times per seed.
    """
    flat = np.ascontiguousarray(sequences.reshape(len(sequences), -1), dtype=np.float64)
    digest = np.full(len(flat), np.uint64(14695981039346656037), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for column in range(flat.shape[1]):
        digest = (digest ^ flat[:, column].view(np.uint64)) * prime
    return digest


def dedup_windows(sequences: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop duplicate windows, keeping the first occurrence and the original order."""
    if len(sequences) == 0:
        return sequences, labels, 0
    _, first = np.unique(window_hashes(sequences), return_index=True)
    keep = np.zeros(len(sequences), dtype=bool)
    keep[first] = True  # np.unique returns the first occurrence of each distinct hash
    return sequences[keep], labels[keep], int((~keep).sum())


def drop_train_overlap(
    sequences: np.ndarray, labels: np.ndarray, train_hashes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove test windows whose content also appears in train --- the R3 gate itself."""
    if len(sequences) == 0:
        return sequences, labels, 0
    keep = ~np.isin(window_hashes(sequences), train_hashes)
    return sequences[keep], labels[keep], int((~keep).sum())


def distinct_windows_per_label(frame: pd.DataFrame, window: int) -> dict[str, int]:
    """How many DISTINCT windows each label can actually contribute.

    Edge-IIoTset is a sparse union of protocol-specific Wireshark fields: a packet populates
    only its own protocol's columns. Application-layer traffic is therefore rich here, while a
    UDP or ICMP flood packet is all-zeros across every behavioural feature --- its only
    distinguishing content lived in ``udp.stream``/``icmp.checksum``/``icmp.seq_le``, which are
    nonces and are dropped. A class whose distinct-window count is tiny cannot be trained or
    evaluated: its test windows are byte-identical to its training windows, which is leakage by
    construction rather than generalisation. This measures that per class so the exclusion is a
    recorded measurement, not a hand-picked class list.
    """
    codes, names = pd.factorize(frame["label"])
    features = frame[list(AGRI_NATIVE_FEATURES)].to_numpy(dtype=np.float64)
    sequences, labels = build_windows(
        features, codes.astype(np.int64), np.zeros(len(frame), dtype=np.int64), window
    )
    digests = window_hashes(sequences)
    return {str(name): len(np.unique(digests[labels == index])) for index, name in enumerate(names)}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - manifest provenance only
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/edge_iiotset"))
    parser.add_argument("--per-class-cap", type=int, default=70_000)
    parser.add_argument("--chunk-size", type=int, default=400_000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--sequence-cap", type=int, default=400_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--taxonomy",
        choices=["family", "native", "both"],
        default="both",
        help="family = the project's 8 classes (comparable); native = all 15 (the agri target)",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--min-distinct-windows",
        type=int,
        default=500,
        help=(
            "exclude any class contributing fewer than this many DISTINCT windows. The corpus "
            "cannot express network-layer floods on leakage-safe features (DDoS_UDP "
            "contributes exactly 1), and a class at that count is memorised, not learned. "
            "0 keeps every class."
        ),
    )
    parser.add_argument(
        "--dedup-level",
        choices=["window", "packet", "none"],
        default="window",
        help=(
            "granularity of the R3 gate. 'window' (default) is the correct unit for a "
            "PACKET-granularity corpus: the record the model sees is a W-packet window, "
            "and deduplicating individual packets would delete the repetition that IS a "
            "flood. 'packet' reproduces the window-corpus rule row-wise, as an ablation."
        ),
    )
    parser.add_argument("--run-name", default="agri_native")
    args = parser.parse_args()

    started = time.time()
    print(f"[load] {args.root / CSV_NAME} | cap {args.per_class_cap:,}/class")
    frame = load_capped(args.root, args.per_class_cap, args.chunk_size)
    print(f"[load] {len(frame):,} rows | {len(AGRI_NATIVE_FEATURES)} native features")

    distinct = distinct_windows_per_label(frame, args.window)
    print("[viability] distinct windows each class can contribute:")
    for name, count in sorted(distinct.items(), key=lambda kv: -kv[1]):
        mark = "" if count >= args.min_distinct_windows else "   <-- EXCLUDED, degenerate"
        print(f"   {name:<24} {count:>8,}{mark}")
    excluded = sorted(n for n, c in distinct.items() if c < args.min_distinct_windows)
    if excluded:
        frame = frame[~frame["label"].isin(excluded)].reset_index(drop=True)
        print(f"[viability] excluded {len(excluded)}: {excluded} -> {len(frame):,} rows remain")

    packet_hashes = record_hash(frame[[*AGRI_NATIVE_FEATURES, "label"]])
    n_distinct_packets = int(packet_hashes.nunique())
    print(
        f"[packets] {len(frame):,} rows | {n_distinct_packets:,} distinct "
        f"({100 * (1 - n_distinct_packets / len(frame)):.1f}% duplicate on the 26 features)"
    )
    if args.dedup_level == "packet":
        deduped = deduplicate(frame[[*AGRI_NATIVE_FEATURES, "label"]])
        deduped["source_file"] = CSV_NAME
        print(f"[dedup packet] {len(frame):,} -> {len(deduped):,} rows")
    else:
        deduped = frame.reset_index(drop=True)
    n_dupes = len(frame) - len(deduped)
    counts = deduped["label"].value_counts()
    print("[labels] " + " | ".join(f"{k} {v:,}" for k, v in counts.items()))

    taxonomies = ["family", "native"] if args.taxonomy == "both" else [args.taxonomy]
    results: dict[str, dict] = {}

    for taxonomy in taxonomies:
        codes, names, benign_index = encode(deduped["label"], taxonomy)
        print(f"\n{'=' * 92}\n[{taxonomy}] {len(names)} classes: {names}\n{'=' * 92}")
        per_seed: list[dict] = []

        for seed in args.seeds:
            blocks = make_blocks(deduped, args.block_size)
            split = stratified_block_split(
                deduped, blocks, test_fraction=args.test_fraction, seed=seed
            )
            train_mask = blocks.isin(split.train_blocks).to_numpy()
            test_mask = blocks.isin(split.test_blocks).to_numpy()

            scaler = fit_scaler(deduped.loc[train_mask, list(AGRI_NATIVE_FEATURES)].to_numpy())
            scaled = deduped.copy()
            scaled[list(AGRI_NATIVE_FEATURES)] = apply_scaler(
                deduped[list(AGRI_NATIVE_FEATURES)].to_numpy(), scaler
            )

            x_train, y_train = windows_for(scaled, codes, train_mask, args.window)
            x_test, y_test = windows_for(scaled, codes, test_mask, args.window)
            n_raw_train, n_raw_test = len(x_train), len(x_test)

            # R3 at the granularity of the record the model actually consumes.
            if args.dedup_level != "none":
                x_train, y_train, dup_train = dedup_windows(x_train, y_train)
                x_test, y_test, dup_test = dedup_windows(x_test, y_test)
                x_test, y_test, overlap = drop_train_overlap(x_test, y_test, window_hashes(x_train))
            else:
                dup_train = dup_test = overlap = 0
            if len(x_train) == 0 or len(x_test) == 0:
                raise SystemExit(
                    f"[{taxonomy} | seed {seed}] empty split after the R3 gate: "
                    f"train {len(x_train)}, test {len(x_test)} (from {n_raw_train}/{n_raw_test} "
                    "raw windows). The corpus cannot support this configuration."
                )
            print(
                f"  [{taxonomy} | seed {seed}] windows train {n_raw_train:,}->{len(x_train):,} "
                f"(-{dup_train:,} dup) | test {n_raw_test:,}->{len(x_test):,} "
                f"(-{dup_test:,} dup, -{overlap:,} seen in train)"
            )
            if len(x_train) > args.sequence_cap:
                keep = np.random.default_rng(seed).choice(
                    len(x_train), args.sequence_cap, replace=False
                )
                keep.sort()  # complete sequences only; order of whole windows is free
                x_train, y_train = x_train[keep], y_train[keep]

            t0 = time.time()
            model = train_centralized(
                x_train,
                y_train,
                hidden_size=args.hidden_size,
                n_classes=len(names),
                epochs=args.epochs,
                seed=seed,
            )
            y_pred, probability = predict(model, x_test)
            multi = multiclass_metrics(y_test, y_pred, names)
            binary = binary_metrics(
                (y_test != benign_index).astype(np.int64), 1.0 - probability[:, benign_index]
            )
            print(
                f"  [{taxonomy} | seed {seed}] macro-F1 {multi.macro_f1:.4f} | "
                f"bal-acc {multi.balanced_accuracy:.4f} | MCC {multi.mcc:.4f} | "
                f"FPR {binary.false_positive_rate:.4f} | {time.time() - t0:.0f}s"
            )
            per_seed.append(
                {
                    "seed": seed,
                    "n_train_sequences": len(x_train),
                    "n_test_sequences": len(x_test),
                    "r3_overlap_removed": overlap,
                    "n_duplicate_windows_train": dup_train,
                    "n_duplicate_windows_test": dup_test,
                    "macro_f1": multi.macro_f1,
                    "balanced_accuracy": multi.balanced_accuracy,
                    "mcc": multi.mcc,
                    "accuracy": multi.accuracy,
                    "per_class_f1": multi.per_class_f1,
                    "confusion": multi.confusion.tolist(),
                    "false_positive_rate": binary.false_positive_rate,
                    "pr_auc": binary.pr_auc,
                }
            )

        def agg(key: str, rows: list[dict] = per_seed) -> dict[str, float]:
            values = [float(s[key]) for s in rows]
            return {"mean": float(np.mean(values)), "std": float(np.std(values))}

        results[taxonomy] = {
            "class_names": names,
            "per_seed": per_seed,
            "summary": {
                k: agg(k)
                for k in (
                    "macro_f1",
                    "balanced_accuracy",
                    "mcc",
                    "accuracy",
                    "false_positive_rate",
                    "pr_auc",
                )
            },
            "per_class_f1_mean": {
                name: float(np.mean([s["per_class_f1"][name] for s in per_seed])) for name in names
            },
        }

    elapsed = time.time() - started
    manifest = {
        "run_name": args.run_name,
        "question": (
            "Can the GRU learn from Edge-IIoTset's NATIVE columns, now that smart agriculture "
            "is the deployment domain? The 0.12 of 2026-09-21 measured our reconstruction of "
            "CICIoT2023's window features, not this corpus."
        ),
        "corpus": "edge_iiotset",
        "features": list(AGRI_NATIVE_FEATURES),
        "excluded_columns": EXCLUDED_COLUMNS,
        "config": {
            "per_class_cap": args.per_class_cap,
            "block_size": args.block_size,
            "test_fraction": args.test_fraction,
            "window": args.window,
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "sequence_cap": args.sequence_cap,
            "seeds": args.seeds,
        },
        "corpus_stats": {
            "n_rows_capped": len(frame),
            "n_rows_after_packet_dedup": len(deduped),
            "n_distinct_packets": n_distinct_packets,
            "dedup_level": args.dedup_level,
            "distinct_windows_per_label": distinct,
            "excluded_degenerate_classes": excluded,
            "min_distinct_windows": args.min_distinct_windows,
            "n_exact_duplicates": int(n_dupes),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "hardware": {"platform": platform.platform(), "git_commit": git_commit()},
        "results": results,
        "elapsed_seconds": elapsed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"manifest_{args.run_name}.json"
    out.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 92}\nAGRI-NATIVE CHECK -- {len(args.seeds)} seeds, mean +/- std\n{'=' * 92}")
    print(f"{'taxonomy':<12}{'classes':>9}{'macro-F1':>20}{'bal-acc':>18}{'FPR':>18}")
    for taxonomy, body in results.items():
        s = body["summary"]
        print(
            f"{taxonomy:<12}{len(body['class_names']):>9}"
            f"{s['macro_f1']['mean']:>13.4f} ± {s['macro_f1']['std']:.4f}"
            f"{s['balanced_accuracy']['mean']:>11.4f} ± {s['balanced_accuracy']['std']:.4f}"
            f"{s['false_positive_rate']['mean']:>11.4f} ± {s['false_positive_rate']['std']:.4f}"
        )
    print(f"\n  elapsed {elapsed / 60:.1f} min | manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
