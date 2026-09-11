"""Federated GRU intrusion detection with Ascon-authenticated telemetry (CICIoT2023).

Package layout mirrors the paper's separation of concerns:

    data/         characterisation, subsampling, dedup/leakage control, block splitting (III-B)
    features/     four-stage feature selection (III-C)
    sequences/    windowing over contiguous same-label runs (III-D)
    model/        GRU detector + centralised training loop (III-E)
    federated/    clients, aggregator, partitioning, scaler stats, serialization (III-F)
    telemetry/    MQTT/JSON payload simulation + feature-provenance adapter (III-H)
    crypto/       Ascon-AEAD128 authenticated encryption (III-G)
    routing/      verdict-driven, disjoint benign/malicious data paths (III-A)
    eval/         metrics, baselines, reporting, run manifest (III-I)

Development follows the seven gated phases of Section III-J4; see README.
"""

__version__ = "0.0.0"
