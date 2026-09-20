"""Typed run configuration for the whole pipeline.

Uses pydantic models (not raw dicts) so every experimental knob in the paper --- window
length W, Dirichlet alpha, local epochs E, feature count F, rounds R, seeds --- is declared
in one validated place and serialized verbatim into the per-run manifest (Section III-I4).

The sweep lists encode the six ablations of Section III-I3:
    * window length      W in {1, 8, 16, 32}
    * heterogeneity      alpha in {0.1, 0.5, 100}
    * local epochs       E in {1, 3, 5}
    * feature count      F in {8, 12, 16, 24, F0}
    * aggregation        weighted vs. unweighted
    * rounds             R up to 20
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Subsampling + leakage control (Section III-B)."""

    # The official UNB raw distribution, not the pre-merged Kaggle mirror -- see the dataset
    # root decision documented in ascon_smart_agri.data.subsample's module docstring.
    dataset_root: Path = Path("data/ciciot2023_raw")
    # Capped, stratified subsample target M ~ 1.5-2e6 records (III-B1 / R1).
    subsample_target: int = 1_800_000
    # Per-class cap kappa_c: compress dominant DDoS classes, keep every rare-family instance.
    # Calibrated against the raw UNB distribution (not the Kaggle mirror this project no longer
    # uses): 70_000 caps 24 of 34 classes and lands the realised total at 1,772,371 records,
    # inside the paper's stated M ~ 1.5-2e6 (Section III-B1). See ascon_smart_agri.data.subsample.
    per_class_cap: int = 70_000
    chunk_size: int = 500_000  # fixed-size CSV parts read to bound peak memory (R1)
    block_size: int = 256  # contiguous-record block; splitting operates on blocks (III-B3)
    test_fraction: float = 0.2  # stratified subset of blocks -> shared global test set
    seed: int = 0


class FeatureConfig(BaseModel):
    """Four-stage selection, fitted on training blocks only (Section III-C)."""

    correlation_tau: float = 0.95  # Stage 2 Spearman prune threshold
    f_sweep: list[int] = Field(default_factory=lambda: [8, 12, 16, 24])  # + F0, Stage 4
    selected_f: int = 16  # chosen at the knee of the validation macro-F1 curve
    rf_n_estimators: int = 200  # for Stage-3 impurity importance
    # Stage-3 reciprocal rank fusion constant. The paper's equation for this is missing from
    # docs/design_paper.md (the extraction dropped every display equation before Eq. 12), so
    # this is the canonical k=60 of Cormack et al. (2009) -- see features/selection.py.
    rrf_k: int = 60
    # Rows drawn (seeded) from the training split for Spearman/MI/random-forest statistics.
    # None uses every training row; 200k keeps a full fit near a minute at stable estimates.
    selection_sample_size: int | None = 200_000


class SequenceConfig(BaseModel):
    """Windowing over contiguous same-label runs (Section III-D)."""

    window: int = 16  # W; label is that of the final record: y_i = y_{i+W-1}
    w_sweep: list[int] = Field(default_factory=lambda: [1, 8, 16, 32])  # W=1 = ablation


class ModelConfig(BaseModel):
    """GRU detector (Section III-E). Default sizing reproduces Eq. (19) = 33,800 params."""

    n_features: int = 16  # F
    hidden_size: int = 96  # H
    n_classes: int = 8  # C: benign + 7 attack families

    @field_validator("n_classes")
    @classmethod
    def _classes_is_eight(cls, v: int) -> int:
        if v != 8:
            raise ValueError("Primary task is C=8 (benign + 7 families) per Section III-B3.")
        return v


class FederatedConfig(BaseModel):
    """Federated simulation (Section III-F)."""

    n_clients: int = 3  # K (A1: simulated as independent processes/objects)
    dirichlet_alpha: float = 0.5  # block-level Dirichlet partition (Eq. 20)
    alpha_sweep: list[float] = Field(default_factory=lambda: [0.1, 0.5, 100.0])
    rounds: int = 20  # R
    local_epochs: int = 3  # E
    e_sweep: list[int] = Field(default_factory=lambda: [1, 3, 5])
    aggregation: Literal["weighted", "unweighted"] = "weighted"
    # FedProx proximal coefficient (R4 fallback). None => plain FedAvg.
    fedprox_mu: float | None = None


class CryptoConfig(BaseModel):
    """Ascon-AEAD128, NIST SP 800-232 (Section III-G).

    IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER (CLAUDE.md golden rule 1; see
    docs/design_paper.md's "Implementation deviation" section): Ascon no longer runs on the
    gateway-to-cloud telemetry channel Eq. (27)'s ``ad_fields`` describes -- that channel is now
    plaintext. ``ad_fields`` is kept as a record of the (now unencrypted) routing-metadata
    tuple's shape; ``weight_ad_fields``/``weight_schema_version`` describe the AD this config's
    Ascon primitive now actually authenticates, on the federated weight transport
    (federated/crypto.py, Channel 3).
    """

    key_bits: int = 128
    nonce_bits: int = 128
    tag_bits: int = 128
    # Telemetry routing-metadata layout, Eq. (27) -- no longer cryptographic AD (see above).
    ad_fields: list[str] = Field(
        default_factory=lambda: ["edge_id", "device_id", "counter", "schema_version"]
    )
    # Federated weight-transport associated-data layout (federated/crypto.py's
    # WeightAssociatedData): the channel Ascon-AEAD128 actually protects now.
    weight_ad_fields: list[str] = Field(
        default_factory=lambda: ["client_id", "round_index", "direction", "schema_version"]
    )
    weight_schema_version: str = "1"


class TelemetryConfig(BaseModel):
    """MQTT/JSON telemetry simulation + feature-provenance adapter (Section III-H, gap G6)."""

    device_ids: list[str] = Field(default_factory=lambda: ["soil01", "soil02", "soil03"])
    messages_per_stream: int = 100  # total across all devices, round-robin
    schema_version: str = "v1"
    # Plausible agricultural sensor ranges for the simulated payload (Section III-H's own
    # example: {"deviceId": "soil01", "temperature": 24.8, "soilMoisture": 42.5}).
    temperature_range_c: tuple[float, float] = (15.0, 35.0)
    soil_moisture_range_pct: tuple[float, float] = (0.0, 100.0)
    seed: int = 0


class EvalConfig(BaseModel):
    """Evaluation protocol (Section III-I)."""

    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])  # >= 3, mean +/- std
    # Accuracy at/above this triggers the "expected, look at macro-F1/FPR" note (III-I5).
    near_ceiling_accuracy: float = 0.98

    @field_validator("seeds")
    @classmethod
    def _at_least_three_seeds(cls, v: list[int]) -> list[int]:
        if len(v) < 3:
            raise ValueError("Section III-I4 requires >= 3 seeds; single runs are not findings.")
        return v


class RunConfig(BaseModel):
    """Top-level config aggregating every stage."""

    run_name: str = "default"
    output_dir: Path = Path("artifacts")
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    crypto: CryptoConfig = Field(default_factory=CryptoConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a :class:`RunConfig` from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return RunConfig.model_validate(raw)
