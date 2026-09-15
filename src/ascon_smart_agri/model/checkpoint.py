"""Model checkpoint save/load for the runtime pipeline (Phase 7).

Deliberately separate from ``federated/serialization.py``: that module serialises a state dict
*for the client<->server transport boundary* and bundles ``n_k`` (the FedAvg weight) into the
container's metadata, because that is what crosses the wire each round. This module saves a
finished, deployable model checkpoint -- there is no ``n_k`` to carry, and conflating the two
would make a checkpoint's metadata field mean something different depending on where the file
came from. Both use safetensors (never pickle), per the same III-F2 reasoning.

Until this module, no training run in this project ever persisted a model to disk -- every
Phase 3/4 run trained, reported metrics, and discarded the weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .gru import build_detector


def save_model(
    model: torch.nn.Module, path: Path, *, metadata: dict[str, str] | None = None
) -> Path:
    """Save a model's state dict to ``path`` via safetensors, with optional string metadata.

    ``metadata`` is for provenance (e.g. which config/seed/commit produced this checkpoint) --
    it travels in the safetensors header, not as a smuggled tensor, matching how
    ``federated/serialization.py`` carries ``sequence_count``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path), metadata=metadata or {})
    return path


def load_model(path: Path, *, n_features: int, hidden_size: int, n_classes: int) -> torch.nn.Module:
    """Build a fresh :class:`GRUDetector` at the given sizing and load a saved checkpoint into it.

    The sizing must be passed explicitly rather than inferred from the checkpoint: a state dict
    alone does not carry ``n_features``/``hidden_size``/``n_classes`` in a self-describing way,
    and silently guessing them from tensor shapes would be fragile. Callers get these from the
    same run config the checkpoint was trained under (recorded in the checkpoint's own metadata
    by the training script, and in the run manifest).
    """
    model = build_detector(n_features, hidden_size, n_classes)
    state = load_file(str(path))
    model.load_state_dict(state)
    model.eval()
    return model


def read_metadata(path: Path) -> dict[str, str]:
    """Read a checkpoint's metadata without loading the tensors."""
    with path.open("rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header: dict[str, Any] = json.loads(fh.read(header_len))
    return dict(header.get("__metadata__") or {})
