"""Parameter serialization across the client<->server boundary (Phase 4, Section III-F2).

Parameters are serialised in a STRUCTURED TENSOR FORMAT (safetensors), never pickled:
arbitrary-object deserialisation across a transport boundary is a code-execution hazard even
in a prototype. Only the parameter vector theta_k and the sequence count n_k ever cross the
boundary; raw records never do.

TODO(Phase 4): implement safetensors (de)serialization of state dicts + n_k metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

StateDict = dict[str, "torch.Tensor"]


def serialize_state(state: StateDict, sequence_count: int) -> bytes:
    """Serialise a state dict + n_k to safetensors bytes (no pickling)."""
    del state, sequence_count
    raise NotImplementedError("Phase 4: safetensors serialization not implemented yet.")


def deserialize_state(blob: bytes) -> tuple[StateDict, int]:
    """Inverse of :func:`serialize_state`; returns ``(state, sequence_count)``."""
    del blob
    raise NotImplementedError("Phase 4: safetensors deserialization not implemented yet.")
