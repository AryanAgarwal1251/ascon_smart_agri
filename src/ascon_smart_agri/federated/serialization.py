"""Parameter serialization across the client<->server boundary (Phase 4, Section III-F2).

Parameters are serialised in a STRUCTURED TENSOR FORMAT (safetensors), never pickled:
arbitrary-object deserialisation across a transport boundary is a code-execution hazard even
in a prototype. Only the parameter vector theta_k and the sequence count n_k ever cross the
boundary; raw records never do.

Implementation notes:

* ``n_k`` travels in the safetensors **header metadata**, not as a tensor smuggled into the
  state dict. Keeping it out of the tensor map means a deserialised state dict can be loaded
  straight into a module without first stripping a bookkeeping key that would fail a strict
  ``load_state_dict``.
* ``safetensors.torch.load`` returns tensors only, so the metadata is read by parsing the
  container header directly: an 8-byte little-endian length followed by a JSON header carrying
  ``__metadata__``. That layout is part of the safetensors format specification, not an
  implementation detail of this repository.
* The metadata value is a string because the format requires ``dict[str, str]``; it is parsed
  back to ``int`` on the way out, and a malformed or absent count is an error rather than a
  silent zero, since ``n_k`` is the FedAvg weight (Eq. 21) and a wrong one skews aggregation
  invisibly.
"""

from __future__ import annotations

import json
import struct

import torch
from safetensors.torch import load, save

StateDict = dict[str, "torch.Tensor"]

_SEQUENCE_COUNT_KEY = "sequence_count"
_HEADER_LENGTH_BYTES = 8


def serialize_state(state: StateDict, sequence_count: int) -> bytes:
    """Serialise a state dict + n_k to safetensors bytes (no pickling)."""
    if sequence_count < 0:
        raise ValueError(f"sequence_count must be non-negative, got {sequence_count}")
    # contiguous(): safetensors rejects non-contiguous tensors, which slicing can produce.
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
    # Annotated explicitly: the pre-commit mypy environment has no safetensors, so save()
    # is Any there while the local run sees the real signature.
    blob: bytes = save(tensors, metadata={_SEQUENCE_COUNT_KEY: str(sequence_count)})
    return blob


def deserialize_state(blob: bytes) -> tuple[StateDict, int]:
    """Inverse of :func:`serialize_state`; returns ``(state, sequence_count)``."""
    if len(blob) < _HEADER_LENGTH_BYTES:
        raise ValueError("blob is too short to be a safetensors container")

    header_length = struct.unpack("<Q", blob[:_HEADER_LENGTH_BYTES])[0]
    header_end = _HEADER_LENGTH_BYTES + header_length
    if header_end > len(blob):
        raise ValueError("safetensors header length exceeds the blob")

    header = json.loads(blob[_HEADER_LENGTH_BYTES:header_end])
    metadata = header.get("__metadata__") or {}
    raw_count = metadata.get(_SEQUENCE_COUNT_KEY)
    if raw_count is None:
        raise ValueError(
            f"serialized state carries no '{_SEQUENCE_COUNT_KEY}' metadata; it is the "
            "FedAvg weight (Eq. 21) and must not be defaulted"
        )
    try:
        sequence_count = int(raw_count)
    except ValueError as exc:
        raise ValueError(
            f"'{_SEQUENCE_COUNT_KEY}' metadata is not an integer: {raw_count!r}"
        ) from exc

    tensors: StateDict = load(blob)
    return tensors, sequence_count
