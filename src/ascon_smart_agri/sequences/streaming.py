"""Streaming, per-device window assembly for the runtime pipeline (Phase 7, Section III-A/D).

``sequences/windowing.py`` builds ``(N, W, F)`` sequences OFFLINE, from an already-collected
batch of rows. This module does the streaming equivalent the runtime plane needs: as single
provenanced feature vectors arrive one message at a time
(``telemetry/provenance.py``'s ``FeatureProvenanceAdapter.network_features_for``), accumulate
the last ``W`` per DEVICE and yield a window once ``W`` have arrived -- exactly matching offline
windowing's ``y_i = y_{i+W-1}`` semantics: the window "completes" and becomes classifiable the
moment the W-th record for that device arrives.

Deliberately its own module, not folded into ``windowing.py``: that module's functions all
assume the full row order is already known and available in memory; this one holds STATE across
calls (one rolling buffer per device) and is built for exactly the call-by-call arrival Phase
5's telemetry stream produces. Kept separate for the same "phase boundary" reason
``provenance.py``'s own docstring already gives for not assembling windows itself.

Implementation notes:

* **Rolling per device, not per stream.** Section III-D's contiguity rule ("windows... within a
  single source file") only makes sense per physical source; the runtime analogue is per
  DEVICE, since a device's own successive telemetry messages are the closest thing to a
  contiguous stream this architecture has. A single buffer shared across devices would silently
  splice unrelated devices' readings into one "window".
* **A window is emitted the moment its buffer reaches W, then slides by one** (oldest record
  dropped, buffer keeps accepting) -- consistent with ``y_i = y_{i+W-1}``: a fresh, classifiable
  window is available after every new message once the buffer first fills, not merely once
  every W messages.
* **No feature is ever fabricated to fill a short buffer.** Before a device's W-th message
  arrives, ``push`` returns ``None`` rather than padding with zeros or repeating a value --
  padding would fabricate structure the real message stream never contained, the same objection
  that already rules out synthetic oversampling (Section III-E) and fabricated network features
  (Section III-H).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .._types import Array


class DeviceWindowBuffer:
    """Assembles ``(W, F)`` windows per device from single feature vectors, one push at a time."""

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window = window
        self._buffers: dict[str, deque[Array]] = {}
        self._n_features: dict[str, int] = {}

    def push(self, device_id: str, features: Array) -> Array | None:
        """Add one record for ``device_id``; return the completed ``(W, F)`` window, or ``None``.

        Returns ``None`` until this device has accumulated ``window`` records; from then on,
        every call returns a fresh window (the buffer slides by one, oldest record dropped).
        """
        vector = np.asarray(features)
        if vector.ndim != 1:
            raise ValueError(f"features must be a 1-D (F,) vector, got shape {vector.shape}")

        expected_f = self._n_features.setdefault(device_id, vector.shape[0])
        if vector.shape[0] != expected_f:
            raise ValueError(
                f"device {device_id!r}: expected {expected_f} features, got {vector.shape[0]}"
            )

        buffer = self._buffers.setdefault(device_id, deque(maxlen=self.window))
        buffer.append(vector)
        if len(buffer) < self.window:
            return None
        return np.stack(buffer)

    def reset(self, device_id: str | None = None) -> None:
        """Clear one device's buffer, or every device's if ``device_id`` is ``None``."""
        if device_id is None:
            self._buffers.clear()
            self._n_features.clear()
        else:
            self._buffers.pop(device_id, None)
            self._n_features.pop(device_id, None)

    def depth(self, device_id: str) -> int:
        """How many records this device's buffer currently holds (0 to ``window``)."""
        return len(self._buffers.get(device_id, ()))
