"""Shared type aliases.

``Array`` is a parameterised numpy array alias so annotations satisfy mypy's strict
``disallow_any_generics`` without pinning a specific dtype in these interface stubs. Phase code
may narrow to ``npt.NDArray[np.floating]`` / ``[np.integer]`` where the dtype is known.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.generic]
