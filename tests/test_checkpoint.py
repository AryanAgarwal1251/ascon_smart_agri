"""Unit tests for model checkpoint save/load (Phase 7).

Until this module, no training run in this project ever persisted a model to disk. The property
that matters: a saved-then-loaded model produces IDENTICAL predictions to the original.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ascon_smart_agri.model.checkpoint import load_model, read_metadata, save_model
from ascon_smart_agri.model.gru import build_detector


def test_save_then_load_reproduces_identical_predictions(tmp_path: Path) -> None:
    model = build_detector(4, 8, 3)
    model.eval()
    x = torch.randn(5, 6, 4)
    with torch.no_grad():
        original = model(x)

    path = save_model(model, tmp_path / "model.safetensors")
    loaded = load_model(path, n_features=4, hidden_size=8, n_classes=3)

    with torch.no_grad():
        restored = loaded(x)
    torch.testing.assert_close(original, restored)


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    model = build_detector(2, 4, 2)

    path = save_model(model, tmp_path / "nested" / "dir" / "model.safetensors")

    assert path.exists()


def test_metadata_round_trips(tmp_path: Path) -> None:
    model = build_detector(2, 4, 2)
    path = save_model(model, tmp_path / "model.safetensors", metadata={"seed": "0", "alpha": "0.5"})

    assert read_metadata(path) == {"seed": "0", "alpha": "0.5"}


def test_metadata_defaults_to_empty(tmp_path: Path) -> None:
    model = build_detector(2, 4, 2)
    path = save_model(model, tmp_path / "model.safetensors")

    assert read_metadata(path) == {}


def test_loaded_model_is_in_eval_mode(tmp_path: Path) -> None:
    model = build_detector(2, 4, 2)
    path = save_model(model, tmp_path / "model.safetensors")

    loaded = load_model(path, n_features=2, hidden_size=4, n_classes=2)

    assert loaded.training is False


def test_checkpoint_is_safetensors_not_pickle(tmp_path: Path) -> None:
    model = build_detector(2, 4, 2)
    path = save_model(model, tmp_path / "model.safetensors")

    header_start = path.read_bytes()[:256]
    assert not header_start.startswith(b"\x80")  # pickle protocol marker
    assert b"__metadata__" in header_start or b"dtype" in header_start  # safetensors JSON header
