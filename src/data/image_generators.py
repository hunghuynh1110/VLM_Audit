"""
Generates the two visual stimuli used in Phase 1 modality conditions.

Both images are deterministic and cached to data/stimuli/ on first call.
Subsequent calls load from disk — no regeneration during a run.

  NOISE      → Gaussian noise (semantically empty visual input)
  GRAY_PATCH → Uniform gray rectangle (activates vision encoder without
               any shape the model can read demographic signal from)
  TEXT_ONLY  → None (no image passed to model)
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.config import CFG
from src.data.asi_items import ModalityCondition

__all__ = [
    "generate_gaussian_noise",
    "generate_gray_patch",
    "get_condition_image",
]

_STIMULI_DIR: Path = CFG["paths"]["stimuli"]
_DEFAULT_SIZE: tuple[int, int] = tuple(CFG["phase1"]["noise_image_size"])
_NOISE_SEED = 42


def generate_gaussian_noise(
    size: tuple[int, int] = _DEFAULT_SIZE,
    seed: int = _NOISE_SEED,
) -> Image.Image:
    """
    Return a reproducible Gaussian noise RGB image.
    Cached at data/stimuli/noise_{w}x{h}.png.
    """
    w, h = size
    cache_path = _STIMULI_DIR / f"noise_{w}x{h}_seed{seed}.png"

    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    rng = np.random.default_rng(seed)
    # Sample from N(128, 50²), clip to [0, 255]
    pixels = rng.normal(loc=128, scale=50, size=(h, w, 3))
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels, mode="RGB")

    _STIMULI_DIR.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    return img


def generate_gray_patch(
    size: tuple[int, int] = _DEFAULT_SIZE,
) -> Image.Image:
    """
    Return a uniform gray rectangle with no shapes or structure.
    Cached at data/stimuli/gray_patch_{w}x{h}.png.

    Replaces the humanoid silhouette after validation showed the silhouette
    encodes detectable gender and race signal (gender gap 0.853, max race P 0.547).
    """
    w, h = size
    cache_path = _STIMULI_DIR / f"gray_patch_{w}x{h}.png"

    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    img = Image.new("RGB", (w, h), (150, 150, 150))
    _STIMULI_DIR.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    return img


def get_condition_image(condition: ModalityCondition) -> Image.Image | None:
    """
    Return the image for a given modality condition, or None for TEXT_ONLY.
    This is the single entry point used by the Phase 1 runner.
    """
    if condition == ModalityCondition.TEXT_ONLY:
        return None
    if condition == ModalityCondition.NOISE:
        return generate_gaussian_noise()
    if condition == ModalityCondition.GRAY_PATCH:
        return generate_gray_patch()
    raise ValueError(f"Unknown condition: {condition}")
