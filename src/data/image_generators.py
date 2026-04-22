"""
Generates the two visual stimuli used in Phase 1 modality conditions.

Both images are deterministic and cached to data/stimuli/ on first call.
Subsequent calls load from disk — no regeneration during a run.

  NOISE      → Gaussian noise (semantically empty visual input)
  SILHOUETTE → Gray featureless human silhouette (activates vision encoder
               without introducing gendered or identifying content)
  TEXT_ONLY  → None (no image passed to model)
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.config import CFG
from src.data.asi_items import ModalityCondition

__all__ = [
    "generate_gaussian_noise",
    "generate_humanoid_silhouette",
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


def generate_humanoid_silhouette(
    size: tuple[int, int] = _DEFAULT_SIZE,
) -> Image.Image:
    """
    Return a gray featureless human silhouette on a neutral background.
    Cached at data/stimuli/silhouette_{w}x{h}.png.

    Shape is purely geometric — no facial features, no clothing, no gender markers.
    """
    w, h = size
    cache_path = _STIMULI_DIR / f"silhouette_{w}x{h}.png"

    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    BG    = (210, 210, 210)  # light gray background
    BODY  = (90, 90, 90)     # dark gray silhouette

    img  = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    # All coordinates scale with image size for resolution independence
    cx = w // 2

    # Head (circle)
    head_r  = int(w * 0.08)
    head_cy = int(h * 0.18)
    draw.ellipse(
        [cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
        fill=BODY,
    )

    # Torso (trapezoid: wider at shoulders, narrower at waist)
    shoulder_w = int(w * 0.18)
    waist_w    = int(w * 0.12)
    torso_top  = head_cy + head_r + int(h * 0.01)
    torso_bot  = torso_top + int(h * 0.28)
    draw.polygon(
        [
            (cx - shoulder_w, torso_top),
            (cx + shoulder_w, torso_top),
            (cx + waist_w,    torso_bot),
            (cx - waist_w,    torso_bot),
        ],
        fill=BODY,
    )

    # Arms
    arm_w     = int(w * 0.055)
    arm_top   = torso_top + int(h * 0.01)
    arm_bot   = arm_top + int(h * 0.26)
    arm_gap   = int(w * 0.005)
    for side in (-1, 1):
        inner = cx + side * (shoulder_w + arm_gap)
        outer = inner + side * arm_w
        x0, x1 = sorted([inner, outer])
        draw.rectangle([x0, arm_top, x1, arm_bot], fill=BODY)

    # Legs
    leg_w   = int(w * 0.075)
    leg_gap = int(w * 0.01)
    leg_top = torso_bot
    leg_bot = leg_top + int(h * 0.30)
    for side in (-1, 1):
        x0 = cx + side * leg_gap
        x1 = x0 + side * leg_w
        draw.rectangle([min(x0, x1), leg_top, max(x0, x1), leg_bot], fill=BODY)

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
    if condition == ModalityCondition.SILHOUETTE:
        return generate_humanoid_silhouette()
    raise ValueError(f"Unknown condition: {condition}")
