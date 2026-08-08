import numpy as np
import pytest
from PIL import Image

from src.data.asi_items import ModalityCondition
from src.data.image_generators import (
    generate_gaussian_noise,
    generate_gray_patch,
    get_condition_image,
)

SIZE = (256, 256)  # use small size in tests — faster, same logic as 1024x1024


# --- Gaussian noise ---

def test_noise_size():
    img = generate_gaussian_noise(size=SIZE)
    assert img.size == SIZE
    assert img.mode == "RGB"


def test_noise_pixel_range():
    img = generate_gaussian_noise(size=SIZE)
    arr = np.array(img)
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_noise_not_uniform():
    img = generate_gaussian_noise(size=SIZE)
    arr = np.array(img, dtype=float)
    assert arr.std() > 10, "Noise image looks suspiciously flat"


def test_noise_reproducible():
    img1 = generate_gaussian_noise(size=SIZE, seed=42)
    img2 = generate_gaussian_noise(size=SIZE, seed=42)
    assert np.array_equal(np.array(img1), np.array(img2))


def test_noise_different_seeds_differ():
    img1 = generate_gaussian_noise(size=SIZE, seed=42)
    img2 = generate_gaussian_noise(size=SIZE, seed=99)
    assert not np.array_equal(np.array(img1), np.array(img2))


# --- Gray patch ---
#
# Replaced the humanoid silhouette after Gate 1 stimulus validation: the
# silhouette leaked gender (gap 0.853, threshold 0.20) and race (max P 0.547).
# The gray patch measured 0.062. See findings/stimulus_validation/summary.md.
# These tests pin the property that made it acceptable — it carries no shape.

def test_gray_patch_size():
    img = generate_gray_patch(size=SIZE)
    assert img.size == SIZE
    assert img.mode == "RGB"


def test_gray_patch_is_perfectly_uniform():
    """The whole point of the gray patch: zero structure to read signal from."""
    img = generate_gray_patch(size=SIZE)
    arr = np.array(img)
    unique_values = np.unique(arr.reshape(-1, 3), axis=0)
    assert len(unique_values) == 1, (
        f"Gray patch must be a single flat colour; found {len(unique_values)} distinct "
        "colours. Any structure reintroduces the demographic leakage that got the "
        "silhouette rejected."
    )


def test_gray_patch_expected_colour():
    img = generate_gray_patch(size=SIZE)
    arr = np.array(img)
    assert tuple(arr[0, 0]) == (150, 150, 150)


def test_gray_patch_reproducible():
    a = np.array(generate_gray_patch(size=SIZE))
    b = np.array(generate_gray_patch(size=SIZE))
    assert np.array_equal(a, b)


# --- get_condition_image ---

def test_condition_text_only_returns_none():
    assert get_condition_image(ModalityCondition.TEXT_ONLY) is None


def test_condition_noise_returns_image():
    img = get_condition_image(ModalityCondition.NOISE)
    assert isinstance(img, Image.Image)


def test_condition_gray_patch_returns_image():
    img = get_condition_image(ModalityCondition.GRAY_PATCH)
    assert isinstance(img, Image.Image)


def test_all_conditions_covered():
    """Every ModalityCondition must be handled — a new one must not fall through."""
    for cond in ModalityCondition:
        result = get_condition_image(cond)
        assert result is None or isinstance(result, Image.Image)


# --- Caching ---

def test_cache_files_created(tmp_path, monkeypatch):
    import src.data.image_generators as gen_mod
    monkeypatch.setattr(gen_mod, "_STIMULI_DIR", tmp_path)

    gen_mod.generate_gaussian_noise(size=SIZE)
    gen_mod.generate_gray_patch(size=SIZE)

    assert (tmp_path / f"noise_{SIZE[0]}x{SIZE[1]}_seed42.png").exists()
    assert (tmp_path / f"gray_patch_{SIZE[0]}x{SIZE[1]}.png").exists()


def test_cached_load_matches(tmp_path, monkeypatch):
    import src.data.image_generators as gen_mod
    monkeypatch.setattr(gen_mod, "_STIMULI_DIR", tmp_path)

    img1 = gen_mod.generate_gaussian_noise(size=SIZE)
    img2 = gen_mod.generate_gaussian_noise(size=SIZE)  # loads from cache
    assert np.array_equal(np.array(img1), np.array(img2))
