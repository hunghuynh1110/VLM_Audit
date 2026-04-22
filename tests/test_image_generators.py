import numpy as np
import pytest
from PIL import Image

from src.data.asi_items import ModalityCondition
from src.data.image_generators import (
    generate_gaussian_noise,
    generate_humanoid_silhouette,
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


# --- Humanoid silhouette ---

def test_silhouette_size():
    img = generate_humanoid_silhouette(size=SIZE)
    assert img.size == SIZE
    assert img.mode == "RGB"


def test_silhouette_not_uniform():
    img = generate_humanoid_silhouette(size=SIZE)
    arr = np.array(img)
    unique_values = len(np.unique(arr.reshape(-1, 3), axis=0))
    assert unique_values >= 2, "Silhouette should have at least background and body colors"


def test_silhouette_has_background_color():
    img = generate_humanoid_silhouette(size=SIZE)
    arr = np.array(img)
    bg_color = np.array([210, 210, 210])
    has_bg = np.any(np.all(arr == bg_color, axis=2))
    assert has_bg, "Silhouette missing expected background color (210, 210, 210)"


def test_silhouette_has_body_color():
    img = generate_humanoid_silhouette(size=SIZE)
    arr = np.array(img)
    body_color = np.array([90, 90, 90])
    has_body = np.any(np.all(arr == body_color, axis=2))
    assert has_body, "Silhouette missing expected body color (90, 90, 90)"


# --- get_condition_image ---

def test_condition_text_only_returns_none():
    assert get_condition_image(ModalityCondition.TEXT_ONLY) is None


def test_condition_noise_returns_image():
    img = get_condition_image(ModalityCondition.NOISE)
    assert isinstance(img, Image.Image)


def test_condition_silhouette_returns_image():
    img = get_condition_image(ModalityCondition.SILHOUETTE)
    assert isinstance(img, Image.Image)


# --- Caching ---

def test_cache_files_created(tmp_path, monkeypatch):
    import src.data.image_generators as gen_mod
    monkeypatch.setattr(gen_mod, "_STIMULI_DIR", tmp_path)

    gen_mod.generate_gaussian_noise(size=SIZE)
    gen_mod.generate_humanoid_silhouette(size=SIZE)

    assert (tmp_path / f"noise_{SIZE[0]}x{SIZE[1]}_seed42.png").exists()
    assert (tmp_path / f"silhouette_{SIZE[0]}x{SIZE[1]}.png").exists()


def test_cached_load_matches(tmp_path, monkeypatch):
    import src.data.image_generators as gen_mod
    monkeypatch.setattr(gen_mod, "_STIMULI_DIR", tmp_path)

    img1 = gen_mod.generate_gaussian_noise(size=SIZE)
    img2 = gen_mod.generate_gaussian_noise(size=SIZE)  # loads from cache
    assert np.array_equal(np.array(img1), np.array(img2))
