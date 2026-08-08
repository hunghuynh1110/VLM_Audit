"""
Tests for the SIGIR 2018 loader.

These pin the two bugs that made get_image_paths() silently useless:
  1. `{i}_imageurl` was parsed with ast.literal_eval, but the field is a
     newline-separated string. Parsing threw, the except branch wrapped the
     whole 38-line blob as one "url", and Path(blob).name returned only the
     last line — so the loader saw 1 image per query instead of 38.
  2. Paths were rebuilt as images_dir / filename, discarding the "0/".."9/"
     directory prefix that the archive actually uses.

Both failed silently: the function returned plausible-looking Paths that did
not exist, and its docstring explicitly said it did not check.
"""

import pytest

from src.data.sigir_loader import QUERIES, get_image_paths, compute_low_asi_baselines

K = 9  # PIPELINE_SPEC retrieval depth, matches the SIGIR 3x3 grid


def _images_available() -> bool:
    try:
        get_image_paths(k=1, verify=True)
        return True
    except FileNotFoundError:
        return False


needs_images = pytest.mark.skipif(
    not _images_available(),
    reason="images.zip not extracted to data/sigir2018/images/",
)


@needs_images
def test_all_ten_queries_present():
    paths = get_image_paths(k=K)
    assert set(paths) == set(QUERIES)
    assert len(paths) == 10


@needs_images
def test_exactly_k_images_per_query():
    paths = get_image_paths(k=K)
    for q, ps in paths.items():
        assert len(ps) == K, f"{q!r} returned {len(ps)} images, expected {K}"


@needs_images
def test_every_returned_path_exists():
    for q, ps in get_image_paths(k=K).items():
        for p in ps:
            assert p.exists(), f"{q!r}: missing {p}"


@needs_images
def test_paths_keep_their_query_subdirectory():
    """Regression: the folder prefix identifies the query and must survive."""
    for q, ps in get_image_paths(k=K).items():
        for p in ps:
            assert p.parent.name.isdigit(), (
                f"{q!r}: expected a numbered query subdirectory, got {p.parent.name!r}"
            )


@needs_images
def test_each_query_maps_to_one_folder():
    """All images for a query live in the same numbered folder, and folders are unique."""
    folders = {}
    for q, ps in get_image_paths(k=K).items():
        names = {p.parent.name for p in ps}
        assert len(names) == 1, f"{q!r} spans multiple folders: {names}"
        folders[q] = names.pop()
    assert len(set(folders.values())) == 10, f"folders not unique per query: {folders}"


@needs_images
def test_full_ranked_list_is_38_and_k_is_a_prefix():
    full = get_image_paths(k=None)
    topk = get_image_paths(k=K)
    for q in QUERIES:
        assert len(full[q]) == 38, f"{q!r} has {len(full[q])} images, expected 38"
        assert full[q][:K] == topk[q], f"{q!r}: k-slice is not the ranked prefix"


@needs_images
def test_no_duplicate_images_within_a_query():
    for q, ps in get_image_paths(k=None).items():
        assert len(ps) == len(set(ps)), f"{q!r} contains duplicates"


def test_missing_images_raise_a_useful_error(tmp_path):
    """verify=True must fail loudly rather than return non-existent paths."""
    with pytest.raises(FileNotFoundError, match="Extract images.zip"):
        get_image_paths(images_subdir="definitely_not_here", k=K, verify=True)


def test_verify_false_returns_paths_without_checking():
    paths = get_image_paths(images_subdir="definitely_not_here", k=K, verify=False)
    assert len(paths) == 10


def test_low_asi_baselines_are_on_the_1_to_7_scale():
    """r̄_lowASI,q feeds δ_m directly; an out-of-range value would corrupt it."""
    baselines = compute_low_asi_baselines()
    assert set(baselines) == set(QUERIES)
    for q, v in baselines.items():
        assert 1.0 <= v <= 7.0, f"{q!r} baseline {v} outside the [1,7] objectivity scale"
