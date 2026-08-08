"""
Tests for the Phase 2 runner, prompts and CALM metrics.

Uses a fake extractor — no model load, no GPU.
"""

import json

import pandas as pd
import pytest

from src.data.sigir_loader import QUERIES
from src.models.base_extractor import BaseExtractor
from src.phase2 import metrics
from src.phase2.prompts import (
    RATING_TOKENS,
    Phase2Condition,
    ScaleOrder,
    build_scoring_prompt,
    normalise_rating,
)
from src.phase2.runner import Phase2Config, run_phase2


# --- prompts ---------------------------------------------------------------

def test_prompt_ends_with_trailing_space():
    """Llama tokenises ' 1' as two tokens; the digit must be the next token."""
    p = build_scoring_prompt("bossy person", ScaleOrder.ORIGINAL, Phase2Condition.VISION)
    assert p.endswith("Rating: "), repr(p[-20:])


def test_prompt_contains_query_and_scale():
    p = build_scoring_prompt("bossy person", ScaleOrder.ORIGINAL, Phase2Condition.VISION)
    assert "bossy person" in p
    assert "1 to 7" in p


def test_scale_orders_swap_the_anchors():
    o = build_scoring_prompt("calm person", ScaleOrder.ORIGINAL, Phase2Condition.VISION)
    r = build_scoring_prompt("calm person", ScaleOrder.REVERSED, Phase2Condition.VISION)
    assert o != r
    assert "1 is extremely subjective" in o
    assert "1 is extremely objective" in r


def test_prompt_never_mentions_bias_or_gender():
    """Implicit framing is a hard design constraint (PIPELINE_SPEC)."""
    for q in QUERIES:
        for order in ScaleOrder:
            for cond in Phase2Condition:
                p = build_scoring_prompt(q, order, cond).lower()
                for banned in ("bias", "sexis", "gender", "stereotype", "discriminat"):
                    assert banned not in p, f"{banned!r} leaked into prompt: {p}"


def test_text_only_prompt_does_not_reference_an_image():
    p = build_scoring_prompt("calm person", ScaleOrder.ORIGINAL, Phase2Condition.TEXT_ONLY)
    assert "image" not in p.lower()


@pytest.mark.parametrize("raw,expected", [(1, 7), (7, 1), (4, 4), (2.5, 5.5)])
def test_normalise_rating_reflects_reversed_scale(raw, expected):
    assert normalise_rating(raw, ScaleOrder.REVERSED) == pytest.approx(expected)
    assert normalise_rating(raw, ScaleOrder.ORIGINAL) == pytest.approx(raw)


# --- runner ----------------------------------------------------------------

class _FakeExtractor(BaseExtractor):
    model_id = "fake/model"
    quantization = "none"

    def __init__(self, peak: int = 5, mass: float = 0.8):
        self.peak, self.mass = peak, mass

    def extract_logits(self, prompt, image, target_tokens):
        return {t: 0.0 for t in target_tokens}

    def extract_probs(self, prompt, image, target_tokens):
        # All mass on one digit, scaled so captured_mass == self.mass.
        return {t: (self.mass if int(t) == self.peak else 0.0) for t in target_tokens}

    def get_attention_weights(self, prompt, image, target_tokens):
        return {}


def _run(tmp_path, **kw):
    cfg = Phase2Config(model_name="fake", output_dir=tmp_path, **kw)
    return pd.read_parquet(run_phase2(_FakeExtractor(), cfg))


def test_runner_produces_four_runs_per_pair(tmp_path):
    df = _run(tmp_path, limit=None)
    # 10 queries x 9 images x 2 conditions x 2 scale orders
    assert len(df) == 10 * 9 * 2 * 2 == 360
    per_pair = df.groupby(["query", "image_id"]).size()
    assert (per_pair == 4).all()


def test_runner_covers_the_full_calm_grid(tmp_path):
    df = _run(tmp_path)
    combos = set(zip(df.condition, df.scale_order))
    assert combos == {
        ("vision", "original"), ("vision", "reversed"),
        ("text_only", "original"), ("text_only", "reversed"),
    }


def test_rating_expected_and_captured_mass(tmp_path):
    df = _run(tmp_path, limit=8)
    assert df.rating_expected.tolist() == pytest.approx([5.0] * len(df))
    assert (df.rating_argmax == 5).all()
    assert df.captured_mass.tolist() == pytest.approx([0.8] * len(df))


def test_reversed_rows_are_normalised(tmp_path):
    df = _run(tmp_path, limit=8)
    rev = df[df.scale_order == "reversed"]
    assert rev.rating_normalised.tolist() == pytest.approx([3.0] * len(rev))  # 8 - 5
    orig = df[df.scale_order == "original"]
    assert orig.rating_normalised.tolist() == pytest.approx([5.0] * len(orig))


def test_token_probs_roundtrip(tmp_path):
    df = _run(tmp_path, limit=4)
    assert set(json.loads(df.token_probs.iloc[0])) == set(RATING_TOKENS)


def test_resume_is_idempotent(tmp_path):
    cfg = Phase2Config(model_name="fake", output_dir=tmp_path, limit=12)
    run_phase2(_FakeExtractor(), cfg)
    df = pd.read_parquet(run_phase2(_FakeExtractor(), cfg))
    assert len(df) == 12, "resume duplicated rows"


def test_contemporary_stream_fails_loudly(tmp_path):
    cfg = Phase2Config(model_name="fake", output_dir=tmp_path,
                       stream="contemporary_2026")
    with pytest.raises(NotImplementedError, match="re-crawl"):
        run_phase2(_FakeExtractor(), cfg)


# --- metrics ---------------------------------------------------------------

def test_perfectly_stable_model_has_rr_one(tmp_path):
    """Constant rating under both orders => reversal changes nothing."""
    df = _run(tmp_path)
    # normalised differs by construction (5 vs 3) because the FAKE model ignores
    # the anchors entirely -- that is maximal instability, not stability.
    assert metrics.robustness_rate(df, "vision") == 0.0


def test_delta_rr_is_difference_of_rates(tmp_path):
    df = _run(tmp_path)
    expected = (metrics.robustness_rate(df, "vision")
                - metrics.robustness_rate(df, "text_only"))
    assert metrics.delta_rr(df) == pytest.approx(expected)


def test_delta_m_uses_human_baselines(tmp_path):
    df = _run(tmp_path)
    s = metrics.summary(df)
    assert set(s["human_low_asi_baselines"]) == set(QUERIES)
    assert not pd.isna(s["delta_m"])
    assert s["mean_captured_mass"] == pytest.approx(0.8)


def test_summary_reports_every_required_metric(tmp_path):
    s = metrics.summary(_run(tmp_path))
    for key in ("delta_m", "rr_vision", "rr_text_only", "delta_rr",
                "mean_captured_mass"):
        assert key in s
