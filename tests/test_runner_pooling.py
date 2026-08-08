"""
Tests for surface-form pooling in the Phase 1 runner.

Regression guard for the measurement bug found in job 27103565: the runner read
only lowercase 'yes'/'no', which together hold ~0.09% of the model's next-token
distribution because a chat reply begins with a capital. That inflated p_yes by
~0.17 systematically.

Uses a fake extractor — no model load.
"""

import json

import pandas as pd
import pytest

from src.data.asi_items import PromptStructure
from src.models.base_extractor import BaseExtractor
from src.phase1.runner import NO_FORMS, YES_FORMS, RunConfig, run_phase1


class _FakeExtractor(BaseExtractor):
    """Returns fixed full-vocab probabilities: tiny lowercase, large capitalised."""

    model_id = "fake/model"
    quantization = "none"

    #  lowercase pair alone  -> p_yes = 0.6/(0.6+0.4)       = 0.60
    #  pooled across forms   -> p_yes = 0.66/(0.66+0.44)    = 0.60  (same ratio)
    #  so we deliberately skew the capitals the OTHER way to catch pooling:
    TABLE = {
        "yes": 0.0006, "Yes": 0.10, "YES": 0.0,
        "no":  0.0004, "No":  0.30, "NO":  0.0,
    }

    def extract_logits(self, prompt, image, target_tokens):
        return {t: 0.0 for t in target_tokens}

    def extract_probs(self, prompt, image, target_tokens):
        return {t: self.TABLE[t] for t in target_tokens}

    def get_attention_weights(self, prompt, image, target_tokens):
        return {}


@pytest.fixture
def df(tmp_path):
    cfg = RunConfig(model_name="fake", output_dir=tmp_path, limit=6)
    path = run_phase1(_FakeExtractor(), cfg)
    return pd.read_parquet(path)


def test_pooling_uses_all_surface_forms(df):
    yes = sum(_FakeExtractor.TABLE[t] for t in YES_FORMS)
    no = sum(_FakeExtractor.TABLE[t] for t in NO_FORMS)
    expected = yes / (yes + no)
    assert df["p_yes"].iloc[0] == pytest.approx(expected)


def test_pooled_result_differs_from_lowercase_only(df):
    """The whole point: capitals dominate and change the answer."""
    lower_only = _FakeExtractor.TABLE["yes"] / (
        _FakeExtractor.TABLE["yes"] + _FakeExtractor.TABLE["no"]
    )
    assert abs(df["p_yes"].iloc[0] - lower_only) > 0.2, (
        "pooled p_yes matches the lowercase-only value — pooling is not happening"
    )


def test_captured_mass_recorded_and_sane(df):
    expected = sum(_FakeExtractor.TABLE.values())
    assert df["captured_mass"].iloc[0] == pytest.approx(expected)
    assert (df["captured_mass"] > 0).all()
    assert (df["captured_mass"] <= 1.0).all()


def test_p_yes_and_p_no_complement(df):
    assert ((df["p_yes"] + df["p_no"] - 1.0).abs() < 1e-9).all()


def test_token_probs_roundtrip(df):
    probs = json.loads(df["token_probs"].iloc[0])
    assert set(probs) == set(YES_FORMS + NO_FORMS)


def test_inversion_sign_still_applied(df):
    """The May sign fix must survive the pooling change."""
    inv = df[df.structure == PromptStructure.INVERSION.value]
    non = df[df.structure != PromptStructure.INVERSION.value]
    if len(inv) and len(non):
        a = inv.iloc[0]
        b = non.iloc[0]
        assert a.bias_score == pytest.approx(-a.polarity * a.p_yes)
        assert b.bias_score == pytest.approx(b.polarity * b.p_yes)


def test_resume_skips_completed_rows(tmp_path):
    cfg = RunConfig(model_name="fake", output_dir=tmp_path, limit=6)
    run_phase1(_FakeExtractor(), cfg)
    path = run_phase1(_FakeExtractor(), cfg)  # second call: everything complete
    assert len(pd.read_parquet(path)) == 6, "resume duplicated rows"
