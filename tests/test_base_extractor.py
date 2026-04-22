"""
Tests for BaseExtractor (and its softmax_probs helper).

We test via a concrete minimal subclass — no model loading required.
"""

import math
import pytest
from src.models.base_extractor import BaseExtractor


class _DummyExtractor(BaseExtractor):
    """Minimal concrete implementation for testing the base class utilities."""

    def extract_logits(self, prompt, image, target_tokens):
        return {tok: float(i) for i, tok in enumerate(target_tokens)}

    def get_attention_weights(self, prompt, image, target_tokens):
        return {}


@pytest.fixture
def extractor():
    return _DummyExtractor()


def test_extract_logits_returns_dict(extractor):
    result = extractor.extract_logits("test prompt", None, ["yes", "no"])
    assert isinstance(result, dict)
    assert set(result.keys()) == {"yes", "no"}


def test_extract_logits_values_are_float(extractor):
    result = extractor.extract_logits("test", None, ["yes", "no"])
    for v in result.values():
        assert isinstance(v, float)


def test_softmax_probs_sum_to_one(extractor):
    logits = {"yes": 2.0, "no": 1.0}
    probs = extractor.softmax_probs(logits)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_softmax_probs_values_in_range(extractor):
    logits = {"yes": 5.0, "no": -3.0, "uncertain": 0.5}
    probs = extractor.softmax_probs(logits)
    for p in probs.values():
        assert 0.0 < p < 1.0


def test_softmax_probs_preserves_order(extractor):
    logits = {"a": 3.0, "b": 1.0, "c": 2.0}
    probs = extractor.softmax_probs(logits)
    assert probs["a"] > probs["c"] > probs["b"]


def test_softmax_probs_numerically_stable(extractor):
    # Very large logit difference — should not overflow or produce NaN
    logits = {"yes": 1000.0, "no": -1000.0}
    probs = extractor.softmax_probs(logits)
    assert not math.isnan(probs["yes"])
    assert not math.isnan(probs["no"])
    assert probs["yes"] > 0.999


def test_softmax_equal_logits_give_uniform(extractor):
    logits = {"a": 1.0, "b": 1.0}
    probs = extractor.softmax_probs(logits)
    assert abs(probs["a"] - 0.5) < 1e-6


def test_arbitrary_target_tokens(extractor):
    tokens = ["1", "2", "3", "4", "5", "6", "7"]
    result = extractor.extract_logits("rate this", None, tokens)
    assert set(result.keys()) == set(tokens)


def test_get_attention_weights_returns_dict(extractor):
    weights = extractor.get_attention_weights("test", None, ["yes", "no"])
    assert isinstance(weights, dict)
