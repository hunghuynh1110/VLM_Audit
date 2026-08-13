"""
Tests for the shared vision-extractor machinery.

The degeneracy self-test is the guard that stands between a silently broken
node and thousands of rows of noise, so it is tested against the exact
signatures that actually occurred: constant logits (uniform softmax) and
non-finite logits. Model loading is stubbed -- none of this needs a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from src.models.hf_vision_extractor import HFVisionExtractor

VOCAB = {"yes": 10, "no": 11, "Ġmaybe": 12, "1": 20, "2": 21,
         "un": 30, "likely": 31}


class _StubTokenizer:
    def get_vocab(self):
        return dict(VOCAB)

    def encode(self, text, add_special_tokens=False):
        # "unlikely" is deliberately multi-token, to exercise the warning path
        if text == "unlikely":
            return [VOCAB["un"], VOCAB["likely"]]
        return [VOCAB.get(text, 0)]


class _StubExtractor(HFVisionExtractor):
    """Concrete extractor whose 'model' returns a fixed logit vector."""

    def __init__(self, logits: torch.Tensor, run_selftest: bool = False):
        self.processor = SimpleNamespace(tokenizer=_StubTokenizer())
        self._logits = logits

        def _model(**kwargs):
            # shape (batch, seq, vocab); only the last position is read
            return SimpleNamespace(logits=self._logits.unsqueeze(0).unsqueeze(0))

        self.model = _model
        self._token_id_cache = {}
        self._last_attention = None
        if run_selftest:
            self._assert_not_degenerate()

    def _build_inputs(self, prompt, image, assistant_prefix=""):
        return {}


def _healthy(vocab_size=64):
    torch.manual_seed(0)
    return torch.randn(vocab_size)


class TestResolveTokenId:
    def test_plain_surface_form_wins(self):
        ex = _StubExtractor(_healthy())
        assert ex._resolve_token_id("yes") == 10

    def test_falls_back_to_space_prefixed_form(self):
        ex = _StubExtractor(_healthy())
        assert ex._resolve_token_id("maybe") == 12

    def test_result_is_cached(self):
        ex = _StubExtractor(_healthy())
        ex._resolve_token_id("yes")
        assert ex._token_id_cache["yes"] == 10

    def test_multi_token_target_warns_and_takes_first(self):
        ex = _StubExtractor(_healthy())
        with pytest.warns(UserWarning, match="sub-tokens"):
            assert ex._resolve_token_id("unlikely") == VOCAB["un"]


class TestSelfTest:
    def test_constant_logits_are_rejected(self):
        # The exact cross-GPU-zero-copy signature: every logit identical, which
        # softmaxes to a perfectly uniform 1/vocab_size distribution.
        with pytest.raises(RuntimeError, match="degenerate"):
            _StubExtractor(torch.zeros(64), run_selftest=True)

    def test_nonfinite_logits_are_rejected(self):
        bad = _healthy()
        bad[3] = float("nan")
        with pytest.raises(RuntimeError, match="degenerate"):
            _StubExtractor(bad, run_selftest=True)

    def test_healthy_logits_pass(self):
        ex = _StubExtractor(_healthy(), run_selftest=True)
        assert ex is not None


class TestExtractProbs:
    def test_probabilities_come_from_the_full_vocabulary_softmax(self):
        logits = torch.full((64,), -20.0)
        logits[10] = 0.0          # "yes"
        logits[11] = 0.0          # "no"
        ex = _StubExtractor(logits)

        probs = ex.extract_probs("p", None, ["yes", "no"])
        # Two dominant tokens of equal logit -> ~0.5 each, and together they
        # should account for nearly all the mass (this is captured_mass).
        assert probs["yes"] == pytest.approx(probs["no"], rel=1e-6)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)

    def test_captured_mass_is_tiny_when_targets_are_off_distribution(self):
        logits = torch.full((64,), -20.0)
        logits[0] = 10.0          # all mass elsewhere
        ex = _StubExtractor(logits)
        probs = ex.extract_probs("p", None, ["yes", "no"])
        assert sum(probs.values()) < 1e-6

    def test_extract_logits_returns_raw_scores(self):
        logits = _healthy()
        ex = _StubExtractor(logits)
        got = ex.extract_logits("p", None, ["yes", "no"])
        assert got["yes"] == pytest.approx(logits[10].item())
        assert got["no"] == pytest.approx(logits[11].item())
