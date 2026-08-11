"""
Tests for the cross-GPU copy workaround.

The workaround intercepts every Tensor.to() call in the process, so a mistake in
how it reads the destination device out of that call is not a small bug: parse a
device where there is none and healthy copies get pointlessly staged through the
host; miss one and a cross-GPU copy goes down the broken peer path and silently
returns zeros, which is the exact failure this module exists to prevent.

No GPU required — the device parsing is pure logic, and the parts that need two
GPUs are skipped rather than faked.
"""

import pytest
import torch

from src.models import p2p_workaround
from src.models.p2p_workaround import _is_cross_gpu, _target_device


class TestTargetDevice:
    """Tensor.to() has many overloads; only some of them name a device."""

    @pytest.mark.parametrize("args, kwargs, expected", [
        (("cuda:1",),                      {},                          "cuda:1"),
        ((torch.device("cuda", 1),),       {},                          "cuda:1"),
        ((1,),                             {},                          "cuda:1"),
        (("cpu",),                         {},                          "cpu"),
        (("cuda:1", torch.float16),        {},                          "cuda:1"),
        ((),                               {"device": "cuda:1"},        "cuda:1"),
        ((),          {"device": "cuda:1", "dtype": torch.float16},     "cuda:1"),
    ])
    def test_device_forms_are_recognised(self, args, kwargs, expected):
        assert str(_target_device(args, kwargs)) == expected

    @pytest.mark.parametrize("args, kwargs", [
        ((torch.float32,),          {}),                    # to(dtype)
        ((torch.bfloat16,),         {"non_blocking": True}),
        ((),                        {"dtype": torch.float16}),
        ((),                        {}),                    # to()
        ((True,),                   {}),                    # non_blocking, not device 1
    ])
    def test_non_device_calls_yield_none(self, args, kwargs):
        assert _target_device(args, kwargs) is None

    def test_dtype_strings_are_not_mistaken_for_devices(self):
        # torch rejects these itself, but the parser must not turn them into a
        # device and reroute the call.
        for s in ("float32", "torch.float32", "bfloat16"):
            assert _target_device((s,), {}) is None

    def test_to_other_tensor_takes_that_tensors_device(self):
        other = torch.zeros(2)
        assert _target_device((other,), {}) == other.device


class TestIsCrossGpu:
    def test_cpu_source_is_never_cross_gpu(self):
        assert not _is_cross_gpu(torch.zeros(2), torch.device("cuda", 1))

    def test_cpu_destination_is_never_cross_gpu(self):
        t = torch.zeros(2)
        assert not _is_cross_gpu(t, torch.device("cpu"))

    def test_none_destination_is_never_cross_gpu(self):
        assert not _is_cross_gpu(torch.zeros(2), None)


class TestPatch:
    def test_enabling_is_idempotent_and_leaves_cpu_copies_alone(self):
        p2p_workaround.enable_host_staged_cross_device_copies()
        p2p_workaround.enable_host_staged_cross_device_copies()

        t = torch.arange(8, dtype=torch.float32)
        assert torch.equal(t.to("cpu"), t)
        assert t.to(torch.float16).dtype is torch.float16

    def test_is_affected_is_false_without_two_gpus(self):
        if torch.cuda.device_count() >= 2:
            pytest.skip("needs a single-GPU/CPU host to assert the early exit")
        assert p2p_workaround.is_affected(verbose=False) is False

    def test_probe_uses_the_unpatched_copy(self):
        # If is_affected() went through the patched Tensor.to it would stage via
        # the host, always find the data intact, and disable the workaround on
        # exactly the nodes that need it. Enable here rather than relying on an
        # earlier test having done so, so this holds when run in isolation.
        p2p_workaround.enable_host_staged_cross_device_copies()
        assert p2p_workaround._ORIG_TO is not torch.Tensor.to

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
    def test_cross_gpu_copy_is_correct_once_patched(self):
        p2p_workaround.enable_host_staged_cross_device_copies()
        src = torch.arange(4096, dtype=torch.float32, device="cuda:0")
        assert torch.equal(src.to("cuda:1").cpu(), src.cpu())
