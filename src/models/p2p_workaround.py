"""
Work around silently-zeroed GPU-to-GPU copies on Bunya.

THE FAULT
    On Bunya's PCIe GPU nodes a direct device-to-device copy does not move the
    data. The destination reads back as all zeros, with no error and no warning:

        src (on cuda:0) : [0, 1, 2, 3, 4, 5, 6, 7, ...]
        src.to(cuda:1)  : [0, 0, 0, 0, 0, 0, 0, 0, ...]

    torch.cuda.can_device_access_peer() reports True for every pair, so torch
    takes the peer path and the copy is lost. Confirmed identical under torch
    2.7.1+cu126, 2.8.0+cu128 and 2.11.0+cu130 (driver 595.58.03, CUDA 13.2), so
    it is the machine, not the library. Host-to-device and device-to-host copies
    are unaffected -- they were used as ground truth to detect this.

WHY IT MATTERED
    device_map="auto" spreads a model over every visible GPU, and accelerate
    moves activations across device boundaries between layers. Every one of
    those crossings delivered zeros, so any sharded model produced a hidden
    state of zeros, which the final RMSNorm and lm_head turn into exactly-zero
    logits -- a perfectly uniform softmax of 1/vocab_size, reported without a
    single warning. See progress/findings_summary.md section 11.

THE WORKAROUND
    Route every cuda->cuda copy through host memory, which is the path that
    works. Only tensors that actually cross a device boundary pay the cost --
    activations of a few MB, not the weights, which are loaded host->device and
    then stay put. Weight loading and single-GPU runs are untouched.

    This is a mitigation, not a repair: the underlying node fault should be
    reported to UQ RCC. Call is_affected() to test a machine before trusting it.
"""

from __future__ import annotations

from typing import Optional

import torch

_PATCHED = False


def _target_device(args: tuple, kwargs: dict) -> Optional[torch.device]:
    """
    Pull the destination device out of a Tensor.to() call, or None.

    Tensor.to has several overloads -- to(device), to(dtype), to(device, dtype),
    to(other_tensor), plus keyword forms -- and only some of them name a device.
    """
    dev = kwargs.get("device")
    if dev is None:
        for a in args:
            if isinstance(a, torch.device):
                dev = a
                break
            if isinstance(a, str):
                # "cuda", "cuda:1", "cpu"; a dtype string never reaches here
                try:
                    dev = torch.device(a)
                except (RuntimeError, TypeError):
                    continue
                break
            if isinstance(a, int) and not isinstance(a, bool):
                dev = torch.device("cuda", a)
                break
            if isinstance(a, torch.Tensor):
                dev = a.device
                break
    if dev is None:
        return None
    if isinstance(dev, (str, int)):
        dev = torch.device(dev) if isinstance(dev, str) else torch.device("cuda", dev)
    return dev


def _is_cross_gpu(src: torch.Tensor, dst: Optional[torch.device]) -> bool:
    if dst is None or src.device.type != "cuda" or dst.type != "cuda":
        return False
    # .to("cuda") with no index means the current device.
    dst_index = dst.index if dst.index is not None else torch.cuda.current_device()
    return dst_index != src.device.index


def enable_host_staged_cross_device_copies() -> None:
    """Make every cuda->cuda copy hop through the CPU. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    orig_to = torch.Tensor.to
    orig_cuda = torch.Tensor.cuda
    orig_copy_ = torch.Tensor.copy_

    def safe_to(self, *args, **kwargs):
        dst = _target_device(args, kwargs)
        if _is_cross_gpu(self, dst):
            # Land on the host first, then go up to the target GPU. Both legs
            # are host<->device copies, which this machine performs correctly.
            return orig_to(orig_to(self, "cpu"), *args, **kwargs)
        return orig_to(self, *args, **kwargs)

    def safe_cuda(self, device=None, *args, **kwargs):
        dst = torch.device("cuda", device) if isinstance(device, int) else (
            torch.device(device) if isinstance(device, str) else device)
        if _is_cross_gpu(self, dst):
            return orig_cuda(orig_to(self, "cpu"), device, *args, **kwargs)
        return orig_cuda(self, device, *args, **kwargs)

    def safe_copy_(self, src, *args, **kwargs):
        if isinstance(src, torch.Tensor) and _is_cross_gpu(src, self.device):
            return orig_copy_(self, orig_to(src, "cpu"), *args, **kwargs)
        return orig_copy_(self, src, *args, **kwargs)

    torch.Tensor.to = safe_to
    torch.Tensor.cuda = safe_cuda
    torch.Tensor.copy_ = safe_copy_
    _PATCHED = True
    print("[p2p_workaround] cuda->cuda copies are now staged through host memory")


def is_affected(verbose: bool = True) -> bool:
    """
    Does a direct GPU-to-GPU copy on this machine lose the data?

    Deliberately uses a known ramp rather than random values, and compares
    against a host-staged copy of the same tensor, so a wrong answer is
    obvious rather than merely improbable.
    """
    if torch.cuda.device_count() < 2:
        return False

    probe = torch.arange(1024, dtype=torch.float32, device="cuda:0")
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    direct = probe.to("cuda:1")
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)

    ok = bool(torch.equal(direct.cpu(), probe.cpu()))
    if verbose:
        print(f"[p2p_workaround] cuda:0 -> cuda:1 copy intact: {ok}")
    return not ok
