"""
Does a cross-GPU tensor copy on this node actually move the data?

The sharded-11B trace showed two things that cannot both be true of a working
machine: layer 0's output differed from the identical single-GPU forward, and
NaNs produced on cuda:0 were absent from the tensor read on cuda:1. Values do
not un-corrupt themselves, so the suspicion is that device-to-device copies are
returning something other than the source bytes.

This checks that directly, with no model involved:
  1. per-device compute sanity (a copy that works is useless if the maths does not)
  2. round-trip integrity  i -> j -> i, exact equality expected
  3. one-way integrity     i -> j, compared against a CPU staging copy
  4. the same with P2P access explicitly disabled, to see whether peer access
     is the difference

Run under scripts/bunya_diagnose_p2p.sh.
"""

from __future__ import annotations

import os
import sys

import torch


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def sync_all() -> None:
    """
    Synchronise EVERY device, not just the current one.

    torch.cuda.synchronize() with no argument syncs only the current device
    (device 0 here). A copy between devices 1 and 2 is not covered by that, so
    comparing straight after it can read the destination before the copy lands
    and report corruption that is really just a race in the test.
    """
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def main() -> None:
    n = torch.cuda.device_count()
    print(f"torch={torch.__version__}  devices={n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}] {p.name}  {p.total_memory // 2**30} GB")
    print(f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    banner("1. peer access matrix (can_device_access_peer)")
    for i in range(n):
        row = []
        for j in range(n):
            row.append("-" if i == j else str(torch.cuda.can_device_access_peer(i, j))[0])
        print(f"  gpu{i}: {' '.join(row)}")

    banner("2. per-device compute sanity")
    for i in range(n):
        d = f"cuda:{i}"
        a = torch.randn(2048, 2048, device=d)
        got = (a @ a.t()).float()
        ref = (a.cpu() @ a.cpu().t())
        err = (got.cpu() - ref).abs().max().item()
        print(f"  gpu{i}: matmul max_abs_err={err:.3e}  finite={bool(torch.isfinite(got).all())}")

    banner("3. round trip  i -> j -> i  (must be bit-exact)")
    sizes = [1 << 10, 1 << 16, 1 << 20, 1 << 24]
    bad = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for numel in sizes:
                src = torch.randn(numel, device=f"cuda:{i}")
                ref = src.cpu().clone()          # ground truth taken before any peer copy
                sync_all()
                back = src.to(f"cuda:{j}").to(f"cuda:{i}")
                sync_all()
                ok = bool(torch.equal(ref, back.cpu()))
                if not ok:
                    bad += 1
                    b = back.cpu()
                    diff = (ref - b).abs()
                    print(f"  MISMATCH gpu{i}->gpu{j}->gpu{i} numel={numel:>9}: "
                          f"n_diff={int((ref != b).sum())} max={diff.max().item():.4g} "
                          f"nan_in_back={int(torch.isnan(b).sum())}")
    print(f"  round-trip mismatches: {bad}")

    banner("4. one-way  i -> j  vs CPU staging (must be bit-exact)")
    bad2 = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for numel in sizes:
                src = torch.randn(numel, device=f"cuda:{i}")
                sync_all()
                direct = src.to(f"cuda:{j}")
                staged = src.cpu().to(f"cuda:{j}")
                sync_all()
                if not torch.equal(direct, staged):
                    bad2 += 1
                    print(f"  MISMATCH gpu{i}->gpu{j} numel={numel:>9}: "
                          f"n_diff={int((direct != staged).sum())} "
                          f"nan_direct={int(torch.isnan(direct).sum())} "
                          f"nan_staged={int(torch.isnan(staged).sum())}")
    print(f"  one-way mismatches: {bad2}")

    banner("5. non-contiguous / narrow copies (what accelerate actually does)")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            base = torch.randn(4096, 128, device=f"cuda:{i}")
            view = base[:, 8:72]                      # non-contiguous slice
            sync_all()
            direct = view.to(f"cuda:{j}")
            staged = view.cpu().to(f"cuda:{j}")
            sync_all()
            ok = bool(torch.equal(direct, staged))
            print(f"  gpu{i}->gpu{j} narrow view: {'OK' if ok else 'MISMATCH'}")

    print(f"\n[p2p] total mismatches: {bad + bad2}")
    sys.exit(1 if (bad or bad2) else 0)


if __name__ == "__main__":
    main()
