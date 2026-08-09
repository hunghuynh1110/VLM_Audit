"""
Print the actual values a cross-GPU copy produces.

"100% of elements differ" has several very different explanations, and the raw
numbers separate them:
  - destination reads as uninitialised garbage / zeros  -> copy never landed
  - destination holds plausible but unrelated values    -> reading wrong memory
  - destination is a shifted or strided version of src  -> offset/stride bug
  - destination matches a KNOWN pattern written to dev j -> peer mapping aliased

The source here is a known ramp (0,1,2,...), not random, so any structure in the
result is visible immediately. Device j is pre-filled with a distinctive
sentinel so "the copy did nothing" is distinguishable from "the copy wrote
something wrong".
"""

from __future__ import annotations

import torch


def sync_all() -> None:
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def main() -> None:
    n = torch.cuda.device_count()
    print(f"torch={torch.__version__} devices={n}\n")

    N = 16
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            src = torch.arange(N, dtype=torch.float32, device=f"cuda:{i}")
            sentinel = torch.full((N,), -999.0, dtype=torch.float32, device=f"cuda:{j}")
            sync_all()

            dst = torch.empty(N, dtype=torch.float32, device=f"cuda:{j}")
            dst.copy_(src)
            sync_all()

            print(f"gpu{i} -> gpu{j}")
            print(f"   src (on {i})      : {src.cpu().tolist()}")
            print(f"   dst (on {j})      : {dst.cpu().tolist()}")
            print(f"   sentinel intact?  : {sentinel.cpu().tolist()[:4]} ...")
            print(f"   equal             : {bool(torch.equal(src.cpu(), dst.cpu()))}")

            # .to() path, which is what accelerate's hooks use
            viato = src.to(f"cuda:{j}")
            sync_all()
            print(f"   src.to(cuda:{j})   : {viato.cpu().tolist()}")
            print()

    # Does a second read change the answer? (transient vs persistent corruption)
    print("=" * 60)
    print("re-read stability on a large buffer")
    src = torch.arange(1 << 20, dtype=torch.float32, device="cuda:0")
    sync_all()
    d1 = src.to("cuda:1")
    sync_all()
    a = d1.cpu()
    b = d1.cpu()
    print(f"   two reads of the same dst agree: {bool(torch.equal(a, b))}")
    print(f"   dst matches src                : {bool(torch.equal(a, src.cpu()))}")
    print(f"   first 8 of src : {src.cpu()[:8].tolist()}")
    print(f"   first 8 of dst : {a[:8].tolist()}")
    print(f"   n_zero in dst  : {int((a == 0).sum())} / {a.numel()}")
    print(f"   dst min/max    : {a.min().item()} {a.max().item()}")


if __name__ == "__main__":
    main()
