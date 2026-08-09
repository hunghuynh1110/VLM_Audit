# Draft email to UQ RCC — silent GPU-to-GPU copy corruption

Send to: `rcc-support@uq.edu.au` (check the current address on the RCC support page)

---

**Subject:** Silent data corruption on multi-GPU jobs — cross-GPU tensor copies return zeros (gpu_cuda)

Hi RCC team,

I think I've hit a node-level fault on Bunya's `gpu_cuda` partition that silently
corrupts any multi-GPU PyTorch job. It produces no error and no warning, so jobs
complete "successfully" with meaningless output. I wanted to report it in case it
affects other users.

**Symptom**

Any GPU-to-GPU tensor copy returns zeros instead of the data. Minimal reproduction,
plain PyTorch with no other libraries:

```python
import torch
a = torch.arange(8, device="cuda:0")
b = a.to("cuda:1")
print("src:", a.tolist())   # [0, 1, 2, 3, 4, 5, 6, 7]
print("dst:", b.tolist())   # [0, 0, 0, 0, 0, 0, 0, 0]
```

The destination buffer reads as all zeros. This happens in every direction between
every pair of visible GPUs. `torch.cuda.can_device_access_peer()` returns `True` for
all pairs, so PyTorch takes the peer-to-peer path and the transfer is silently lost.

**Scope of testing**

- Reproduces on both **A100** and **H100** nodes in `gpu_cuda`
- Reproduces on **torch 2.7.1+cu126, 2.8.0+cu128 and 2.11.0+cu130** (three separate
  virtualenvs), so it is not a PyTorch version issue
- Confirmed not a synchronisation artefact — all devices explicitly synchronised
  before reading the destination
- **Host↔device copies are correct**; only device→device fails
- `NCCL_P2P_DISABLE=1` has no effect (it governs NCCL collectives, not `Tensor.to()`)

**Suspected cause**

This matches the known PCIe peer-to-peer failure mode where IOMMU or PCIe ACS
(Access Control Services) is enabled: `cudaDeviceCanAccessPeer()` still reports
`True`, but P2P DMA silently fails. NVIDIA documents that ACS must be disabled for
P2P transfers to work correctly on PCIe systems. Could you confirm the ACS/IOMMU
configuration and P2P status on the `gpu_cuda` GPU nodes?

**Impact**

Any job using `device_map="auto"`, HuggingFace `accelerate`, model parallelism, or
manual `.to(cuda:N)` across devices will silently produce wrong results. In my case
a 90B vision-language model returned a perfectly uniform probability distribution
(exactly 1/vocab_size for every token) because activations crossing a device
boundary arrived as zeros. The job exited 0 with no warnings, and the output looked
superficially plausible.

**Relevant job IDs** (account `a_ai_collab`, user `s4938484`)

| job | what |
|---|---|
| 27105190, 27105703 | 3×H100 production runs that produced corrupted output |
| 27113604 | reproduction on a smaller model, 3×A100 |
| 27113643, 27113672 | cross-version P2P copy tests |
| 27113674 | validation that host-staged copies are correct |

**Two questions**

1. Can the P2P/ACS configuration on the `gpu_cuda` nodes be checked and fixed?
2. In the meantime, could I be granted the **`sxm` QOS** for the `gpu_sxm`
   partition? Those nodes are NVLink-connected rather than PCIe, so they likely use
   a different transfer path and may be unaffected. My account currently has
   `debug, gpu, mig, normal, short, viz` but not `sxm`.

I have a workaround in place (routing cross-GPU copies through host memory), so I'm
not blocked — but it costs performance, and I suspect other users are silently
affected without knowing.

Happy to provide any further diagnostics.

Thanks,
Gia Hung Huynh
Student number 49384848 · `s4938484`
REIT4841 Honours Thesis, supervised by Prof Gianluca Demartini
