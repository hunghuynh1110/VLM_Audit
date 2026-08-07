# Bunya queue wait evidence

Log of observed SLURM queue wait times for VLM Audit jobs on UQ Bunya
(`gpu_cuda` partition, `gpu` QoS, account `a_ai_collab`).

The smoke test is the smallest possible workload — 1× L40 GPU, 4 CPUs, 40 GB
RAM, 5 inferences on the 11B model. Even this minimal job sits in the queue
for hours.

---

## 2026-05-05 — smoke test

**Submitted:** 2026-05-05 ~15:06 AEST
**Estimated start:** 2026-05-05 19:52:41 AEST
**Estimated wait:** ~4 h 46 min before the job even starts running
**Job:** `24230930` / `vlm_smoke` (1× L40, 4 CPU, 40 GB, 2 h walltime)

Captured `squeue` output:

```
$ squeue --start -u s4938484
             JOBID PARTITION     NAME     USER ST          START_TIME  NODES SCHEDNODES           NODELIST(REASON)
          24230930  gpu_cuda vlm_smok s4938484 PD 2026-05-05T19:52:41      1 bun082               (Priority)
```

Reason: `(Priority)` — waiting behind higher-priority jobs in the fair-share
queue, not blocked by resource shortage. Allocated node `bun082` is reserved
for this job at the start time.

**Implication:** every iteration of the dev loop (debug → fix → test) costs
~5 hours of queue wait on top of any actual compute time. This dominates the
wall-clock cost of getting the pipeline to a working state, far more than the
actual GPU time consumed.
