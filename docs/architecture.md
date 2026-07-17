# Architecture

gradsync is deliberately split so the *interesting* systems code is in auditable
Rust and Python is a thin ergonomic shell. Nothing is torch-specific in the core.

```
┌───────────────────────────────────────────────┐
│  Python SDK  (python/gradsync/)                │  user writes their training loop
│    init_process_group()                        │
│    DistributedDataParallel(model, comm)         │
│    model.sync_gradients()                       │
├───────────────────────────────────────────────┤
│  PyO3 extension  (bindings/) -> _gradsync.so   │  Comm, all_reduce_f32, broadcast_f32
├───────────────────────────────────────────────┤
│  Rust core  (core/)                            │
│    comm.rs        rendezvous + ncclComm_t       │
│    collectives.rs allreduce / broadcast         │
│    topology.rs    parse `nvidia-smi topo -m`    │
│    nccl.rs        raw FFI to <nccl.h>           │
├───────────────────────────────────────────────┤
│  NCCL + CUDA  (libnccl, libcudart)             │
└───────────────────────────────────────────────┘

Infra:  infra/terraform/  ->  N GPU VMs on a private VPC on GCP
```

## The distributed mechanism

Standard synchronous data parallelism:

1. **Rendezvous.** Rank 0 mints a 128-byte NCCL unique id and serves it over TCP
   to every other rank (`python/gradsync/launch.py`). This is the only
   out-of-band step; after it, NCCL builds its own transports.
2. **Communicator init.** Every rank calls `ncclCommInitRank`. NCCL probes the
   fabric and picks the fastest path between each pair of ranks on its own:
   **NVLink/NVSwitch inside a node**, **network (GPUDirect-TCPX / RoCE / IB)
   across nodes**. gradsync does not choose the transport — it *reports* it.
3. **Broadcast weights.** DDP broadcasts rank 0's initial parameters so all ranks
   start identical.
4. **Overlapped per-bucket AllReduce.** Parameters are grouped into ~25 MB
   buckets in reverse order. An autograd hook fires when each parameter's grad is
   ready; when a bucket completes, its fused in-place average-AllReduce is
   enqueued on a dedicated **comm stream** (`ncclGroupStart/End`) *during*
   backward, overlapping with the compute still running on the default stream. No
   host copies — the reduction is GPU-to-GPU. `sync_gradients()` orders the
   optimizer after the comm stream.

## Stream ordering (why it's correct)

The core's collectives are **enqueue-only** — they never synchronize. Ordering is
the caller's job, done with CUDA stream waits so the two streams interleave safely:

```
backward (default/compute stream):  ... grad_k ready ──┐
                                                        │ comm_stream.wait_stream(compute)
comm stream:                          bucket AllReduce ─┘  (runs concurrently with
                                                            remaining backward)
sync_gradients():  compute.wait_stream(comm_stream)   # step waits for all reductions
optimizer.step() (compute stream):                    ──► safe: grads fully reduced
```

`comm_stream.wait_stream(compute)` guarantees a bucket is never reduced before its
gradient kernels finish; `compute.wait_stream(comm_stream)` guarantees weights are
never updated mid-reduction. The Python DDP uses torch's stream primitives for
this, so gradsync doesn't reimplement CUDA events.

## Why "you can't emulate NVLink" matters

NVLink and NVSwitch are physical interconnect. You get them by renting a
multi-GPU box (A100/H100). gradsync's honest claim is: it drives real NCCL
collectives, uses NVLink automatically when the hardware has it, and uses the
network across nodes. `report_topology()` and `NCCL_DEBUG=INFO` let a run *prove*
which path each link took, rather than pretending to simulate the fabric.

## Build modes

| Mode | Command | Collectives | Where |
|------|---------|-------------|-------|
| stub | `cargo build` | return `StubBuild` error | any laptop |
| real | `maturin build --features nccl` | run on NCCL | GPU host w/ CUDA+NCCL |

Stub mode exists so the SDK, launcher, rendezvous, and topology parser can be
developed and unit-tested without a GPU. The collectives light up on a GPU host.
