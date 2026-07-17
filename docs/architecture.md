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
4. **Per-step AllReduce.** After `loss.backward()`, `sync_gradients()` runs an
   in-place average-AllReduce over each gradient tensor's *device pointer*. No
   host copies — the reduction is GPU-to-GPU.

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
