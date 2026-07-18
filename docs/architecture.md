# Architecture

gradsync layers systems code (Rust) over an ergonomic Python shell. Nothing is torch-specific.

```
┌───────────────────────────────────────────────┐
│  Python SDK  (python/gradsync/)                │
│    init_process_group()                        │
│    DistributedDataParallel(model, comm)        │
│    model.sync_gradients()                      │
├───────────────────────────────────────────────┤
│  PyO3 extension  (bindings/) -> _gradsync.so   │
├───────────────────────────────────────────────┤
│  Rust core  (core/)                            │
│    • comm.rs        rendezvous + NCCL          │
│    • collectives.rs AllReduce / Broadcast      │
│    • topology.rs    hardware detection         │
│    • nccl.rs        FFI to <nccl.h>            │
├───────────────────────────────────────────────┤
│  NCCL + CUDA  (GPU compute + communication)   │
└───────────────────────────────────────────────┘
```

## Distributed mechanism

1. **Rendezvous** — Rank 0 creates NCCL unique id, broadcasts over TCP
2. **Communicator init** — NCCL probes fabric, picks NVLink (inside node) or network (across nodes)
3. **Broadcast weights** — All ranks sync initial parameters
4. **Overlapped AllReduce** — Parameters bucketed (~25 MB). Autograd hooks fire during backward, enqueue AllReduce on comm stream while compute continues. No host copies.

## Stream ordering

Correctness relies on CUDA stream waits:

```
backward (default stream):      ... grad_k ──┐
                                             │ comm_stream.wait_stream(compute)
comm stream:                  AllReduce ─────┘  (runs concurrently)

sync_gradients():  compute.wait_stream(comm_stream)
optimizer.step():                         ──► safe: grads fully reduced
```

- `comm_stream.wait_stream(compute)` → AllReduce doesn't start early
- `compute.wait_stream(comm_stream)` → Optimizer doesn't start early

## Build modes

| Mode | Command | Collectives work? | Location |
|------|---------|-------------------|----------|
| Stub | `cargo build` | No (error) | Laptop (no GPU) |
| Real | `maturin build --features nccl` | Yes | GPU host (CUDA + NCCL) |

Stub mode lets you develop rendezvous, topology parsing, and Python SDK without a GPU. Collectives light up on GPU.

## Why we don't emulate NVLink

NVLink is physical hardware. We don't pretend to simulate it — we use the real thing when available. `report_topology()` and `NCCL_DEBUG=INFO` prove which path (NVLink vs network) each collective took.
