"""The user-facing SDK: process-group setup, an overlapping DDP, topology info.

This module is intentionally readable Python. It leans on the compiled
`gradsync._gradsync` extension (built from the Rust core) for the actual NCCL
calls, and on the small TCP rendezvous in ``launch.py`` to distribute the NCCL
unique id from rank 0 to every other rank.

The DDP here does **communication/compute overlap** — the technique that makes
data-parallel training actually scale:

  * Parameters are grouped into fixed-size *buckets* (roughly in reverse order,
    which is the order backward produces gradients).
  * An autograd hook fires the moment each parameter's gradient is ready. When a
    whole bucket is ready, its AllReduce is launched immediately on a dedicated
    CUDA *comm stream* — while the rest of the backward pass is still computing on
    the default stream.
  * By the time ``backward()`` returns, most reductions are already done or in
    flight. ``sync_gradients()`` just waits for the comm stream before the
    optimizer step.

Naive DDP (reduce everything after a full backward) is available via
``overlap=False`` or on CPU, for comparison.

Rendezvous environment (set by ``python -m gradsync.launch`` or your scheduler):

    GRADSYNC_RANK          global rank of this process        (0..world_size-1)
    GRADSYNC_WORLD_SIZE    total number of ranks
    GRADSYNC_LOCAL_RANK    GPU ordinal to bind on this host
    GRADSYNC_MASTER_ADDR   host of rank 0
    GRADSYNC_MASTER_PORT   TCP port rank 0 listens on for id exchange
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

# Optional import of the compiled Rust extension. In stub mode (no NCCL),
# this will fail gracefully and return None. Actual AllReduce calls will error,
# but pure Python logic (bucketing, overlap planning) works for testing.
try:
    from . import _gradsync
except ImportError:
    _gradsync = None  # type: ignore

from .launch import exchange_unique_id

# f32 elements per MB, used to turn a bucket size in MB into an element count.
_ELEMS_PER_MB = 1024 * 1024 // 4


@dataclass
class Comm:
    """A thin handle bundling the NCCL communicator with rank metadata."""

    _raw: "_gradsync.Comm"
    rank: int
    world_size: int
    local_rank: int

    def all_reduce_f32(self, ptr: int, count: int, average: bool = True) -> None:
        self._raw.all_reduce_f32(ptr, count, average)

    def all_reduce_bucket_f32(
        self, bufs: Sequence[Tuple[int, int]], average: bool = True, stream: int = 0
    ) -> None:
        self._raw.all_reduce_bucket_f32(list(bufs), average, stream)

    def sync_stream(self, stream: int = 0) -> None:
        self._raw.sync_stream(stream)

    def broadcast_f32(self, ptr: int, count: int, root: int = 0) -> None:
        self._raw.broadcast_f32(ptr, count, root)


def _env_int(name: str, default: int | None = None) -> int:
    v = os.environ.get(name)
    if v is None:
        if default is None:
            raise RuntimeError(f"{name} is not set; launch with `python -m gradsync.launch`")
        return default
    return int(v)


def init_process_group() -> Comm:
    """Join the collective group. Call once per process at startup.

    Rank 0 mints the NCCL unique id and serves it over TCP; every rank then
    initializes its communicator bound to GRADSYNC_LOCAL_RANK.
    """
    rank = _env_int("GRADSYNC_RANK")
    world_size = _env_int("GRADSYNC_WORLD_SIZE")
    local_rank = _env_int("GRADSYNC_LOCAL_RANK", 0)
    master_addr = os.environ.get("GRADSYNC_MASTER_ADDR", "127.0.0.1")
    master_port = _env_int("GRADSYNC_MASTER_PORT", 29500)

    unique_id = _gradsync.generate_unique_id() if rank == 0 else None
    # Rank 0 publishes the id; everyone else receives it. See launch.py.
    unique_id = exchange_unique_id(unique_id, rank, world_size, master_addr, master_port)

    raw = _gradsync.Comm(unique_id, world_size, rank, local_rank)
    return Comm(_raw=raw, rank=rank, world_size=world_size, local_rank=local_rank)


def plan_buckets(sizes: Sequence[int], cap_elems: int) -> List[List[int]]:
    """Group parameter indices into buckets, each up to ``cap_elems`` elements.

    Pure and side-effect free so it can be unit-tested without a GPU. ``sizes[i]``
    is the element count of parameter ``i``; the returned buckets hold *indices*
    into ``sizes``. A single parameter larger than the cap gets its own bucket.
    Greedy fill in the given order (the caller passes params in reverse, matching
    backward order, so a bucket completes as soon as backward reaches its start).
    """
    buckets: List[List[int]] = []
    cur: List[int] = []
    cur_n = 0
    for i, n in enumerate(sizes):
        if cur and cur_n + n > cap_elems:
            buckets.append(cur)
            cur, cur_n = [], 0
        cur.append(i)
        cur_n += n
    if cur:
        buckets.append(cur)
    return buckets


def all_reduce_grads(comm: Comm, parameters) -> None:
    """Average the ``.grad`` of every parameter across all ranks, in place.

    The naive path: one synchronous AllReduce per parameter, after a full
    backward. Simple and correct, but comm never overlaps compute. Used as the
    fallback and as a baseline to compare the overlapped path against.
    """
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad
        if not g.is_contiguous():
            g = g.contiguous()
            p.grad = g
        comm.all_reduce_f32(g.data_ptr(), g.numel(), average=True)


class DistributedDataParallel:
    """Data-parallel wrapper with communication/compute overlap.

        model = DistributedDataParallel(model, comm)   # broadcasts rank-0 weights
        ...
        loss.backward()          # bucket AllReduces fire during this call
        model.sync_gradients()   # wait for the comm stream, then step
        optimizer.step()

    Set ``overlap=False`` (or run on CPU) to get the naive reduce-after-backward
    behavior for comparison.

    Assumption: every parameter that requires grad receives a gradient each step
    (true for the MNIST example). Handling truly unused parameters needs the extra
    "which params were touched" negotiation that full frameworks do; it is called
    out in the roadmap rather than implemented here.
    """

    def __init__(self, module, comm: Comm, bucket_mb: float = 25.0, overlap: bool = True):
        self.module = module
        self.comm = comm

        # torch is only needed for the GPU DDP machinery; import lazily so the
        # rest of the SDK (rendezvous, topology) has no hard torch dependency.
        import torch

        self._torch = torch
        self.overlap = overlap and torch.cuda.is_available()

        self._broadcast_parameters()

        if self.overlap:
            self._params = [p for p in self.module.parameters() if p.requires_grad]
            # Reverse order approximates the order backward yields gradients, so a
            # bucket becomes ready as a contiguous run of the backward pass finishes.
            self._params.reverse()
            sizes = [p.numel() for p in self._params]
            cap = max(1, int(bucket_mb * _ELEMS_PER_MB))
            self._buckets = plan_buckets(sizes, cap)
            self._bucket_of = {}
            for b_idx, bucket in enumerate(self._buckets):
                for p_idx in bucket:
                    self._bucket_of[p_idx] = b_idx
            self._comm_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._reset_step_state()
            self._register_hooks()

    # -- setup -------------------------------------------------------------

    def _broadcast_parameters(self) -> None:
        # Make every rank start from rank 0's weights.
        for p in self.module.parameters():
            data = p.data
            if not data.is_contiguous():
                data = data.contiguous()
                p.data = data
            self.comm.broadcast_f32(data.data_ptr(), data.numel(), root=0)

    def _register_hooks(self) -> None:
        # Fires after each parameter's gradient has been fully accumulated.
        for p_idx, p in enumerate(self._params):
            p.register_post_accumulate_grad_hook(self._make_hook(p_idx))

    def _make_hook(self, p_idx: int):
        def hook(_param):
            self._on_grad_ready(p_idx)

        return hook

    def _reset_step_state(self) -> None:
        self._ready_counts = [0] * len(self._buckets)
        self._reduced = [False] * len(self._buckets)

    # -- overlap engine ----------------------------------------------------

    def _on_grad_ready(self, p_idx: int) -> None:
        b_idx = self._bucket_of[p_idx]
        self._ready_counts[b_idx] += 1
        if self._ready_counts[b_idx] == len(self._buckets[b_idx]) and not self._reduced[b_idx]:
            self._reduce_bucket(b_idx)

    def _reduce_bucket(self, b_idx: int) -> None:
        torch = self._torch
        self._reduced[b_idx] = True

        # The gradients were produced on the current (compute) stream; make the
        # comm stream wait for them before it starts the reduction.
        self._comm_stream.wait_stream(torch.cuda.current_stream())

        bufs: List[Tuple[int, int]] = []
        with torch.cuda.stream(self._comm_stream):
            for p_idx in self._buckets[b_idx]:
                p = self._params[p_idx]
                if p.grad is None:
                    continue
                g = p.grad
                if not g.is_contiguous():
                    g = p.grad = g.contiguous()
                bufs.append((g.data_ptr(), g.numel()))
            if bufs:
                # Fused, in-place, average AllReduce for the whole bucket, enqueued
                # on the comm stream (async — returns while it runs).
                self.comm.all_reduce_bucket_f32(bufs, average=True, stream=self._comm_stream.cuda_stream)

    def sync_gradients(self) -> None:
        """Finalize the step: ensure every bucket is reduced, then order the
        optimizer after the comm stream. Call once after ``loss.backward()``."""
        if not self.overlap:
            all_reduce_grads(self.comm, self.module.parameters())
            return

        torch = self._torch
        # Reduce any bucket that never completed on its own (e.g. a run where a
        # bucket got some but not all of its grads); reduces only present grads.
        for b_idx in range(len(self._buckets)):
            if not self._reduced[b_idx] and self._ready_counts[b_idx] > 0:
                self._reduce_bucket(b_idx)

        # The optimizer runs on the current stream; make it wait for all the
        # reductions enqueued on the comm stream to finish first.
        torch.cuda.current_stream().wait_stream(self._comm_stream)
        self._reset_step_state()

    # -- passthrough -------------------------------------------------------

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self, *a, **k):
        return self.module.state_dict(*a, **k)


def report_topology() -> dict:
    """Return what NCCL/hardware this host offers, for logging at startup.

    Reads ``nvidia-smi topo -m`` and asks the Rust core whether any GPU pair is
    on NVLink. Handy to print on rank 0 so a run's logs record whether intra-node
    traffic used NVLink and whether the build can do real collectives.
    """
    info = {"nccl_enabled": _gradsync.nccl_enabled(), "has_nvlink": False, "topo_raw": ""}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL
        )
        info["topo_raw"] = out
        info["has_nvlink"] = _gradsync.topo_has_nvlink(out)
    except (OSError, subprocess.CalledProcessError):
        # No nvidia-smi (e.g. laptop dev): leave defaults.
        pass
    return info
