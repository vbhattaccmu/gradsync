"""The user-facing SDK: process-group setup, a DDP wrapper, and topology info.

This module is intentionally readable Python. It leans on the compiled
`gradsync._gradsync` extension (built from the Rust core) for the actual NCCL
calls, and on the small TCP rendezvous in ``launch.py`` to distribute the NCCL
unique id from rank 0 to every other rank.

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

from . import _gradsync  # compiled Rust extension
from .launch import exchange_unique_id


@dataclass
class Comm:
    """A thin handle bundling the NCCL communicator with rank metadata."""

    _raw: "_gradsync.Comm"
    rank: int
    world_size: int
    local_rank: int

    def all_reduce_f32(self, ptr: int, count: int, average: bool = True) -> None:
        self._raw.all_reduce_f32(ptr, count, average)

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

    if rank == 0:
        unique_id = _gradsync.generate_unique_id()
    else:
        unique_id = None

    # Rank 0 publishes the id; everyone else receives it. See launch.py.
    unique_id = exchange_unique_id(
        unique_id, rank, world_size, master_addr, master_port
    )

    raw = _gradsync.Comm(unique_id, world_size, rank, local_rank)
    return Comm(_raw=raw, rank=rank, world_size=world_size, local_rank=local_rank)


def all_reduce_grads(comm: Comm, parameters) -> None:
    """Average the ``.grad`` of every parameter across all ranks, in place.

    Operates directly on each gradient tensor's device pointer, so nothing is
    copied to host memory — the reduction runs GPU-to-GPU over whatever transport
    NCCL picked (NVLink within a node, network across nodes).
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
    """Minimal DDP: broadcast initial weights, average grads after backward.

    Deliberately explicit rather than clever — you call ``sync_gradients()``
    yourself after ``loss.backward()``. This keeps the data-parallel mechanism
    visible, which is the point of the project.

        model = DistributedDataParallel(model, comm)
        ...
        loss.backward()
        model.sync_gradients()
        optimizer.step()
    """

    def __init__(self, module, comm: Comm):
        self.module = module
        self.comm = comm
        self._broadcast_parameters()

    def _broadcast_parameters(self) -> None:
        # Make every rank start from rank 0's weights.
        for p in self.module.parameters():
            data = p.data
            if not data.is_contiguous():
                data = data.contiguous()
                p.data = data
            self.comm.broadcast_f32(data.data_ptr(), data.numel(), root=0)

    def sync_gradients(self) -> None:
        all_reduce_grads(self.comm, self.module.parameters())

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()


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
