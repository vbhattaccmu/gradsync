"""Rendezvous + a small multi-process launcher.

Two responsibilities:

1. ``exchange_unique_id`` — a tiny TCP handshake so rank 0 can hand its 128-byte
   NCCL unique id to every other rank. NCCL needs this out-of-band; once every
   rank has it, NCCL sets up its own (much faster) transports for the actual
   data. We use plain sockets to keep the dependency surface at zero.

2. ``python -m gradsync.launch`` — spawn one process per local GPU on this node,
   setting the GRADSYNC_* env vars each process reads in ``init_process_group``.

Multi-node usage (2 nodes, 1 GPU each) mirrors torchrun:

    # on node 0 (the master):
    python -m gradsync.launch --nnodes 2 --node-rank 0 --nproc-per-node 1 \
        --master-addr 10.128.0.2 --master-port 29500 examples/train_mnist.py

    # on node 1:
    python -m gradsync.launch --nnodes 2 --node-rank 1 --nproc-per-node 1 \
        --master-addr 10.128.0.2 --master-port 29500 examples/train_mnist.py

The Terraform in infra/ wires the two VMs and their private IPs so MASTER_ADDR is
just the master's internal address.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import subprocess
import sys
import time

_ID_BYTES = 128


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed during rendezvous")
        buf += chunk
    return buf


def exchange_unique_id(
    unique_id: bytes | None,
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    timeout_s: float = 60.0,
) -> bytes:
    """Distribute rank 0's NCCL unique id to all ranks over TCP.

    Rank 0 binds and serves the id to (world_size - 1) clients. Every other rank
    connects and reads exactly 128 bytes. Returns the id bytes on every rank.
    """
    if rank == 0:
        assert unique_id is not None and len(unique_id) == _ID_BYTES
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", master_port))
        srv.listen(world_size)
        srv.settimeout(timeout_s)
        served = 0
        while served < world_size - 1:
            conn, _ = srv.accept()
            with conn:
                conn.sendall(struct.pack("!I", _ID_BYTES))
                conn.sendall(unique_id)
            served += 1
        srv.close()
        return unique_id

    # Non-zero ranks: retry-connect until the master is up.
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((master_addr, master_port), timeout=5.0) as c:
                (n,) = struct.unpack("!I", _recv_exact(c, 4))
                return _recv_exact(c, n)
        except OSError as e:  # master not ready yet
            last_err = e
            time.sleep(0.5)
    raise ConnectionError(f"rank {rank} could not reach master {master_addr}:{master_port}: {last_err}")


def _main() -> int:
    ap = argparse.ArgumentParser(description="gradsync multi-process launcher")
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--node-rank", type=int, default=0, help="0-based index of this node")
    ap.add_argument("--nproc-per-node", type=int, default=1, help="processes (GPUs) on this node")
    ap.add_argument("--master-addr", default="127.0.0.1")
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("script", help="training script to run")
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    world_size = args.nnodes * args.nproc_per_node
    procs = []
    for local_rank in range(args.nproc_per_node):
        global_rank = args.node_rank * args.nproc_per_node + local_rank
        env = dict(os.environ)
        env.update(
            GRADSYNC_RANK=str(global_rank),
            GRADSYNC_WORLD_SIZE=str(world_size),
            GRADSYNC_LOCAL_RANK=str(local_rank),
            GRADSYNC_MASTER_ADDR=args.master_addr,
            GRADSYNC_MASTER_PORT=str(args.master_port),
            # Ask NCCL to log the transport it picks, so runs record NVLink vs NET.
            NCCL_DEBUG=env.get("NCCL_DEBUG", "WARN"),
        )
        p = subprocess.Popen([sys.executable, args.script, *args.script_args], env=env)
        procs.append(p)

    code = 0
    for p in procs:
        p.wait()
        code = code or p.returncode
    return code


if __name__ == "__main__":
    raise SystemExit(_main())
