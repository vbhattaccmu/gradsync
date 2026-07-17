"""Smoke test: AllReduce a tensor across every rank and check the math.

Each rank fills a tensor with its own rank number, then sum-AllReduces. After the
collective every rank must hold the same value: sum(0..world_size-1). This proves
the whole path — rendezvous, communicator init, and a real NCCL collective over
NVLink/network — end to end.

Run on a single 2-GPU node:
    python -m gradsync.launch --nproc-per-node 2 examples/allreduce_test.py

Run across two 1-GPU nodes (see README for the two commands).
"""

import torch

import gradsync


def main() -> None:
    comm = gradsync.init_process_group()

    if comm.rank == 0:
        topo = gradsync.report_topology()
        print(f"[rank 0] nccl_enabled={topo['nccl_enabled']} has_nvlink={topo['has_nvlink']}")

    device = torch.device(f"cuda:{comm.local_rank}")
    n = 1_000_000
    # Every rank contributes a tensor full of its own rank id.
    x = torch.full((n,), float(comm.rank), device=device, dtype=torch.float32)

    # Sum across ranks (average=False keeps it a plain sum for an exact check).
    comm.all_reduce_f32(x.data_ptr(), x.numel(), average=False)

    expected = sum(range(comm.world_size))
    got = x[0].item()
    ok = abs(got - expected) < 1e-3
    print(f"[rank {comm.rank}] allreduce sum -> {got:.1f} (expected {expected}) {'OK' if ok else 'FAIL'}")
    assert ok, f"rank {comm.rank}: expected {expected}, got {got}"


if __name__ == "__main__":
    main()
