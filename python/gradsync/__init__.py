"""gradsync — multi-GPU data-parallel training over NCCL.

A Rust core (wrapping NCCL/CUDA) does the collectives; this package is the
ergonomic Python shell on top. The typical entry points are:

    from gradsync import init_process_group, DistributedDataParallel

    comm = init_process_group()                 # reads env from the launcher
    model = DistributedDataParallel(model, comm)
    ...                                          # your normal training loop
    # gradients are all-reduced across every GPU/node automatically

See examples/train_mnist.py for a full run and README.md for how to launch it
across two GCP nodes with Terraform.
"""

from .trainer import (
    init_process_group,
    DistributedDataParallel,
    all_reduce_grads,
    report_topology,
    Comm,
)

__all__ = [
    "init_process_group",
    "DistributedDataParallel",
    "all_reduce_grads",
    "report_topology",
    "Comm",
]

__version__ = "0.1.0"
