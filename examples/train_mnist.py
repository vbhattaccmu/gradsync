"""Data-parallel MNIST across N GPUs / nodes using gradsync.

This is the headline demo: a normal PyTorch training loop where the *only*
distributed-specific lines are `init_process_group`, wrapping the model in
`DistributedDataParallel`, and calling `model.sync_gradients()` after
`loss.backward()`. Each rank trains on a disjoint shard of the data; gradsync
averages the gradients across all ranks every step, so all ranks stay in sync and
you get a linear-ish throughput speedup.

Launch (2 nodes, 1 GPU each) — see README for the matching node-1 command:
    python -m gradsync.launch --nnodes 2 --node-rank 0 --nproc-per-node 1 \
        --master-addr <MASTER_INTERNAL_IP> --master-port 29500 examples/train_mnist.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import gradsync


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main() -> None:
    comm = gradsync.init_process_group()
    device = torch.device(f"cuda:{comm.local_rank}")
    torch.manual_seed(0)  # same init on every rank; DDP also broadcasts to be sure

    if comm.rank == 0:
        topo = gradsync.report_topology()
        print(f"[rank 0] world_size={comm.world_size} nvlink={topo['has_nvlink']}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST("./data", train=True, download=(comm.rank == 0), transform=tfm)

    # Each rank sees a disjoint shard — this is what makes it data-parallel.
    sampler = DistributedSampler(train_set, num_replicas=comm.world_size, rank=comm.rank, shuffle=True)
    loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=2)

    model = Net().to(device)
    model = gradsync.DistributedDataParallel(model, comm)  # broadcasts rank-0 weights
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        t0 = time.time()
        seen = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            model.sync_gradients()  # <-- gradsync AllReduce across all ranks
            opt.step()
            seen += images.size(0)
        dt = time.time() - t0
        if comm.rank == 0:
            print(f"epoch {epoch}: loss={loss.item():.4f}  {seen / dt:,.0f} img/s/rank")


if __name__ == "__main__":
    main()
