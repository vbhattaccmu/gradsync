# gradsync

**Multi-GPU data-parallel training over NCCL — Rust core, Python SDK, one Terraform command to a GPU cluster.**

gradsync is a small, readable, open-source demonstration of how modern
distributed training actually moves gradients between GPUs:

- **Within a node**, GPUs talk over **NVLink / NVSwitch**.
- **Across nodes**, **NCCL** coordinates collectives (AllReduce) over the
  network, using **GPUDirect** to move data GPU-to-GPU without CPU copies.

The *interesting* code lives in an auditable **Rust core** that wraps NCCL/CUDA;
**Python** is a thin ergonomic shell on top (via PyO3). You get a `torchrun`-style
launcher and a `DistributedDataParallel` wrapper, and a `terraform apply` that
stands up a 2-node GPU cluster on GCP.

> **Honest scope:** NVLink/NVSwitch are *hardware* — gradsync does not simulate
> the fabric. It drives **real** NCCL collectives, automatically uses NVLink when
> the box has it, and uses the network across nodes. `report_topology()` +
> `NCCL_DEBUG=INFO` let a run *prove* which path each link took.

## Layout

```
core/       Rust core: rendezvous, collectives, topology  (wraps libnccl/libcudart)
bindings/   PyO3 -> the `_gradsync` extension module
python/     the gradsync Python SDK (init_process_group, DDP, launcher)
examples/   allreduce_test.py, train_mnist.py
infra/      Terraform for an N-node GPU cluster on GCP
docs/       architecture.md
```

See [docs/architecture.md](docs/architecture.md) for the full picture.

## Quickstart

### 1. Develop on a laptop (no GPU)

Everything except the collectives works and is unit-tested without a GPU:

```bash
cargo test -p gradsync-core        # topology parser + id (de)serialization
```

Collectives return a clear `StubBuild` error until built on a GPU host.

### 2. Stand up a GPU cluster on GCP

Needs a GCP project with **GPU quota** (A2/A100 quota is 0 by default — request an
increase first) and billing enabled.

```bash
cd infra/terraform
cp example.tfvars terraform.tfvars   # fill in project_id, ssh_user, ...
terraform init
terraform apply                      # creates 2x A100 VMs on a private VPC
terraform output launch_hint         # prints the exact launch commands
```

Each VM's startup script installs Rust, builds the wheel **with real NCCL**
(`maturin build --features nccl`), and pip-installs gradsync.

### 3. Run the distributed training

Two nodes, one A100 each. Use node 0's **internal** IP as the master address
(`terraform output master_addr`):

```bash
# on node 0
python -m gradsync.launch --nnodes 2 --node-rank 0 --nproc-per-node 1 \
    --master-addr <MASTER_INTERNAL_IP> --master-port 29500 examples/train_mnist.py

# on node 1
python -m gradsync.launch --nnodes 2 --node-rank 1 --nproc-per-node 1 \
    --master-addr <MASTER_INTERNAL_IP> --master-port 29500 examples/train_mnist.py
```

To exercise **NVLink inside a node**, use a 2-GPU machine type (`a2-highgpu-2g`,
`gpu_count = 2`) and one node:

```bash
python -m gradsync.launch --nproc-per-node 2 examples/allreduce_test.py
```

Set `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` to see NCCL log whether each
link used NVLink (`P2P/NVLink`) or the network (`NET`).

> **Cost:** A100 VMs bill ~$3-4/GPU/hr. Run `terraform destroy` the moment you're
> done.

## The only distributed-specific lines in your training loop

```python
import gradsync

comm  = gradsync.init_process_group()
model = gradsync.DistributedDataParallel(model, comm)
...
loss.backward()
model.sync_gradients()   # NCCL AllReduce-average across every GPU/node
optimizer.step()
```

## Status / roadmap

- [x] Rust core: rendezvous, AllReduce/Broadcast, topology reporting
- [x] PyO3 bindings + Python SDK + torchrun-style launcher
- [x] Terraform for an N-node GCP GPU cluster
- [ ] GPUDirect-TCPX flags for A3/H100 cross-node
- [ ] Overlap: bucketed AllReduce during backward (comm/compute overlap)
- [ ] AllGather / ReduceScatter for sharded (ZeRO-style) optimizers
- [ ] Benchmarks: NVLink vs network bandwidth, scaling efficiency

## License

Apache-2.0. See [LICENSE](LICENSE).
