#!/usr/bin/env bash
# Runs once on each VM at boot. The Deep Learning VM image already provides the
# NVIDIA driver, CUDA, NCCL, and PyTorch, so all we do is fetch gradsync, build
# the Rust wheel with the `nccl` feature, and install it.
#
# Templated vars: node_rank=${node_rank}, node_count=${node_count}.
set -euxo pipefail

echo "gradsync startup: node ${node_rank} of ${node_count}" | logger -t gradsync

# Wait for the DLVM's driver install to settle.
for i in $(seq 1 60); do
  if nvidia-smi >/dev/null 2>&1; then break; fi
  sleep 10
done
nvidia-smi || echo "WARN: nvidia-smi not ready" | logger -t gradsync

export HOME=/root
export CARGO_HOME=/opt/rust
export RUSTUP_HOME=/opt/rustup
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"

# Rust toolchain (needed to build the core + PyO3 bindings on-host).
curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
export PATH="$CARGO_HOME/bin:$PATH"

python3 -m pip install --upgrade pip maturin torch torchvision

# Fetch gradsync. Replace with your fork/branch as needed.
cd /opt
if [ ! -d gradsync ]; then
  git clone https://github.com/vbhattaccmu/gradsync.git
fi
cd gradsync/bindings

# Build + install the wheel WITH real NCCL collectives.
export CUDA_HOME=/usr/local/cuda
maturin build --release --features nccl -o /tmp/wheels
python3 -m pip install /tmp/wheels/gradsync-*.whl

echo "gradsync ready on node ${node_rank}" | logger -t gradsync
