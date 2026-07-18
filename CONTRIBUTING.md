# Contributing to gradsync

Thanks for your interest in improving gradsync! This project prioritizes **readability and correctness** over feature count, so contributions that clarify the code or add rigorous tests are especially welcome.

## Development setup

### Prerequisites
- Rust 1.70+ (`rustup` — https://rustup.rs/)
- Python 3.9+
- For GPU work: CUDA 12.1+, NCCL 2.18+, and a compatible GPU

### Build (no GPU required)

```bash
git clone https://github.com/vbhattaccmu/gradsync.git
cd gradsync

# Rust core (stub mode, no NCCL needed)
cargo build -p gradsync-core
cargo test -p gradsync-core
cargo clippy --workspace
cargo fmt --all

# Python SDK (unit tests only)
python -m pytest python/tests/ -v
```

### Build with GPU

```bash
# On a GPU host with CUDA + NCCL installed:
maturin build --release --features nccl -o wheels
pip install wheels/gradsync-*.whl

# Test the bindings
python -c "from gradsync import init_process_group; print('NCCL bindings OK')"
```

## Code style

- **Rust:** `cargo fmt` + `cargo clippy` must pass. No unsafe except FFI.
- **Python:** Black style (enforced by CI). Docstrings on public functions.
- **Commits:** One logical change per commit. Title 50 chars; body 72 chars. Prefix with area: "Rust:", "Python:", "CI:", "Docs:", "Infra:".

Example:
```
Rust: add group_all_reduce for bucket fusion

Previously, AllReducing each tensor separately meant one collective
launch per tensor. With ncclGroupStart/End, fuse a whole bucket
into one launch and save per-collective latency.

Adds group_all_reduce() in collectives.rs; bindings layer gets
all_reduce_bucket_f32() that takes a Vec of (ptr, count) pairs.
```

## Roadmap priorities

### High impact (core functionality)

1. **Benchmarking harness** — Automated overlap measurement:
   - Time step() with `overlap=True` vs `overlap=False`
   - Plot scaling efficiency (throughput / world_size)
   - Report NVLink vs network utilization
   - CI nightly runs to catch regressions

2. **`find_unused_parameters` support** — For conditional models:
   - Autograd hook on every parameter
   - Mark used / unused, communicate bitmap across ranks
   - AllReduce only the used parameters
   - Guard against deadlock if ranks disagree on usage

3. **AllGather / ReduceScatter** — Foundation for ZeRO:
   - Enqueue-only (no sync) like AllReduce
   - Fused group ops like group_all_reduce
   - Python SDK wrapper (more complex API)

### Medium impact (convenience)

4. **Mixed-precision (AMP) support** — Gradient scaling:
   - Pass `loss_scale` to AllReduce
   - Scale grads down before reduce, up after
   - Integrate with PyTorch's `autocast` / `GradScaler`

5. **Better diagnostics & monitoring**:
   - Trace collection (which bucket ran, how long)
   - Bandwidth saturation detection
   - Breakdown of time: compute vs comm vs overhead

6. **GPUDirect-TCPX optimizations** — For A3/H100:
   - Detect hardware
   - Pass special NCCL flags
   - Benchmark vs standard TCP

### Lower priority (specialized)

7. **Gradient compression** — For slow networks
8. **Pipeline parallelism** — (out of scope, but users ask)
9. **Async All-Reduce** — (don't block in sync_gradients())

## Testing

### Rust (unit tests)

```bash
# Topology parsing, id serialization (GPU-free, runs everywhere)
cargo test -p gradsync-core
```

Add new tests in the relevant `src/*.rs` file in `#[cfg(test)]` modules.

### Python (unit tests)

```bash
# Bucketing logic, pure Python
python -m pytest python/tests/ -v
```

Add new tests in `python/tests/test_*.py`.

### Integration (requires GPU)

Create `tests/integration.py` for end-to-end runs:
```python
# Example: test that overlap is faster than no-overlap
# (These will skip on CI without a GPU)
```

CI currently runs Rust + Python unit tests. GPU integration tests are manual (run on a cluster).

## Submitting a PR

1. **Fork** the repo and create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes** and commit:
   ```bash
   cargo fmt --all
   cargo clippy --workspace
   cargo test -p gradsync-core
   python -m pytest python/tests/
   git commit -m "Area: description"
   ```

3. **Push** and open a PR:
   ```bash
   git push origin feature/my-feature
   ```

4. **CI must pass:**
   - `cargo fmt --check`
   - `cargo clippy -D warnings`
   - `cargo test`
   - `pytest`
   - (GPU tests manual; link results if applicable)

5. **Respond to review** — maintainer(s) will suggest clarifications or improvements.

## Documentation

- **Code comments** — Explain the *why*, not the what. Readers can see the what.
- **Doc comments** (Rust `///`, Python `"""`):
  - Public items must have doc comments.
  - Include examples when behavior is non-obvious.
- **Architecture docs** (`docs/*.md`):
  - High-level design decisions.
  - Ordering invariants (e.g., CUDA streams).
  - Rationale for significant choices.
  - Update when changing core mechanisms.

Example Rust doc:
```rust
/// Enqueue a fused in-place AllReduce over a bucket of tensors.
///
/// All buffers are reduced in a single NCCL operation
/// (ncclGroupStart/End), which amortizes the launch/latency cost.
/// Returns immediately; use `sync_stream` to wait.
///
/// # Panics
///
/// Never; errors are propagated via Result.
pub fn group_all_reduce(...) -> Result<()> { ... }
```

## Reporting bugs

Open an issue with:
1. **Minimal reproduction** — code snippet or command to reproduce.
2. **Environment** — GPU model, CUDA version, NCCL version, Python version.
3. **Expected vs actual** — what should happen vs what does.
4. **Logs** — output of `NCCL_DEBUG=INFO` if relevant.

Example:
```
Title: AllReduce hangs on 4-GPU node

Environment:
- 4× A100 40GB on a2-ultragpu-4g
- CUDA 12.1, NCCL 2.18.5, PyTorch 2.0, Python 3.10

Reproduction:
$ python -m gradsync.launch --nproc-per-node 4 examples/allreduce_test.py
[rank 0] allreduce sum -> ... (hangs here)

Logs (NCCL_DEBUG=INFO):
[... logs indicating AllReduce stuck ...]

Expected:
Should print OK and exit.
```

## Questions?

- **Design questions:** Open a discussion or issue.
- **How do I...?** Check [README.md](README.md) and [docs/architecture.md](docs/architecture.md) first; then ask.
- **Blame for a confusing line:** That's a documentation bug; raise it.

---

**Thank you for contributing!** The best contributions are those that make the codebase easier to understand and more correct.
