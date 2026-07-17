//! PyO3 bindings: expose gradsync-core to Python as `gradsync._gradsync`.
//!
//! The surface is deliberately tiny and low-level — a `Comm` object plus a couple
//! of collectives that take a raw device pointer (an int, from
//! `tensor.data_ptr()`). All the ergonomics (DDP wrapper, launcher) live in the
//! pure-Python SDK so they are easy to read and iterate on without recompiling.

// The pyo3 0.22 method/function macros expand to code with an `.into()` on the
// returned error; clippy attributes the resulting `useless_conversion` lint to
// our function signatures. It's a macro-generated false positive, so silence it.
#![allow(clippy::useless_conversion)]

use gradsync_core as gs;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::os::raw::c_void;

fn map_err(e: gs::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Rank 0 calls this and ships the returned 128 bytes to the other ranks.
#[pyfunction]
fn generate_unique_id() -> PyResult<Vec<u8>> {
    let id = gs::UniqueId::generate().map_err(map_err)?;
    Ok(id.as_bytes().to_vec())
}

/// A live NCCL communicator for one rank / one GPU.
///
/// `unsendable`: the underlying `ncclComm_t` is a raw device handle bound to the
/// thread/GPU that created it, so PyO3 must refuse to move it across threads.
#[pyclass(unsendable)]
struct Comm {
    inner: gs::Communicator,
}

#[pymethods]
impl Comm {
    /// Join the collective group. `unique_id` is the 128 bytes from rank 0.
    #[new]
    fn new(unique_id: Vec<u8>, world_size: i32, rank: i32, device: i32) -> PyResult<Self> {
        if unique_id.len() != 128 {
            return Err(PyValueError::new_err("unique_id must be exactly 128 bytes"));
        }
        let mut buf = [0u8; 128];
        buf.copy_from_slice(&unique_id);
        let id = gs::UniqueId::from_bytes(&buf);
        let inner = gs::Communicator::init_rank(&id, world_size, rank, device).map_err(map_err)?;
        Ok(Comm { inner })
    }

    #[getter]
    fn rank(&self) -> i32 {
        self.inner.rank()
    }
    #[getter]
    fn world_size(&self) -> i32 {
        self.inner.world_size()
    }
    #[getter]
    fn device(&self) -> i32 {
        self.inner.device()
    }

    /// In-place float32 sum-AllReduce over a device buffer — **synchronous**.
    ///
    /// `ptr` is a CUDA device pointer as an int (e.g. `tensor.data_ptr()`),
    /// `count` the number of f32 elements. With `average=True` NCCL divides by the
    /// world size — the usual choice for gradient averaging. Enqueues on the
    /// default stream and blocks until done. This is the naive DDP path; the
    /// overlapped path uses `all_reduce_bucket_f32` on a dedicated stream instead.
    #[pyo3(signature = (ptr, count, average=true))]
    fn all_reduce_f32(&self, ptr: usize, count: usize, average: bool) -> PyResult<()> {
        let buf = gs::DeviceBuf {
            ptr: ptr as *mut c_void,
            count,
            dtype: gs::DType::F32,
        };
        let op = if average {
            gs::RedOp::Avg
        } else {
            gs::RedOp::Sum
        };
        let stream: gs::Stream = std::ptr::null_mut();
        gs::all_reduce(&self.inner, buf, op, stream).map_err(map_err)?;
        gs::sync_stream(stream).map_err(map_err)
    }

    /// In-place float32 AllReduce over a **bucket** of tensors, fused into one
    /// NCCL group op and enqueued on `stream` — **asynchronous** (no sync).
    ///
    /// `bufs` is a list of `(device_ptr, count)` pairs. `stream` is a raw
    /// `cudaStream_t` as an int (pass `torch.cuda.Stream.cuda_stream`; `0` means
    /// the default stream). The caller orders the stream via CUDA stream waits —
    /// this is what allows the reduction to overlap with backward compute.
    #[pyo3(signature = (bufs, average=true, stream=0))]
    fn all_reduce_bucket_f32(
        &self,
        bufs: Vec<(usize, usize)>,
        average: bool,
        stream: usize,
    ) -> PyResult<()> {
        let device_bufs: Vec<gs::DeviceBuf> = bufs
            .into_iter()
            .map(|(ptr, count)| gs::DeviceBuf {
                ptr: ptr as *mut c_void,
                count,
                dtype: gs::DType::F32,
            })
            .collect();
        let op = if average {
            gs::RedOp::Avg
        } else {
            gs::RedOp::Sum
        };
        gs::group_all_reduce(&self.inner, &device_bufs, op, stream as gs::Stream).map_err(map_err)
    }

    /// Block until all work on `stream` (a raw `cudaStream_t` int; `0` = default)
    /// has completed.
    #[pyo3(signature = (stream=0))]
    fn sync_stream(&self, stream: usize) -> PyResult<()> {
        gs::sync_stream(stream as gs::Stream).map_err(map_err)
    }

    /// Broadcast a float32 device buffer from `root` to all ranks — synchronous.
    fn broadcast_f32(&self, ptr: usize, count: usize, root: i32) -> PyResult<()> {
        let buf = gs::DeviceBuf {
            ptr: ptr as *mut c_void,
            count,
            dtype: gs::DType::F32,
        };
        let stream: gs::Stream = std::ptr::null_mut();
        gs::broadcast(&self.inner, buf, root, stream).map_err(map_err)?;
        gs::sync_stream(stream).map_err(map_err)
    }
}

/// Parse `nvidia-smi topo -m` output; returns True if any GPU pair is on NVLink.
#[pyfunction]
fn topo_has_nvlink(nvidia_smi_topo_output: &str) -> bool {
    let m = gs::topology::parse_topo_matrix(nvidia_smi_topo_output);
    gs::topology::has_nvlink(&m)
}

#[pyfunction]
fn nccl_enabled() -> bool {
    gs::nccl_enabled()
}

#[pymodule]
fn _gradsync(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_unique_id, m)?)?;
    m.add_function(wrap_pyfunction!(topo_has_nvlink, m)?)?;
    m.add_function(wrap_pyfunction!(nccl_enabled, m)?)?;
    m.add_class::<Comm>()?;
    Ok(())
}
