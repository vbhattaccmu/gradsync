//! The collective operations, as safe wrappers over the NCCL calls.
//!
//! Buffers are device pointers (CUDA VA) plus an element count. In the Python
//! layer these come straight from a torch tensor's `.data_ptr()`, so gradsync
//! never copies gradients through host memory — the AllReduce runs GPU-to-GPU
//! over NVLink or the network, which is the whole point.
//!
//! Ordering model: every collective here is **enqueue-only** — it schedules work
//! on the given CUDA `stream` and returns immediately, it does not synchronize.
//! The caller is responsible for ordering:
//!   * The simple synchronous API (the `all_reduce_f32` binding) enqueues on the
//!     default stream and then calls `sync_stream`.
//!   * The overlapped DDP path enqueues each gradient bucket on a dedicated comm
//!     stream and uses CUDA stream waits (via torch) to order the comm stream
//!     after the backward kernels and the optimizer after the comm stream.
//!
//! This split is exactly what lets communication overlap with backward compute.

#[cfg(feature = "nccl")]
use crate::comm::check;
use crate::comm::Communicator;
use crate::error::Result;
use crate::nccl;
use std::os::raw::c_void;

/// A CUDA stream handle. `Stream::DEFAULT` is the null (default) stream.
pub type Stream = *mut c_void;

/// Element type of a device buffer.
#[derive(Clone, Copy, Debug)]
pub enum DType {
    F16,
    F32,
    F64,
}

impl DType {
    fn nccl(self) -> std::os::raw::c_int {
        match self {
            DType::F16 => nccl::dtype::NCCL_FLOAT16,
            DType::F32 => nccl::dtype::NCCL_FLOAT32,
            DType::F64 => nccl::dtype::NCCL_FLOAT64,
        }
    }
}

/// Reduction operator for AllReduce.
#[derive(Clone, Copy, Debug)]
pub enum RedOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
}

impl RedOp {
    fn nccl(self) -> std::os::raw::c_int {
        match self {
            RedOp::Sum => nccl::redop::NCCL_SUM,
            RedOp::Prod => nccl::redop::NCCL_PROD,
            RedOp::Max => nccl::redop::NCCL_MAX,
            RedOp::Min => nccl::redop::NCCL_MIN,
            RedOp::Avg => nccl::redop::NCCL_AVG,
        }
    }
}

/// A raw device buffer: a CUDA device pointer plus its element count.
///
/// # Safety
/// The pointer must be a valid device allocation of at least `count` elements of
/// the given `DType`, alive until the enqueued collective completes on `stream`.
#[derive(Clone, Copy)]
pub struct DeviceBuf {
    pub ptr: *mut c_void,
    pub count: usize,
    pub dtype: DType,
}

/// Enqueue an in-place AllReduce on `stream`. Does **not** synchronize.
///
/// This is the operation that averages gradients in data-parallel training.
pub fn all_reduce(comm: &Communicator, buf: DeviceBuf, op: RedOp, stream: Stream) -> Result<()> {
    #[cfg(feature = "nccl")]
    unsafe {
        let rc = nccl::ncclAllReduce(
            buf.ptr,
            buf.ptr, // in place
            buf.count,
            buf.dtype.nccl(),
            op.nccl(),
            comm.raw(),
            stream,
        );
        check(rc)
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (comm, buf, op, stream);
        Err(crate::error::Error::StubBuild)
    }
}

/// Enqueue an in-place AllReduce over a whole *bucket* of buffers as a single
/// fused NCCL operation (one `ncclGroupStart`/`ncclGroupEnd` around N reduces).
///
/// This is what the overlapped DDP calls per bucket: fusing amortizes the
/// per-collective launch/latency cost across many gradient tensors, and reducing
/// in place avoids the flatten/unflatten copies a single-buffer bucket would need.
/// Does **not** synchronize.
pub fn group_all_reduce(
    comm: &Communicator,
    bufs: &[DeviceBuf],
    op: RedOp,
    stream: Stream,
) -> Result<()> {
    #[cfg(feature = "nccl")]
    unsafe {
        check(nccl::ncclGroupStart())?;
        // If any enqueue fails we still must close the group before returning, or
        // the NCCL state is left dangling. Capture the first error and end anyway.
        let mut first_err = Ok(());
        for buf in bufs {
            let rc = nccl::ncclAllReduce(
                buf.ptr,
                buf.ptr,
                buf.count,
                buf.dtype.nccl(),
                op.nccl(),
                comm.raw(),
                stream,
            );
            if rc != nccl::NCCL_SUCCESS && first_err.is_ok() {
                first_err = check(rc);
            }
        }
        check(nccl::ncclGroupEnd())?;
        first_err
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (comm, bufs, op, stream);
        Err(crate::error::Error::StubBuild)
    }
}

/// Enqueue a broadcast of `buf` from `root` to every rank on `stream`. Used to
/// sync initial weights. Does **not** synchronize.
pub fn broadcast(comm: &Communicator, buf: DeviceBuf, root: i32, stream: Stream) -> Result<()> {
    #[cfg(feature = "nccl")]
    unsafe {
        let rc = nccl::ncclBroadcast(
            buf.ptr,
            buf.ptr,
            buf.count,
            buf.dtype.nccl(),
            root,
            comm.raw(),
            stream,
        );
        check(rc)
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (comm, buf, root, stream);
        Err(crate::error::Error::StubBuild)
    }
}

/// Block the calling host thread until all work on `stream` has completed.
pub fn sync_stream(stream: Stream) -> Result<()> {
    #[cfg(feature = "nccl")]
    unsafe {
        let rc = nccl::cudaStreamSynchronize(stream);
        if rc != 0 {
            return Err(crate::error::Error::Cuda(rc));
        }
        Ok(())
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = stream;
        Err(crate::error::Error::StubBuild)
    }
}
