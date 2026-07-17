//! The collective operations, as safe wrappers over the NCCL calls.
//!
//! Buffers are device pointers (CUDA VA) plus an element count. In the Python
//! layer these come straight from a torch tensor's `.data_ptr()`, so gradsync
//! never copies gradients through host memory — the AllReduce runs GPU-to-GPU
//! over NVLink or the network, which is the whole point.

use crate::comm::Communicator;
#[cfg(feature = "nccl")]
use crate::comm::check;
use crate::error::Result;
use crate::nccl;
use std::os::raw::c_void;

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
/// the given `DType`, alive for the duration of the call.
#[derive(Clone, Copy)]
pub struct DeviceBuf {
    pub ptr: *mut c_void,
    pub count: usize,
    pub dtype: DType,
}

/// In-place AllReduce: every rank ends up with the reduction of all ranks' data.
/// This is the operation that averages gradients in data-parallel training.
///
/// Passing `stream = null` uses the default stream and blocks until complete.
pub fn all_reduce(comm: &Communicator, buf: DeviceBuf, op: RedOp, stream: *mut c_void) -> Result<()> {
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
        check(rc)?;
        let rc = nccl::cudaStreamSynchronize(stream);
        if rc != 0 {
            return Err(crate::error::Error::Cuda(rc));
        }
        Ok(())
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (comm, buf, op, stream);
        Err(crate::error::Error::StubBuild)
    }
}

/// Broadcast `buf` from `root` to every rank. Used to sync initial weights.
pub fn broadcast(comm: &Communicator, buf: DeviceBuf, root: i32, stream: *mut c_void) -> Result<()> {
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
        check(rc)?;
        let rc = nccl::cudaStreamSynchronize(stream);
        if rc != 0 {
            return Err(crate::error::Error::Cuda(rc));
        }
        Ok(())
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (comm, buf, root, stream);
        Err(crate::error::Error::StubBuild)
    }
}
