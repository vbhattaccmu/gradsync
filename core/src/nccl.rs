//! Raw FFI declarations for the subset of the NCCL C API that gradsync uses.
//!
//! These are only linked when the `nccl` feature is enabled. Everything here is
//! `unsafe` and mirrors <nccl.h>; the safe wrappers live in `comm` and
//! `collectives`. Keep this file a faithful, minimal transcription of the C API
//! so it is easy to audit against the header.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

/// Opaque NCCL communicator handle (`ncclComm_t`).
pub type ncclComm_t = *mut c_void;

/// Opaque CUDA stream handle (`cudaStream_t`).
pub type cudaStream_t = *mut c_void;

/// NCCL's out-of-band bootstrap id. It is a fixed 128-byte blob that rank 0
/// generates and shares with every other rank (over TCP, a file, etc.).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ncclUniqueId {
    pub internal: [c_char; 128],
}

impl Default for ncclUniqueId {
    fn default() -> Self {
        ncclUniqueId { internal: [0; 128] }
    }
}

/// `ncclResult_t`. 0 == success.
pub type ncclResult_t = c_int;
pub const NCCL_SUCCESS: ncclResult_t = 0;

/// `ncclDataType_t` — only the variants we expose.
pub mod dtype {
    use std::os::raw::c_int;
    pub const NCCL_FLOAT32: c_int = 7;
    pub const NCCL_FLOAT16: c_int = 6;
    pub const NCCL_FLOAT64: c_int = 8;
}

/// `ncclRedOp_t`.
pub mod redop {
    use std::os::raw::c_int;
    pub const NCCL_SUM: c_int = 0;
    pub const NCCL_PROD: c_int = 1;
    pub const NCCL_MAX: c_int = 2;
    pub const NCCL_MIN: c_int = 3;
    pub const NCCL_AVG: c_int = 4;
}

#[cfg(feature = "nccl")]
extern "C" {
    pub fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;

    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: c_int,
        commId: ncclUniqueId,
        rank: c_int,
    ) -> ncclResult_t;

    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;

    pub fn ncclAllReduce(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: c_int,
        op: c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;

    pub fn ncclBroadcast(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        count: usize,
        datatype: c_int,
        root: c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;

    pub fn ncclAllGather(
        sendbuff: *const c_void,
        recvbuff: *mut c_void,
        sendcount: usize,
        datatype: c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;

    pub fn ncclGetErrorString(result: ncclResult_t) -> *const c_char;

    // Group calls: everything enqueued between start and end is fused into a
    // single NCCL operation. We wrap a bucket of per-tensor AllReduces in a group
    // so one bucket == one network launch, no flatten/unflatten copies.
    pub fn ncclGroupStart() -> ncclResult_t;
    pub fn ncclGroupEnd() -> ncclResult_t;

    // CUDA runtime bits we need to synchronize after enqueuing a collective.
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> c_int;
    pub fn cudaSetDevice(device: c_int) -> c_int;
}
