//! gradsync-core: a thin, safe Rust layer over NCCL + CUDA.
//!
//! The design goal is that the *interesting* logic (rendezvous, collectives,
//! topology reporting) lives here in auditable Rust, and the Python SDK is a thin
//! ergonomic shell on top via PyO3. Nothing in this crate is torch-specific: it
//! moves bytes at device pointers, so it can back any framework.
//!
//! Build modes:
//!
//! - default (laptop): collectives return `Error::StubBuild`; everything else
//!   (topology parsing, id (de)serialization) works and is unit-tested.
//! - `--features nccl`: links libnccl + libcudart; collectives run for real,
//!   buildable only on a GPU host (see infra/terraform).

// In stub builds (no `nccl` feature) several accessors and the FFI wrappers are
// legitimately unused — they exist for the real GPU build. Don't warn on those.
#![cfg_attr(not(feature = "nccl"), allow(dead_code))]

pub mod collectives;
pub mod comm;
pub mod error;
pub mod nccl;
pub mod topology;

pub use collectives::{all_reduce, broadcast, DType, DeviceBuf, RedOp};
pub use comm::{Communicator, UniqueId};
pub use error::{Error, Result};

/// True if this build can actually run collectives.
pub const fn nccl_enabled() -> bool {
    cfg!(feature = "nccl")
}
