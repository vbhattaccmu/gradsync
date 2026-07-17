//! Error type shared across the core.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("NCCL error {code}: {msg}")]
    Nccl { code: i32, msg: String },

    #[error("CUDA runtime error {0}")]
    Cuda(i32),

    /// Returned by every collective when the crate was built without the `nccl`
    /// feature. This lets the SDK, launcher, and topology parser be developed and
    /// unit-tested on a laptop; the real ops light up on a GPU host built with
    /// `--features nccl`.
    #[error("built without the `nccl` feature: rebuild on a GPU host with --features nccl")]
    StubBuild,
}
