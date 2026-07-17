//! Topology reporting.
//!
//! gradsync does not choose transports — NCCL does. But a big part of the
//! educational value of this project is *showing* what NCCL chose, so a user can
//! see "these two GPUs are talking over NVLink" vs "these two are going over the
//! network." We surface that by reading NCCL's own debug output and the machine's
//! `nvidia-smi topo -m` matrix, rather than guessing.
//!
//! The recommended way to see the ground truth at runtime is to launch with
//! `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH`, which prints the chosen rings
//! and whether each link is P2P/NVLink or NET.

/// A coarse classification of the link between two local GPUs, parsed from
/// `nvidia-smi topo -m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkKind {
    /// Same GPU.
    Self_,
    /// NVLink (one or more bonded links): NV1, NV2, ...
    NvLink,
    /// PCIe path through a single switch or host bridge.
    Pcie,
    /// Crosses a CPU/NUMA boundary (SYS) — slowest intra-node path.
    System,
    /// Anything we could not classify.
    Unknown,
}

impl LinkKind {
    fn from_token(tok: &str) -> Self {
        let t = tok.trim();
        if t == "X" {
            LinkKind::Self_
        } else if t.starts_with("NV") {
            LinkKind::NvLink
        } else if t == "PIX" || t == "PXB" || t == "PHB" {
            LinkKind::Pcie
        } else if t == "SYS" || t == "NODE" {
            LinkKind::System
        } else {
            LinkKind::Unknown
        }
    }
}

/// Parse the body of `nvidia-smi topo -m` into an NxN matrix of link kinds.
///
/// This is intentionally string-based and dependency-free so it works on any GPU
/// host without extra crates. Rows look like:
///   `GPU0    X    NV12  SYS  ...`
pub fn parse_topo_matrix(output: &str) -> Vec<Vec<LinkKind>> {
    let mut matrix = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if !line.starts_with("GPU") {
            continue;
        }
        // Split off the "GPUn" label, then read tokens until we hit the trailing
        // CPU-affinity / NUMA columns (which are numeric ranges, not link codes).
        let mut tokens = line.split_whitespace();
        let _label = tokens.next();
        let row: Vec<LinkKind> = tokens
            .take_while(|t| {
                let c = t.chars().next().unwrap_or(' ');
                c == 'X' || c == 'N' || c == 'P' || c == 'S'
            })
            .map(LinkKind::from_token)
            .collect();
        if !row.is_empty() {
            matrix.push(row);
        }
    }
    matrix
}

/// Does this host have at least one NVLink connection between two GPUs?
pub fn has_nvlink(matrix: &[Vec<LinkKind>]) -> bool {
    matrix.iter().flatten().any(|k| *k == LinkKind::NvLink)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvlink_and_sys() {
        // A trimmed 2-GPU A100 topo matrix.
        let sample = "\
        \tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\n\
        GPU0\tX\tNV12\t0-23\t0\n\
        GPU1\tNV12\tX\t0-23\t0\n";
        let m = parse_topo_matrix(sample);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0][0], LinkKind::Self_);
        assert_eq!(m[0][1], LinkKind::NvLink);
        assert!(has_nvlink(&m));
    }

    #[test]
    fn detects_no_nvlink() {
        let sample = "\
        GPU0\tX\tSYS\t0-15\t0\n\
        GPU1\tSYS\tX\t16-31\t1\n";
        let m = parse_topo_matrix(sample);
        assert!(!has_nvlink(&m));
        assert_eq!(m[0][1], LinkKind::System);
    }
}
