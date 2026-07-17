// Build script: link against CUDA runtime and NCCL when the `nccl` feature is on.
//
// This only runs meaningfully on a GPU host that has the CUDA toolkit and NCCL
// installed (see infra/terraform/startup.sh.tpl, which provisions both). On a
// laptop without the `nccl` feature, this is a no-op and the crate builds in
// stub mode.

fn main() {
    if std::env::var("CARGO_FEATURE_NCCL").is_err() {
        return;
    }

    // Allow overriding install locations; default to the common CUDA path.
    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nccl_home = std::env::var("NCCL_HOME").unwrap_or_else(|_| cuda_home.clone());

    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-search=native={nccl_home}/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nccl");

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NCCL_HOME");
}
