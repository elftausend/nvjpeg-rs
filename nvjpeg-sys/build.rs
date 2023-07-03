use std::path::{Path, PathBuf};

fn main() {

    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    // not required if the main branch version of custos is used
    link_cuda();
}

// https://github.com/coreylowman/cudarc/blob/main/build.rs
fn link_cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    let candidates: Vec<PathBuf> = root_candidates().collect();

    let toolkit_root = root_candidates()
        .find(|path| path.join("include").join("cuda.h").is_file())
        .unwrap_or_else(|| {
            panic!(
                "Unable to find `include/cuda.h` under any of: {candidates:?}. Set the `CUDA_ROOT` environment variable to `$CUDA_ROOT/include/cuda.h` to override path.",
            )
        });

    for path in lib_candidates(&toolkit_root) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=nvjpeg");
}

fn root_candidates() -> impl Iterator<Item = PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::into);
    env_vars.chain(roots).map(Into::<PathBuf>::into)
}

fn lib_candidates(root: &Path) -> Vec<PathBuf> {
    [
        "lib",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ]
    .iter()
    .map(|&p| root.join(p))
    .filter(|p| p.is_dir())
    .collect()
}
