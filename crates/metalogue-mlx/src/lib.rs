//! MLX-backend to `metalogue`.

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("This library only works on Apple Silicon platforms.");

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
include!("metalogue_mlx.rs");
