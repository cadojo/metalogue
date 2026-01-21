//! Metal compute utilities for GPU programming with Python bindings.

// The core library only works on macOS
#![cfg(target_os = "macos")]

mod core;
mod error;
mod py;

// Re-export core types for Rust usage
pub use core::*;
pub use error::Error;

// Re-export core dependencies for Rust usage
pub use objc2;
pub use objc2_foundation;
pub use objc2_metal;
