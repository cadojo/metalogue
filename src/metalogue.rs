// This is not a module! This source code is included
// directly in `lib.rs`.

// Declare crate modules
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
