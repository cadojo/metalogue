//! Metal compute utilities for GPU programming with Python bindings.

// The core library only works on macOS
#![cfg(target_os = "macos")]

mod core;
mod error;

// Re-export core types for Rust usage
pub use core::*;
pub use error::Error;
pub use objc2;
pub use objc2_foundation;
pub use objc2_metal;

// Python bindings - these will be implemented later
#[cfg(feature = "pyo3")]
mod python {
    use pyo3::prelude::*;

    /// A Python module implemented in Rust.
    #[pymodule]
    fn metalogue(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Python bindings will be added here
        Ok(())
    }
}
