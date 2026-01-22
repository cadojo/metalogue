//! Interfaces for GPU programming with Apple Silicon.

pub trait Device {
    fn acquire() -> Self;
}
