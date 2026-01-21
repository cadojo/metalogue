//! Error types for metalogue.

use std::fmt;

/// Error type for metalogue operations.
#[derive(Debug)]
pub enum Error {
    /// An I/O error occurred (e.g., reading a file).
    Io(std::io::Error),
    /// No GPU device was found.
    DeviceNotFound,
    /// Failed to compile Metal source into a library.
    LibraryCompilation(String),
    /// Failed to find a function in a library.
    FunctionNotFound(String),
    /// Failed to create a compute pipeline.
    PipelineCreation(String),
    /// Failed to create a buffer.
    BufferCreation,
    /// Failed to create a command queue.
    QueueCreation,
    /// Failed to create a command buffer.
    CommandBufferCreation,
    /// Failed to create a command encoder.
    EncoderCreation,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "I/O error: {err}"),
            Error::DeviceNotFound => write!(f, "no GPU device found"),
            Error::LibraryCompilation(msg) => write!(f, "library compilation failed: {msg}"),
            Error::FunctionNotFound(name) => write!(f, "function not found: {name}"),
            Error::PipelineCreation(msg) => write!(f, "pipeline creation failed: {msg}"),
            Error::BufferCreation => write!(f, "failed to create buffer"),
            Error::QueueCreation => write!(f, "failed to create command queue"),
            Error::CommandBufferCreation => write!(f, "failed to create command buffer"),
            Error::EncoderCreation => write!(f, "failed to create command encoder"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}
