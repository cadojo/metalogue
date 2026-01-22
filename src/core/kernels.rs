//! Kernel compilation and compute pipeline management.

use std::path::Path;

use objc2_metal::{MTLDevice, MTLLibrary};

use super::devices::Device;
use crate::error::Error;

/// A Metal kernel source with an associated function name.
pub struct Kernel {
    pub code: String,
    pub name: String,
}

impl Kernel {
    /// Creates a kernel from source code and function name.
    pub fn new(code: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            name: function_name.into(),
        }
    }

    /// Creates a kernel from a file path and function name.
    pub fn from_file(filepath: &Path, function_name: &str) -> Result<Self, Error> {
        let code = std::fs::read_to_string(filepath)?;
        Ok(Self::new(code, function_name))
    }

    /// Compiles this kernel into a compute pipeline on the given device.
    pub fn to_pipeline(&self, device: &Device) -> Result<Pipeline, Error> {
        let source = objc2_foundation::NSString::from_str(&self.code);
        let library = device
            .as_raw()
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| Error::LibraryCompilation(e.localizedDescription().to_string()))?;

        let name = objc2_foundation::NSString::from_str(&self.name);
        let function = library
            .newFunctionWithName(&name)
            .ok_or_else(|| Error::FunctionNotFound(self.name.clone()))?;

        let inner = device
            .as_raw()
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| Error::PipelineCreation(e.localizedDescription().to_string()))?;

        Ok(Pipeline { inner })
    }
}

/// A compiled compute pipeline ready for execution.
pub struct Pipeline {
    inner: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
}

impl Pipeline {
    /// Returns a reference to the underlying Metal pipeline state.
    pub fn as_raw(
        &self,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState> {
        &self.inner
    }
}
