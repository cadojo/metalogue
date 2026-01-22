//! GPU buffer management and data transfer.

use std::marker::PhantomData;

use objc2_metal::{MTLBuffer, MTLDevice};

use super::devices::Device;
use crate::error::Error;

/// A GPU buffer containing elements of type `T`.
///
/// The `Copy` bound ensures the type can be safely transferred to/from GPU memory.
pub struct Buffer<T: Copy> {
    inner: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> Buffer<T> {
    /// Creates a buffer initialized with the contents of a slice.
    pub fn from_slice(device: &Device, data: &[T]) -> Result<Self, Error> {
        let len = data.len();
        let byte_len = len * std::mem::size_of::<T>();

        let inner = if byte_len == 0 {
            // Metal doesn't support zero-length buffers
            device
                .as_raw()
                .newBufferWithLength_options(
                    std::mem::size_of::<T>().max(1),
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .ok_or(Error::BufferCreation)?
        } else {
            // SAFETY: data.as_ptr() is valid for `byte_len` bytes, and T: Copy ensures
            // the memory can be safely bitwise copied to GPU memory.
            unsafe {
                device
                    .as_raw()
                    .newBufferWithBytes_length_options(
                        std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut std::ffi::c_void),
                        byte_len,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(Error::BufferCreation)?
            }
        };

        Ok(Self {
            inner,
            len,
            _marker: PhantomData,
        })
    }

    /// Creates a buffer with space for `len` elements.
    ///
    /// Use this for output buffers that will be written by the GPU.
    pub fn with_len(device: &Device, len: usize) -> Result<Self, Error> {
        let byte_len = (len * std::mem::size_of::<T>()).max(1);

        let inner = device
            .as_raw()
            .newBufferWithLength_options(
                byte_len,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .ok_or(Error::BufferCreation)?;

        Ok(Self {
            inner,
            len,
            _marker: PhantomData,
        })
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the buffer contents as a slice.
    ///
    /// For output buffers, call this after the GPU work has completed.
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: The buffer was created with `len` elements of type T, and
        // StorageModeShared ensures CPU can read the memory. T: Copy ensures
        // reading the bytes as T is safe.
        let ptr = self.inner.contents().as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Returns a reference to the underlying Metal buffer.
    pub fn as_raw(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer> {
        &self.inner
    }
}
