//! GPU context management for CubeCL-based kernels.
//!
//! This module provides a unified interface for GPU resource management
//! across different backends (WGPU, CUDA).

use cubecl::prelude::*;

#[cfg(feature = "gpu-wgpu")]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime, WgpuServer};

#[cfg(feature = "gpu-cuda")]
use cubecl::cuda::{CudaDevice, CudaRuntime, CudaServer};

/// GPU context that manages the compute client and device resources.
///
/// This struct abstracts over different GPU backends (WGPU, CUDA) and provides
/// a unified interface for kernel execution.
#[derive(Clone)]
pub struct GpuContext {
    #[cfg(feature = "gpu-wgpu")]
    client: ComputeClient<WgpuServer>,
    #[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
    client: cubecl::client::ComputeClient<CudaServer>,
}

impl GpuContext {
    /// Create a new GPU context using the default device.
    ///
    /// This will initialize the GPU and may take a few hundred milliseconds
    /// on the first call as drivers are loaded and the device is initialized.
    pub fn new() -> Self {
        #[cfg(feature = "gpu-wgpu")]
        {
            let device = WgpuDevice::default();
            let client = WgpuRuntime::client(&device);
            Self { client }
        }

        #[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
        {
            let device = CudaDevice::default();
            let client = CudaRuntime::client(&device);
            Self { client }
        }
    }

    /// Get a reference to the underlying compute client.
    ///
    /// This is useful for implementing custom kernels that need direct
    /// access to the CubeCL client.
    #[cfg(feature = "gpu-wgpu")]
    pub fn client(&self) -> &ComputeClient<WgpuServer> {
        &self.client
    }

    #[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
    pub fn client(&self) -> &cubecl::client::ComputeClient<CudaServer> {
        &self.client
    }

    /// Synchronize with the GPU, waiting for all pending operations to complete.
    pub fn sync(&self) {
        pollster::block_on(self.client.sync());
    }

    /// Allocate a GPU buffer and copy data from CPU.
    ///
    /// Returns a handle that can be used with kernel launches.
    pub fn create_buffer(&self, data: &[u8]) -> cubecl::server::Handle {
        self.client.create(data)
    }

    /// Allocate an empty GPU buffer of the specified size in bytes.
    pub fn create_empty_buffer(&self, size: usize) -> cubecl::server::Handle {
        self.client.empty(size)
    }

    /// Read data back from a GPU buffer.
    pub fn read_buffer(&self, handle: cubecl::server::Handle) -> Vec<u8> {
        self.client.read_one(handle).to_vec()
    }
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("backend", &Self::backend_name())
            .finish()
    }
}

impl GpuContext {
    /// Get the name of the active GPU backend.
    pub fn backend_name() -> &'static str {
        #[cfg(feature = "gpu-wgpu")]
        {
            "WGPU"
        }
        #[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
        {
            "CUDA"
        }
    }
}

