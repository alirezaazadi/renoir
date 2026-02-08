use std::fmt::{Display, Formatter};
use std::sync::Arc;

use once_cell::sync::OnceCell;

use cubecl::prelude::{Numeric, TensorHandleRef};
use cubecl::reduce;
use cubecl::reduce::instructions::{Max, Min, Prod, Sum};
use cubecl::reduce::{ReduceFamily, ReducePrecision};
use cubecl::{CubeElement, Runtime};

#[cfg(feature = "gpu-wgpu")]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

#[cfg(feature = "gpu-cuda")]
use cubecl::cuda::{CudaDevice, CudaRuntime};

use crate::block::{BlockStructure, OperatorStructure};
use crate::operator::{Operator, StreamElement, Timestamp};
use crate::scheduler::ExecutionMetadata;

const DEFAULT_BATCH_SIZE: usize = 1 << 16; // 65_536 items => 512 KiB per batch
const DEFAULT_TILE_SIZE: usize = 1 << 18; // 262_144 elements per GPU tile (~1 MB)

#[derive(Debug, thiserror::Error, Clone)]
#[allow(dead_code)]
pub(crate) enum ReduceGpuError {
    #[error("no GPU backend available")]
    NoAvailableBackend,
    #[error("GPU runtime unavailable: {0}")]
    GpuUnavailable(String),
    #[error("CubeCL reduce failure: {0}")]
    CubeCl(String),
}

impl From<reduce::ReduceError> for ReduceGpuError {
    fn from(value: reduce::ReduceError) -> Self {
        Self::CubeCl(value.to_string())
    }
}

/// Available GPU reduce operations.
#[derive(Clone, Copy, Debug)]
pub enum ReduceKernel {
    /// Element-wise sum reduction.
    Sum,
    /// Element-wise product reduction.
    Product,
    /// Minimum value reduction.
    Min,
    /// Maximum value reduction.
    Max,
}

impl ReduceKernel {
    fn label(&self) -> &'static str {
        match self {
            ReduceKernel::Sum => "sum",
            ReduceKernel::Product => "product",
            ReduceKernel::Min => "min",
            ReduceKernel::Max => "max",
        }
    }

    fn combine(&self, current: f64, partial: f64) -> f64 {
        match self {
            ReduceKernel::Sum => current + partial,
            ReduceKernel::Product => current * partial,
            ReduceKernel::Min => current.min(partial),
            ReduceKernel::Max => current.max(partial),
        }
    }

    fn identity_value(&self) -> f64 {
        match self {
            ReduceKernel::Sum => 0.0,
            ReduceKernel::Product => 1.0,
            ReduceKernel::Min => f64::INFINITY,
            ReduceKernel::Max => f64::NEG_INFINITY,
        }
    }
}

/// Configuration for the GPU reduce operator.
#[derive(Clone, Debug)]
pub struct ReduceGpuConfig {
    /// Number of elements to buffer before flushing to GPU.
    pub batch_size: usize,
    /// Which GPU backend to use.
    pub backend: ReduceGpuBackend,
    /// Tile size for the GPU reduction kernel (elements per GPU tile).
    pub tile_size: usize,
}

impl Default for ReduceGpuConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            backend: ReduceGpuBackend::default(),
            tile_size: DEFAULT_TILE_SIZE,
        }
    }
}

impl ReduceGpuConfig {
    /// Set the batch size (number of elements buffered before GPU dispatch).
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Set the GPU backend to use.
    pub fn with_backend(mut self, backend: ReduceGpuBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the tile size for the GPU reduction kernel.
    pub fn with_tile_size(mut self, tile_size: usize) -> Self {
        self.tile_size = tile_size.max(1);
        self
    }
}

/// Selects which GPU backend to use for reduction.
#[derive(Clone, Copy, Debug)]
pub enum ReduceGpuBackend {
    /// Automatically select the best available backend.
    Auto,
    /// Use the WGPU backend (cross-platform: Metal, Vulkan, DirectX 12).
    #[cfg(feature = "gpu-wgpu")]
    Wgpu,
    /// Use the CUDA backend (NVIDIA GPUs).
    #[cfg(feature = "gpu-cuda")]
    Cuda,
}

impl Default for ReduceGpuBackend {
    fn default() -> Self {
        #[cfg(feature = "gpu-wgpu")]
        {
            return ReduceGpuBackend::Wgpu;
        }
        #[cfg(all(not(feature = "gpu-wgpu"), feature = "gpu-cuda"))]
        {
            return ReduceGpuBackend::Cuda;
        }
        #[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
        {
            ReduceGpuBackend::Auto
        }
    }
}

// ---------------------------------------------------------------------------
// Backend handle: abstraction over WGPU / CUDA
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum BackendHandle {
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(Arc<WgpuBackendCtx>),
    #[cfg(feature = "gpu-cuda")]
    Cuda(Arc<CudaBackendCtx>),
}

impl BackendHandle {
    fn reduce(
        &self,
        values: BatchSlice<'_>,
        kernel: ReduceKernel,
        tile_size: usize,
    ) -> Result<f64, ReduceGpuError> {
        match self {
            #[cfg(feature = "gpu-wgpu")]
            BackendHandle::Wgpu(ctx) => match values {
                BatchSlice::F32(slice) => ctx.reduce(slice, kernel, tile_size),
                #[cfg(feature = "gpu-cuda")]
                BatchSlice::F64(_) => unreachable!("wgpu backend requires f32 batches"),
            },
            #[cfg(feature = "gpu-cuda")]
            BackendHandle::Cuda(ctx) => match values {
                BatchSlice::F32(_) => unreachable!("cuda backend requires f64 batches"),
                BatchSlice::F64(slice) => ctx.reduce(slice, kernel, tile_size),
            },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "gpu-wgpu")]
            BackendHandle::Wgpu(ctx) => ctx.runtime_name,
            #[cfg(feature = "gpu-cuda")]
            BackendHandle::Cuda(ctx) => ctx.runtime_name,
        }
    }

    fn summary(&self) -> &str {
        match self {
            #[cfg(feature = "gpu-wgpu")]
            BackendHandle::Wgpu(ctx) => &ctx.adapter_summary,
            #[cfg(feature = "gpu-cuda")]
            BackendHandle::Cuda(ctx) => &ctx.device_summary,
        }
    }

    fn precision(&self) -> GpuPrecision {
        match self {
            #[cfg(feature = "gpu-wgpu")]
            BackendHandle::Wgpu(ctx) => ctx.precision,
            #[cfg(feature = "gpu-cuda")]
            BackendHandle::Cuda(ctx) => ctx.precision,
        }
    }
}

/// Indicates which numeric precision a backend should use.
/// CUDA executes kernels in `f64`, while WGPU always leverages `f32`
/// (because portable adapters seldom expose `SHADER_F64`). Conversions
/// ensure the public API continues to expose `f64`.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuPrecision {
    F32,
    #[cfg(feature = "gpu-cuda")]
    F64,
}

enum BatchStorage {
    F32(Vec<f32>),
    #[cfg(feature = "gpu-cuda")]
    F64(Vec<f64>),
}

#[allow(dead_code)]
enum BatchSlice<'a> {
    F32(&'a [f32]),
    #[cfg(feature = "gpu-cuda")]
    F64(&'a [f64]),
}

impl BatchStorage {
    fn new(precision: GpuPrecision, capacity: usize) -> Self {
        match precision {
            GpuPrecision::F32 => BatchStorage::F32(Vec::with_capacity(capacity)),
            #[cfg(feature = "gpu-cuda")]
            GpuPrecision::F64 => BatchStorage::F64(Vec::with_capacity(capacity)),
        }
    }

    fn push(&mut self, value: f64) {
        match self {
            BatchStorage::F32(buffer) => buffer.push(value as f32),
            #[cfg(feature = "gpu-cuda")]
            BatchStorage::F64(buffer) => buffer.push(value),
        }
    }

    fn len(&self) -> usize {
        match self {
            BatchStorage::F32(buffer) => buffer.len(),
            #[cfg(feature = "gpu-cuda")]
            BatchStorage::F64(buffer) => buffer.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn clear(&mut self) {
        match self {
            BatchStorage::F32(buffer) => buffer.clear(),
            #[cfg(feature = "gpu-cuda")]
            BatchStorage::F64(buffer) => buffer.clear(),
        }
    }

    fn as_slice(&self) -> BatchSlice<'_> {
        match self {
            BatchStorage::F32(buffer) => BatchSlice::F32(buffer),
            #[cfg(feature = "gpu-cuda")]
            BatchStorage::F64(buffer) => BatchSlice::F64(buffer),
        }
    }

    fn pad_to_multiple(&mut self, tile: usize, identity: f64) {
        if tile == 0 {
            return;
        }
        let len = self.len();
        if len < tile {
            return;
        }
        let remainder = len % tile;
        if remainder == 0 {
            return;
        }
        let needed = tile - remainder;
        match self {
            BatchStorage::F32(buffer) => buffer.resize(len + needed, identity as f32),
            #[cfg(feature = "gpu-cuda")]
            BatchStorage::F64(buffer) => buffer.resize(len + needed, identity),
        }
    }
}

// ---------------------------------------------------------------------------
// The ReduceGpu operator
// ---------------------------------------------------------------------------

pub(crate) struct ReduceGpu<Op>
where
    Op: Operator,
{
    prev: Op,
    kernel: ReduceKernel,
    backend: BackendHandle,
    precision: GpuPrecision,
    batch: BatchStorage,
    batch_size: usize,
    tile_size: usize,
    accumulator: Option<f64>,
    timestamp: Option<Timestamp>,
    max_watermark: Option<Timestamp>,
    received_end: bool,
    received_end_iter: bool,
}

impl<Op> Display for ReduceGpu<Op>
where
    Op: Operator,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> ReduceGpu<{}> (precision: {:?})",
            self.prev,
            self.kernel.label(),
            self.precision
        )
    }
}

impl<Op> Clone for ReduceGpu<Op>
where
    Op: Operator,
{
    fn clone(&self) -> Self {
        Self {
            prev: self.prev.clone(),
            kernel: self.kernel,
            backend: self.backend.clone(),
            precision: self.precision,
            batch: BatchStorage::new(self.precision, self.batch_size),
            batch_size: self.batch_size,
            tile_size: self.tile_size,
            accumulator: None,
            timestamp: None,
            max_watermark: None,
            received_end: false,
            received_end_iter: false,
        }
    }
}

impl<Op> ReduceGpu<Op>
where
    Op: Operator,
{
    pub fn new(prev: Op, kernel: ReduceKernel, config: ReduceGpuConfig) -> Self {
        let backend = config
            .backend
            .resolve_backend()
            .unwrap_or_else(|err| panic!("Failed to initialise GPU backend: {err}"));

        let batch_size = config.batch_size.max(1);
        let precision = backend.precision();
        log::info!(
            target: "renoir::gpu",
            "reduce_gpu: using {} backend ({}) with batch size {} for {} kernel ({precision:?})",
            backend.name(),
            backend.summary(),
            batch_size,
            kernel.label(),
        );

        Self {
            prev,
            kernel,
            backend,
            precision,
            batch: BatchStorage::new(precision, batch_size),
            batch_size,
            tile_size: config.tile_size.max(1),
            accumulator: None,
            timestamp: None,
            max_watermark: None,
            received_end: false,
            received_end_iter: false,
        }
    }

    fn push_value(&mut self, value: f64) {
        self.batch.push(value);
        if self.batch.len() >= self.batch_size {
            self.flush_batch();
        }
    }

    fn flush_batch(&mut self) {
        if self.batch.is_empty() {
            return;
        }
        let mut chunk = self.tile_size.min(self.batch.len());
        if chunk == 0 {
            chunk = 1;
        }
        self.batch
            .pad_to_multiple(chunk, self.kernel.identity_value());
        let len = self.batch.len();
        let slice = self.batch.as_slice();
        let partial = self
            .backend
            .reduce(slice, self.kernel, chunk)
            .unwrap_or_else(|err| panic!("reduce_gpu {} batch failed: {err}", self.kernel.label()));
        trace!(
            target: "renoir::gpu",
            "reduce_gpu batch of {len} items executed on {}",
            self.backend.name()
        );
        self.batch.clear();
        match self.accumulator.as_mut() {
            Some(acc) => *acc = self.kernel.combine(*acc, partial),
            None => self.accumulator = Some(partial),
        }
    }
}

impl<Op> Operator for ReduceGpu<Op>
where
    Op: Operator,
    Op::Out: Into<f64>,
{
    type Out = f64;

    fn setup(&mut self, metadata: &mut ExecutionMetadata) {
        self.prev.setup(metadata);
    }

    fn next(&mut self) -> StreamElement<Self::Out> {
        while !self.received_end {
            match self.prev.next() {
                StreamElement::Item(item) => self.push_value(item.into()),
                StreamElement::Timestamped(item, ts) => {
                    let ts_ref = self.timestamp.get_or_insert(ts);
                    *ts_ref = (*ts_ref).max(ts);
                    self.push_value(item.into());
                }
                StreamElement::Watermark(ts) => {
                    let wm = self.max_watermark.get_or_insert(ts);
                    *wm = (*wm).max(ts);
                }
                StreamElement::FlushBatch => self.flush_batch(),
                StreamElement::FlushAndRestart => {
                    self.flush_batch();
                    self.received_end = true;
                    self.received_end_iter = true;
                }
                StreamElement::Terminate => {
                    self.flush_batch();
                    self.received_end = true;
                }
            }
        }

        if let Some(value) = self.accumulator.take() {
            if let Some(ts) = self.timestamp.take() {
                return StreamElement::Timestamped(value, ts);
            }
            return StreamElement::Item(value);
        }

        if let Some(ts) = self.max_watermark.take() {
            return StreamElement::Watermark(ts);
        }

        if self.received_end_iter {
            self.received_end_iter = false;
            self.received_end = false;
            return StreamElement::FlushAndRestart;
        }

        StreamElement::Terminate
    }

    fn structure(&self) -> BlockStructure {
        self.prev
            .structure()
            .add_operator(OperatorStructure::new::<Self::Out, _>("ReduceGpu"))
    }
}

// ---------------------------------------------------------------------------
// Backend resolution
// ---------------------------------------------------------------------------

impl ReduceGpuBackend {
    fn resolve_backend(self) -> Result<BackendHandle, ReduceGpuError> {
        match self {
            ReduceGpuBackend::Auto => {
                #[cfg(feature = "gpu-wgpu")]
                {
                    if let Ok(handle) = WgpuBackendCtx::global() {
                        return Ok(BackendHandle::Wgpu(handle));
                    }
                }
                #[cfg(feature = "gpu-cuda")]
                {
                    if let Ok(handle) = CudaBackendCtx::global() {
                        return Ok(BackendHandle::Cuda(handle));
                    }
                }
                Err(ReduceGpuError::NoAvailableBackend)
            }
            #[cfg(feature = "gpu-wgpu")]
            ReduceGpuBackend::Wgpu => WgpuBackendCtx::global().map(BackendHandle::Wgpu),
            #[cfg(feature = "gpu-cuda")]
            ReduceGpuBackend::Cuda => CudaBackendCtx::global().map(BackendHandle::Cuda),
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-pass CubeCL reduce helper
// ---------------------------------------------------------------------------

/// Runs a CubeCL multi-pass reduction for the requested scalar type.
/// First pass: reduce each tile to one partial value.
/// Second pass: reduce all partial values to a single scalar.
fn multi_pass_reduce<R, T, Inst>(
    client: &cubecl::prelude::ComputeClient<<R as Runtime>::Server>,
    values: &[T],
    tile_size: usize,
    inst_config: <Inst as ReduceFamily>::Config,
) -> Result<T, ReduceGpuError>
where
    R: Runtime,
    T: CubeElement + Numeric + ReducePrecision,
    Inst: ReduceFamily + 'static,
    Inst::Config: Clone,
{
    let mut chunk = tile_size.max(1);
    if chunk > values.len() {
        chunk = values.len();
    }
    debug_assert!(chunk > 0 && !values.is_empty());
    let num_chunks = values.len() / chunk;
    debug_assert!(num_chunks > 0);
    debug_assert_eq!(values.len() % chunk, 0);

    let elem_size = std::mem::size_of::<T>();
    let input_handle = client.create(T::as_bytes(values));
    let input_shape = [num_chunks, chunk];
    let input_strides = [chunk, 1];
    let input_tensor = unsafe {
        TensorHandleRef::<R>::from_raw_parts(&input_handle, &input_strides, &input_shape, elem_size)
    };

    let partial_handle = client.empty(num_chunks * elem_size);
    let partial_shape = [num_chunks, 1];
    let partial_strides = [1, 1];
    let partial_tensor = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &partial_handle,
            &partial_strides,
            &partial_shape,
            elem_size,
        )
    };

    reduce::reduce::<R, T, T, Inst>(
        client,
        input_tensor,
        partial_tensor,
        1,
        None,
        inst_config.clone(),
    )?;

    let final_handle = client.empty(elem_size);
    let partial_as_input_shape = [num_chunks];
    let partial_as_input_strides = [1];
    let partial_as_input = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &partial_handle,
            &partial_as_input_strides,
            &partial_as_input_shape,
            elem_size,
        )
    };
    let final_shape = [1];
    let final_strides = [1];
    let final_tensor = unsafe {
        TensorHandleRef::<R>::from_raw_parts(&final_handle, &final_strides, &final_shape, elem_size)
    };

    reduce::reduce::<R, T, T, Inst>(client, partial_as_input, final_tensor, 0, None, inst_config)?;

    let bytes = client.read_one(final_handle);
    let out = T::from_bytes(&bytes);
    Ok(out[0])
}

// ---------------------------------------------------------------------------
// WGPU backend context (lazily initialised, singleton)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-wgpu")]
struct WgpuBackendCtx {
    client: cubecl::prelude::ComputeClient<<WgpuRuntime as Runtime>::Server>,
    adapter_summary: String,
    runtime_name: &'static str,
    precision: GpuPrecision,
}

#[cfg(feature = "gpu-wgpu")]
impl WgpuBackendCtx {
    fn new() -> Result<Self, ReduceGpuError> {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let runtime_name = WgpuRuntime::name(&client);
        log::info!(
            target: "renoir::gpu",
            "reduce_gpu: initialized WGPU backend ({}) in f32 precision",
            runtime_name
        );
        Ok(Self {
            client,
            adapter_summary: runtime_name.to_string(),
            runtime_name,
            precision: GpuPrecision::F32,
        })
    }

    fn reduce(
        &self,
        values: &[f32],
        kernel: ReduceKernel,
        tile_size: usize,
    ) -> Result<f64, ReduceGpuError> {
        match kernel {
            ReduceKernel::Sum => {
                multi_pass_reduce::<WgpuRuntime, f32, Sum>(&self.client, values, tile_size, ())
                    .map(|v| v as f64)
            }
            ReduceKernel::Product => {
                multi_pass_reduce::<WgpuRuntime, f32, Prod>(&self.client, values, tile_size, ())
                    .map(|v| v as f64)
            }
            ReduceKernel::Min => {
                multi_pass_reduce::<WgpuRuntime, f32, Min>(&self.client, values, tile_size, ())
                    .map(|v| v as f64)
            }
            ReduceKernel::Max => {
                multi_pass_reduce::<WgpuRuntime, f32, Max>(&self.client, values, tile_size, ())
                    .map(|v| v as f64)
            }
        }
    }

    fn global() -> Result<Arc<Self>, ReduceGpuError> {
        static INSTANCE: OnceCell<Result<Arc<WgpuBackendCtx>, ReduceGpuError>> = OnceCell::new();
        INSTANCE
            .get_or_init(|| WgpuBackendCtx::new().map(Arc::new))
            .clone()
    }
}

// ---------------------------------------------------------------------------
// CUDA backend context (lazily initialised, singleton)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-cuda")]
struct CudaBackendCtx {
    client: cubecl::prelude::ComputeClient<<CudaRuntime as Runtime>::Server>,
    device_summary: String,
    runtime_name: &'static str,
    precision: GpuPrecision,
}

#[cfg(feature = "gpu-cuda")]
impl CudaBackendCtx {
    fn new() -> Result<Self, ReduceGpuError> {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::client(&device);
        let runtime_name = CudaRuntime::name(&client);
        log::info!(
            target: "renoir::gpu",
            "reduce_gpu: initialized CUDA backend ({}) in f64 precision",
            runtime_name
        );
        Ok(Self {
            client,
            device_summary: runtime_name.to_string(),
            runtime_name,
            precision: GpuPrecision::F64,
        })
    }

    fn reduce(
        &self,
        values: &[f64],
        kernel: ReduceKernel,
        tile_size: usize,
    ) -> Result<f64, ReduceGpuError> {
        match kernel {
            ReduceKernel::Sum => {
                multi_pass_reduce::<CudaRuntime, f64, Sum>(&self.client, values, tile_size, ())
            }
            ReduceKernel::Product => {
                multi_pass_reduce::<CudaRuntime, f64, Prod>(&self.client, values, tile_size, ())
            }
            ReduceKernel::Min => {
                multi_pass_reduce::<CudaRuntime, f64, Min>(&self.client, values, tile_size, ())
            }
            ReduceKernel::Max => {
                multi_pass_reduce::<CudaRuntime, f64, Max>(&self.client, values, tile_size, ())
            }
        }
    }

    fn global() -> Result<Arc<Self>, ReduceGpuError> {
        static INSTANCE: OnceCell<Result<Arc<CudaBackendCtx>, ReduceGpuError>> = OnceCell::new();
        INSTANCE
            .get_or_init(|| CudaBackendCtx::new().map(Arc::new))
            .clone()
    }
}
