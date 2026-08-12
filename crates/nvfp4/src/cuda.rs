//! CUDA runtime helpers and device-buffer ownership.

use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Deref;
use std::ptr::null_mut;
use std::slice;
use std::sync::OnceLock;

/// Host-side view of a device-buffer readback.
///
/// This value keeps the source device buffer borrowed for as long as the host
/// data is live, making readback boundaries visible to Rust's borrow checker.
#[derive(Debug)]
pub struct HostRead<'a, T> {
    values: Vec<T>,
    _device: PhantomData<&'a DeviceBuffer<T>>,
}

/// Page-locked host allocation suitable for asynchronous CUDA transfers.
pub struct PinnedHostBuffer<T> {
    ptr: *mut T,
    len: usize,
}

// A pinned allocation has unique ownership, and CUDA permits it to be used by
// a different host thread. Moving the owner is therefore equivalent to moving
// a Vec<T> when T is Send.
unsafe impl<T: Send> Send for PinnedHostBuffer<T> {}

impl<T: Copy> PinnedHostBuffer<T> {
    /// Allocates `len` zero-initialized values in pinned host memory.
    pub fn zeroed(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::Shape {
                label: "pinned host allocation",
                expected: "at least one value".to_string(),
                actual: "0 values".to_string(),
            });
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape {
                label: "pinned host allocation",
                expected: "len * element size without overflow".to_string(),
                actual: format!("len={len} element_size={}", size_of::<T>()),
            })?;
        let mut raw = null_mut();
        unsafe {
            check_cuda(
                "cudaHostAlloc",
                ffi::cudaHostAlloc(&mut raw, bytes, ffi::CUDA_HOST_ALLOC_DEFAULT),
            )?;
            raw.cast::<u8>().write_bytes(0, bytes);
        }
        Ok(Self {
            ptr: raw.cast(),
            len,
        })
    }

    /// Allocates pinned host memory and copies `values` into it.
    pub fn from_slice(values: &[T]) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::Shape {
                label: "pinned host allocation",
                expected: "at least one value".to_string(),
                actual: "0 values".to_string(),
            });
        }
        let bytes = std::mem::size_of_val(values);
        let mut raw = null_mut();
        unsafe {
            check_cuda(
                "cudaHostAlloc",
                ffi::cudaHostAlloc(&mut raw, bytes, ffi::CUDA_HOST_ALLOC_DEFAULT),
            )?;
            raw.cast::<T>()
                .copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
        Ok(Self {
            ptr: raw.cast(),
            len: values.len(),
        })
    }

    /// Returns the pinned values.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns the pinned values mutably for direct record decoding.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Replaces the allocation contents without changing its address.
    pub fn copy_from_slice(&mut self, values: &[T]) -> Result<()> {
        if values.len() != self.len {
            return Err(Error::Shape {
                label: "pinned host copy",
                expected: format!("{} values", self.len),
                actual: format!("{} values", values.len()),
            });
        }
        unsafe {
            self.ptr
                .copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
        Ok(())
    }
}

impl<T> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = ffi::cudaFreeHost(self.ptr.cast());
            }
        }
    }
}

/// Borrowed device input role.
pub struct DeviceInput<'a, T> {
    buffer: &'a DeviceBuffer<T>,
}

/// Borrowed device output role.
pub struct DeviceOutput<'a, T> {
    buffer: &'a mut DeviceBuffer<T>,
}

/// Borrowed device in-place role.
pub struct DeviceInOut<'a, T> {
    buffer: &'a mut DeviceBuffer<T>,
}

impl<T> HostRead<'_, T> {
    /// Consumes the readback and returns the owned host vector.
    pub fn into_vec(self) -> Vec<T> {
        self.values
    }

    /// Returns the host-side values as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
}

impl<T: PartialEq> PartialEq<[T]> for HostRead<'_, T> {
    fn eq(&self, other: &[T]) -> bool {
        self.values == other
    }
}

impl<T: PartialEq, const N: usize> PartialEq<[T; N]> for HostRead<'_, T> {
    fn eq(&self, other: &[T; N]) -> bool {
        self.values == other.as_slice()
    }
}

impl<T: PartialEq> PartialEq<Vec<T>> for HostRead<'_, T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        &self.values == other
    }
}

impl<T: PartialEq> PartialEq for HostRead<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<T> Deref for HostRead<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<'a, T> IntoIterator for HostRead<'a, T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a, T> DeviceInput<'a, T> {
    /// Creates an input-role borrow from a device buffer.
    pub fn new(buffer: &'a DeviceBuffer<T>) -> Self {
        Self { buffer }
    }

    /// Returns the number of elements available through this input role.
    pub fn len(&self) -> usize {
        self.buffer.len
    }

    /// Returns true when this input role contains no elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.len == 0
    }

    /// Returns the raw device pointer as an immutable C pointer.
    pub fn as_const_ptr(&self) -> *const c_void {
        self.buffer.ptr.cast()
    }

    /// Returns the underlying device buffer.
    pub fn buffer(&self) -> &DeviceBuffer<T> {
        self.buffer
    }
}

impl<T> Copy for DeviceInput<'_, T> {}

impl<T> Clone for DeviceInput<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> DeviceOutput<'a, T> {
    /// Creates an output-role borrow from a mutable device buffer.
    pub fn new(buffer: &'a mut DeviceBuffer<T>) -> Self {
        Self { buffer }
    }

    /// Returns the number of elements available through this output role.
    pub fn len(&self) -> usize {
        self.buffer.len
    }

    /// Returns true when this output role contains no elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.len == 0
    }

    /// Returns the raw device pointer as a mutable C pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.buffer.ptr.cast()
    }

    /// Returns the underlying device buffer as an immutable borrow.
    pub fn buffer(&self) -> &DeviceBuffer<T> {
        self.buffer
    }

    /// Returns the underlying device buffer as a mutable borrow.
    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.buffer
    }
}

impl<'a, T> DeviceInOut<'a, T> {
    /// Creates an in-place-role borrow from a mutable device buffer.
    pub fn new(buffer: &'a mut DeviceBuffer<T>) -> Self {
        Self { buffer }
    }

    /// Returns the number of elements available through this in-place role.
    pub fn len(&self) -> usize {
        self.buffer.len
    }

    /// Returns true when this in-place role contains no elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.len == 0
    }

    /// Returns the raw device pointer as an immutable C pointer.
    pub fn as_const_ptr(&self) -> *const c_void {
        self.buffer.ptr.cast()
    }

    /// Returns the raw device pointer as a mutable C pointer.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.buffer.ptr.cast()
    }

    /// Returns the underlying device buffer as an immutable borrow.
    pub fn buffer(&self) -> &DeviceBuffer<T> {
        self.buffer
    }

    /// Returns the underlying device buffer as a mutable borrow.
    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        self.buffer
    }
}

pub(crate) fn check_cuda(call: &'static str, status: ffi::cudaError_t) -> Result<()> {
    if status == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda(call, status))
    }
}

pub(crate) fn check_cublas(call: &'static str, status: ffi::cublasStatus_t) -> Result<()> {
    if status == ffi::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cublas(call, status))
    }
}

pub(crate) fn max_shared_memory_per_block() -> Result<usize> {
    static MAX_SHARED_MEMORY_PER_BLOCK: OnceLock<usize> = OnceLock::new();
    if let Some(bytes) = MAX_SHARED_MEMORY_PER_BLOCK.get() {
        return Ok(*bytes);
    }

    let mut device = 0;
    let mut bytes = 0;
    unsafe {
        check_cuda("cudaGetDevice", ffi::cudaGetDevice(&mut device))?;
        check_cuda(
            "cudaDeviceGetAttribute(max shared memory per block)",
            ffi::cudaDeviceGetAttribute(
                &mut bytes,
                ffi::CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK,
                device,
            ),
        )?;
    }
    if bytes <= 0 {
        return Err(Error::Format {
            label: "CUDA shared memory attribute",
            detail: format!("expected a positive byte count, got {bytes}"),
        });
    }
    let bytes = bytes as usize;
    let _ = MAX_SHARED_MEMORY_PER_BLOCK.set(bytes);
    Ok(bytes)
}

/// Blocks until all previously submitted work on the current CUDA device has
/// completed.
///
/// This is useful for host-wall-clock smoke tests and the current benchmark
/// harness. CUDA event timing should be used for GPU-side elapsed time.
pub fn synchronize_device() -> Result<()> {
    unsafe { check_cuda("cudaDeviceSynchronize", ffi::cudaDeviceSynchronize()) }
}

/// Returns CUDA-visible free and total memory bytes for the current device.
pub fn device_memory_info() -> Result<(usize, usize)> {
    let mut free = 0;
    let mut total = 0;
    unsafe {
        check_cuda("cudaMemGetInfo", ffi::cudaMemGetInfo(&mut free, &mut total))?;
    }
    Ok((free, total))
}

/// Selects and initializes the CUDA device for the calling thread.
pub fn set_cuda_device(device: i32) -> Result<()> {
    unsafe { check_cuda("cudaSetDevice", ffi::cudaSetDevice(device)) }
}

/// Non-blocking CUDA stream suitable for graph capture and replay.
pub struct CudaStream {
    stream: ffi::cudaStream_t,
}

impl CudaStream {
    /// Creates a non-blocking CUDA stream.
    pub fn new_non_blocking() -> Result<Self> {
        let mut stream = null_mut();
        unsafe {
            check_cuda(
                "cudaStreamCreateWithFlags",
                ffi::cudaStreamCreateWithFlags(&mut stream, ffi::CUDA_STREAM_NON_BLOCKING),
            )?;
        }
        Ok(Self { stream })
    }

    /// Creates a blocking (default-stream-compatible) CUDA stream.
    pub fn new_blocking() -> Result<Self> {
        let mut stream = null_mut();
        unsafe {
            check_cuda("cudaStreamCreate", ffi::cudaStreamCreate(&mut stream))?;
        }
        Ok(Self { stream })
    }

    /// Blocks until all work submitted to this stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cuda(
                "cudaStreamSynchronize",
                ffi::cudaStreamSynchronize(self.stream),
            )
        }
    }

    /// Makes this stream wait until `event` has completed.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        unsafe {
            check_cuda(
                "cudaStreamWaitEvent",
                ffi::cudaStreamWaitEvent(self.stream, event.event, 0),
            )
        }
    }

    /// Begins CUDA graph capture on this stream.
    pub fn begin_capture(&self) -> Result<()> {
        unsafe {
            check_cuda(
                "cudaStreamBeginCapture",
                ffi::cudaStreamBeginCapture(self.stream, ffi::CUDA_STREAM_CAPTURE_MODE_RELAXED),
            )
        }
    }

    /// Ends CUDA graph capture on this stream and instantiates an executable
    /// graph.
    pub fn end_capture(&self) -> Result<CudaGraphExec> {
        let mut graph = null_mut();
        unsafe {
            check_cuda(
                "cudaStreamEndCapture",
                ffi::cudaStreamEndCapture(self.stream, &mut graph),
            )?;
        }
        CudaGraphExec::instantiate(graph)
    }

    /// Captures CUDA work submitted to this stream and returns an executable
    /// graph.
    pub fn capture<F>(&self, f: F) -> Result<CudaGraphExec>
    where
        F: FnOnce(&CudaStream) -> Result<()>,
    {
        unsafe {
            check_cuda(
                "cudaStreamBeginCapture",
                ffi::cudaStreamBeginCapture(self.stream, ffi::CUDA_STREAM_CAPTURE_MODE_RELAXED),
            )?;
        }

        let result = f(self);
        let mut graph = null_mut();
        let end_result = unsafe {
            check_cuda(
                "cudaStreamEndCapture",
                ffi::cudaStreamEndCapture(self.stream, &mut graph),
            )
        };

        result?;
        end_result?;
        CudaGraphExec::instantiate(graph)
    }

    pub(crate) fn as_raw(&self) -> ffi::cudaStream_t {
        self.stream
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = ffi::cudaStreamDestroy(self.stream);
            }
        }
    }
}

/// Executable CUDA graph captured from a [`CudaStream`].
pub struct CudaGraphExec {
    exec: ffi::cudaGraphExec_t,
    graph: ffi::cudaGraph_t,
}

impl CudaGraphExec {
    fn instantiate(graph: ffi::cudaGraph_t) -> Result<Self> {
        let mut exec = null_mut();
        unsafe {
            check_cuda(
                "cudaGraphInstantiate",
                ffi::cudaGraphInstantiate(&mut exec, graph, 0),
            )?;
        }
        Ok(Self { exec, graph })
    }

    /// Launches this graph on `stream`.
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check_cuda(
                "cudaGraphLaunch",
                ffi::cudaGraphLaunch(self.exec, stream.as_raw()),
            )
        }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                let _ = ffi::cudaGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = ffi::cudaGraphDestroy(self.graph);
            }
        }
    }
}

/// CUDA event used for device-side timing.
pub struct CudaEvent {
    event: ffi::cudaEvent_t,
}

impl CudaEvent {
    /// Creates a CUDA event with the runtime defaults.
    pub fn new() -> Result<Self> {
        let mut event = null_mut();
        unsafe {
            check_cuda("cudaEventCreate", ffi::cudaEventCreate(&mut event))?;
        }
        Ok(Self { event })
    }

    /// Creates an event for stream ordering without timestamp collection.
    pub fn new_sync() -> Result<Self> {
        let mut event = null_mut();
        unsafe {
            check_cuda(
                "cudaEventCreateWithFlags",
                ffi::cudaEventCreateWithFlags(&mut event, ffi::CUDA_EVENT_DISABLE_TIMING),
            )?;
        }
        Ok(Self { event })
    }

    /// Records the event on the default stream.
    pub fn record_default_stream(&self) -> Result<()> {
        unsafe {
            check_cuda(
                "cudaEventRecord",
                ffi::cudaEventRecord(self.event, null_mut()),
            )
        }
    }

    /// Records the event on `stream`.
    pub fn record_on_stream(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            check_cuda(
                "cudaEventRecord",
                ffi::cudaEventRecord(self.event, stream.as_raw()),
            )
        }
    }

    /// Blocks until this event has completed.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            check_cuda(
                "cudaEventSynchronize",
                ffi::cudaEventSynchronize(self.event),
            )
        }
    }

    /// Returns elapsed device time from `self` to `end`, in milliseconds.
    pub fn elapsed_ms_until(&self, end: &Self) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe {
            check_cuda(
                "cudaEventElapsedTime",
                ffi::cudaEventElapsedTime(&mut ms, self.event, end.event),
            )?;
        }
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                let _ = ffi::cudaEventDestroy(self.event);
            }
        }
    }
}

/// Owns a typed allocation in CUDA device memory.
///
/// The buffer frees its allocation with `cudaFree` on drop. It is intentionally
/// small: allocation, host-to-device initialization, zero initialization, and
/// device-to-host copy are enough for the current cuBLASLt experiments.
pub struct DeviceBuffer<T> {
    pub(crate) ptr: *mut T,
    len: usize,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocates device memory without initializing its contents.
    ///
    /// This is crate-private because callers must ensure every element is
    /// written before it can be observed by a kernel or copied to the host.
    pub(crate) fn uninitialized(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape {
                label: "device uninitialized allocation",
                expected: "len * element size without overflow".to_string(),
                actual: format!("len={len} element_size={}", size_of::<T>()),
            })?;
        let mut raw = null_mut();
        unsafe {
            check_cuda("cudaMalloc", ffi::cudaMalloc(&mut raw, bytes))?;
        }
        Ok(Self {
            ptr: raw.cast(),
            len,
        })
    }

    /// Allocates device memory and copies `values` into it.
    pub fn from_host(values: &[T]) -> Result<Self> {
        let mut raw = null_mut();
        let bytes = std::mem::size_of_val(values);
        unsafe {
            check_cuda("cudaMalloc", ffi::cudaMalloc(&mut raw, bytes))?;
            check_cuda(
                "cudaMemcpy(H2D)",
                ffi::cudaMemcpy(
                    raw,
                    values.as_ptr().cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
            )?;
        }
        Ok(Self {
            ptr: raw.cast(),
            len: values.len(),
        })
    }

    /// Allocates `len` elements of device memory and initializes them to zero.
    pub fn zeroed(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape {
                label: "device zeroed allocation",
                expected: "len * element size without overflow".to_string(),
                actual: format!("len={len} element_size={}", size_of::<T>()),
            })?;
        let mut raw = null_mut();
        unsafe {
            check_cuda("cudaMalloc", ffi::cudaMalloc(&mut raw, bytes))?;
            check_cuda("cudaMemset", ffi::cudaMemset(raw, 0, bytes))?;
            // cudaMemset may complete asynchronously on the default stream.
            // A non-blocking stream does not inherit the default-stream order,
            // so establish initialization before returning the allocation.
            check_cuda("cudaDeviceSynchronize", ffi::cudaDeviceSynchronize())?;
        }
        Ok(Self {
            ptr: raw.cast(),
            len,
        })
    }

    /// Synchronizes `stream` and copies the complete device allocation back to host memory.
    pub fn copy_to_host<'a>(&'a self, stream: &CudaStream) -> Result<HostRead<'a, T>> {
        self.copy_prefix_to_host(self.len, stream)
    }

    /// Synchronizes `stream` and copies the first `len` values back to host memory.
    pub fn copy_prefix_to_host<'a>(
        &'a self,
        len: usize,
        stream: &CudaStream,
    ) -> Result<HostRead<'a, T>> {
        if len > self.len {
            return Err(Error::Shape {
                label: "device prefix copy to host",
                expected: format!("at most {} values", self.len),
                actual: format!("{len} values"),
            });
        }
        stream.synchronize()?;
        let mut out = Vec::<T>::with_capacity(len);
        let bytes = len * size_of::<T>();
        unsafe {
            check_cuda(
                "cudaMemcpy(D2H)",
                ffi::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    self.ptr.cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
            )?;
            out.set_len(len);
        }
        Ok(HostRead {
            values: out,
            _device: PhantomData,
        })
    }

    /// Copies host values into this existing device allocation.
    pub fn copy_from_host(&mut self, values: &[T]) -> Result<()> {
        if values.len() != self.len {
            return Err(Error::Shape {
                label: "device copy from host",
                expected: format!("{} values", self.len),
                actual: format!("{} values", values.len()),
            });
        }
        let bytes = std::mem::size_of_val(values);
        unsafe {
            check_cuda(
                "cudaMemcpy(H2D existing)",
                ffi::cudaMemcpy(
                    self.ptr.cast(),
                    values.as_ptr().cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
            )
        }
    }

    /// Copies host values into the prefix of this existing device allocation.
    pub fn copy_prefix_from_host(&mut self, values: &[T]) -> Result<()> {
        if values.len() > self.len {
            return Err(Error::Shape {
                label: "device prefix copy from host",
                expected: format!("at most {} values", self.len),
                actual: format!("{} values", values.len()),
            });
        }
        if values.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::size_of_val(values);
        unsafe {
            check_cuda(
                "cudaMemcpy(H2D existing prefix)",
                ffi::cudaMemcpy(
                    self.ptr.cast(),
                    values.as_ptr().cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
            )
        }
    }

    /// Enqueues a device-to-device copy of the first `len` elements.
    pub fn copy_prefix_from_device_on_stream(
        &mut self,
        source: &Self,
        len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if len > self.len || len > source.len {
            return Err(Error::Shape {
                label: "device prefix copy",
                expected: format!(
                    "at most destination={} and source={} values",
                    self.len, source.len
                ),
                actual: format!("{len} values"),
            });
        }
        if len == 0 {
            return Ok(());
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape {
                label: "device prefix copy bytes",
                expected: "len * element size without overflow".to_string(),
                actual: format!("len={len} element_size={}", size_of::<T>()),
            })?;
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D prefix)",
                ffi::cudaMemcpyAsync(
                    self.ptr.cast(),
                    source.ptr.cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues a device-to-device copy between contiguous element ranges.
    pub fn copy_range_from_device_on_stream(
        &mut self,
        destination_offset: usize,
        source: &Self,
        source_offset: usize,
        len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let destination_end = destination_offset
            .checked_add(len)
            .ok_or_else(|| Error::Shape {
                label: "device range copy destination",
                expected: "offset + length without overflow".to_string(),
                actual: format!("offset={destination_offset} length={len}"),
            })?;
        let source_end = source_offset.checked_add(len).ok_or_else(|| Error::Shape {
            label: "device range copy source",
            expected: "offset + length without overflow".to_string(),
            actual: format!("offset={source_offset} length={len}"),
        })?;
        if destination_end > self.len || source_end > source.len {
            return Err(Error::Shape {
                label: "device range copy",
                expected: format!(
                    "destination end <= {} and source end <= {}",
                    self.len, source.len
                ),
                actual: format!(
                    "destination_end={destination_end} source_end={source_end} length={len}"
                ),
            });
        }
        if len == 0 {
            return Ok(());
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape {
                label: "device range copy bytes",
                expected: "length * element size without overflow".to_string(),
                actual: format!("length={len} element_size={}", size_of::<T>()),
            })?;
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D range)",
                ffi::cudaMemcpyAsync(
                    self.ptr.add(destination_offset).cast(),
                    source.ptr.add(source_offset).cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues a non-overlapping device-to-device copy within this allocation.
    pub fn copy_within_on_stream(
        &mut self,
        source_offset: usize,
        destination_offset: usize,
        len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let source_end = source_offset.checked_add(len).ok_or_else(|| Error::Shape {
            label: "device within-copy source",
            expected: "offset + length without overflow".to_string(),
            actual: format!("offset={source_offset} length={len}"),
        })?;
        let destination_end = destination_offset
            .checked_add(len)
            .ok_or_else(|| Error::Shape {
                label: "device within-copy destination",
                expected: "offset + length without overflow".to_string(),
                actual: format!("offset={destination_offset} length={len}"),
            })?;
        let overlaps = source_offset < destination_end && destination_offset < source_end;
        if source_end > self.len || destination_end > self.len || overlaps {
            return Err(Error::Shape {
                label: "device within-copy",
                expected: format!("non-overlapping ranges within {} values", self.len),
                actual: format!(
                    "source={source_offset}..{source_end} destination={destination_offset}..{destination_end}"
                ),
            });
        }
        if len == 0 {
            return Ok(());
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape {
                label: "device within-copy bytes",
                expected: "length * element size without overflow".to_string(),
                actual: format!("length={len} element_size={}", size_of::<T>()),
            })?;
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(D2D within)",
                ffi::cudaMemcpyAsync(
                    self.ptr.add(destination_offset).cast(),
                    self.ptr.add(source_offset).cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Copies host values into a contiguous element range of this allocation.
    pub fn copy_range_from_host(&mut self, element_offset: usize, values: &[T]) -> Result<()> {
        let end = element_offset
            .checked_add(values.len())
            .ok_or_else(|| Error::Shape {
                label: "device range copy from host",
                expected: "offset + length without overflow".to_string(),
                actual: format!("offset={element_offset} length={}", values.len()),
            })?;
        if end > self.len {
            return Err(Error::Shape {
                label: "device range copy from host",
                expected: format!("end <= {}", self.len),
                actual: format!("offset={element_offset} length={} end={end}", values.len()),
            });
        }
        let bytes = std::mem::size_of_val(values);
        unsafe {
            check_cuda(
                "cudaMemcpy(H2D element range)",
                ffi::cudaMemcpy(
                    self.ptr.add(element_offset).cast(),
                    values.as_ptr().cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
            )
        }
    }

    /// Enqueues a pinned-host copy into a contiguous element range on `stream`.
    pub fn copy_range_from_pinned_on_stream(
        &mut self,
        element_offset: usize,
        values: &PinnedHostBuffer<T>,
        stream: &CudaStream,
    ) -> Result<()> {
        let end = element_offset
            .checked_add(values.len)
            .ok_or_else(|| Error::Shape {
                label: "device asynchronous range copy",
                expected: "offset + length without overflow".to_string(),
                actual: format!("offset={element_offset} length={}", values.len),
            })?;
        if end > self.len {
            return Err(Error::Shape {
                label: "device asynchronous range copy",
                expected: format!("end <= {}", self.len),
                actual: format!("offset={element_offset} length={} end={end}", values.len),
            });
        }
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(H2D element range)",
                ffi::cudaMemcpyAsync(
                    self.ptr.add(element_offset).cast(),
                    values.ptr.cast(),
                    values.len * size_of::<T>(),
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Enqueues a byte range from a pinned allocation into this allocation.
    pub fn copy_bytes_from_pinned_range_on_stream(
        &mut self,
        device_byte_offset: usize,
        values: &PinnedHostBuffer<u8>,
        source_byte_offset: usize,
        bytes: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let device_bytes = self
            .len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::Shape {
                label: "device asynchronous byte range",
                expected: "device length in bytes without overflow".to_string(),
                actual: format!("len={} element_size={}", self.len, size_of::<T>()),
            })?;
        let device_end = device_byte_offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Shape {
                label: "device asynchronous byte range",
                expected: "device offset + length without overflow".to_string(),
                actual: format!("offset={device_byte_offset} bytes={bytes}"),
            })?;
        let source_end = source_byte_offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Shape {
                label: "pinned asynchronous byte range",
                expected: "source offset + length without overflow".to_string(),
                actual: format!("offset={source_byte_offset} bytes={bytes}"),
            })?;
        if device_end > device_bytes || source_end > values.len {
            return Err(Error::Shape {
                label: "asynchronous pinned byte range",
                expected: format!("device end <= {device_bytes}, source end <= {}", values.len),
                actual: format!("device_end={device_end} source_end={source_end}"),
            });
        }
        unsafe {
            check_cuda(
                "cudaMemcpyAsync(H2D byte range)",
                ffi::cudaMemcpyAsync(
                    self.ptr.cast::<u8>().add(device_byte_offset).cast(),
                    values.ptr.add(source_byte_offset).cast(),
                    bytes,
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Copies raw host bytes into a byte range of this existing allocation.
    pub fn copy_bytes_from_host(&mut self, byte_offset: usize, values: &[u8]) -> Result<()> {
        let allocation_bytes = self.device_bytes();
        let end = byte_offset
            .checked_add(values.len())
            .ok_or_else(|| Error::Shape {
                label: "device byte-range copy",
                expected: "offset + length without overflow".to_string(),
                actual: format!("offset={byte_offset} length={}", values.len()),
            })?;
        if end > allocation_bytes {
            return Err(Error::Shape {
                label: "device byte-range copy",
                expected: format!("end <= {allocation_bytes}"),
                actual: format!("offset={byte_offset} length={} end={end}", values.len()),
            });
        }
        unsafe {
            check_cuda(
                "cudaMemcpy(H2D byte range)",
                ffi::cudaMemcpy(
                    self.ptr.cast::<u8>().add(byte_offset).cast(),
                    values.as_ptr().cast(),
                    values.len(),
                    ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
            )
        }
    }

    /// Returns the number of elements in this device allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the number of device bytes owned by this allocation.
    pub fn device_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Returns true when this allocation contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the raw device pointer as an immutable C pointer.
    pub fn as_const_ptr(&self) -> *const c_void {
        self.ptr.cast()
    }

    /// Returns the raw device pointer as a mutable C pointer.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.cast()
    }

    /// Borrows this allocation as a kernel input role.
    pub fn input(&self) -> DeviceInput<'_, T> {
        DeviceInput::new(self)
    }

    /// Borrows this allocation as a kernel output role.
    pub fn output(&mut self) -> DeviceOutput<'_, T> {
        DeviceOutput::new(self)
    }

    /// Borrows this allocation as a kernel in-place role.
    pub fn inout(&mut self) -> DeviceInOut<'_, T> {
        DeviceInOut::new(self)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = ffi::cudaFree(self.ptr.cast());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaStream, DeviceBuffer};
    use crate::fill_f32_into_on_stream;

    #[test]
    fn zeroed_allocation_is_ready_for_a_non_blocking_stream() {
        let stream = CudaStream::new_non_blocking().expect("CUDA stream");
        for _ in 0..16 {
            let mut device = DeviceBuffer::zeroed(1 << 20).expect("zeroed device buffer");
            fill_f32_into_on_stream(device.output(), 7.0, &stream).expect("fill on stream");
            let values = device
                .copy_prefix_to_host(1, &stream)
                .expect("filled value");
            assert_eq!(values.as_slice(), [7.0]);
        }
    }

    #[test]
    fn device_prefix_copy_reads_only_the_requested_values() {
        let stream = CudaStream::new_blocking().expect("CUDA stream");
        let device = DeviceBuffer::from_host(&[1u32, 2, 3, 4]).expect("device buffer");

        assert_eq!(
            device
                .copy_prefix_to_host(2, &stream)
                .expect("prefix copy")
                .as_slice(),
            [1, 2]
        );
        let error = device
            .copy_prefix_to_host(5, &stream)
            .expect_err("oversized prefix must fail");
        assert!(error.to_string().contains("at most 4 values"));
    }

    #[test]
    fn host_prefix_copy_preserves_the_device_suffix() {
        let stream = CudaStream::new_blocking().expect("CUDA stream");
        let mut device = DeviceBuffer::from_host(&[1u32, 2, 3, 4]).expect("device buffer");

        device
            .copy_prefix_from_host(&[9, 8])
            .expect("host prefix copy");

        assert_eq!(
            device
                .copy_to_host(&stream)
                .expect("device copy")
                .as_slice(),
            [9, 8, 3, 4]
        );
    }
}
