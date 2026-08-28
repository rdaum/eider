//! CUPTI and NVPerf GPU-counter collection for diagnostic benchmark passes.

use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double};
use std::ptr::null_mut;

#[derive(Clone, Debug)]
/// One evaluated GPU performance counter metric.
pub struct GpuCounterMetric {
    /// Profiler metric name.
    pub name: String,
    /// Evaluated metric value for the profiled range.
    pub value: f64,
}

/// CUPTI/NVPerf range-profiler collector for diagnostic microbenchmark passes.
pub struct GpuCounterCollector {
    handle: *mut std::ffi::c_void,
    _metric_names: Vec<CString>,
    range_name: CString,
}

impl GpuCounterCollector {
    /// Creates a collector for `metrics` on the current CUDA context.
    pub fn new(metrics: &[&str], range_name: &str) -> Result<Self> {
        let metric_names = metrics
            .iter()
            .map(|metric| {
                CString::new(*metric).map_err(|_| format_error("GPU counter metric contains NUL"))
            })
            .collect::<Result<Vec<_>>>()?;
        let metric_ptrs = metric_names
            .iter()
            .map(|metric| metric.as_ptr())
            .collect::<Vec<_>>();
        let range_name = CString::new(range_name)
            .map_err(|_| format_error("GPU counter range name contains NUL"))?;

        let mut handle = null_mut();
        let mut error = ErrorBuffer::new();
        let status = unsafe {
            ffi::infer_gpu_counter_create(
                metric_ptrs.as_ptr(),
                metric_ptrs.len(),
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(format_error(error.message()));
        }
        Ok(Self {
            handle,
            _metric_names: metric_names,
            range_name,
        })
    }

    /// Starts one CUPTI user replay profiling pass and pushes the configured range.
    pub fn begin(&mut self) -> Result<()> {
        let mut error = ErrorBuffer::new();
        let status = unsafe {
            ffi::infer_gpu_counter_begin(
                self.handle,
                self.range_name.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(format_error(error.message()))
        }
    }

    /// Pops the current range, stops the pass, and returns whether all replay passes are complete.
    pub fn end(&mut self) -> Result<bool> {
        let mut all_passes_submitted = 0;
        let mut error = ErrorBuffer::new();
        let status = unsafe {
            ffi::infer_gpu_counter_end(
                self.handle,
                &mut all_passes_submitted,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(all_passes_submitted != 0)
        } else {
            Err(format_error(error.message()))
        }
    }

    /// Decodes collected counter data after all replay passes have completed.
    pub fn decode(&mut self) -> Result<Vec<GpuCounterMetric>> {
        let mut error = ErrorBuffer::new();
        let status =
            unsafe { ffi::infer_gpu_counter_decode(self.handle, error.as_mut_ptr(), error.len()) };
        if status != 0 {
            return Err(format_error(error.message()));
        }

        let count = unsafe { ffi::infer_gpu_counter_value_count(self.handle) };
        let mut metrics = Vec::with_capacity(count);
        for index in 0..count {
            let mut name: *const c_char = std::ptr::null();
            let mut value = 0.0 as c_double;
            let status =
                unsafe { ffi::infer_gpu_counter_value(self.handle, index, &mut name, &mut value) };
            if status != 0 || name.is_null() {
                return Err(format_error("GPU counter value read failed"));
            }
            let name = unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned();
            metrics.push(GpuCounterMetric { name, value });
        }
        Ok(metrics)
    }
}

impl Drop for GpuCounterCollector {
    fn drop(&mut self) {
        unsafe {
            ffi::infer_gpu_counter_destroy(self.handle);
        }
    }
}

struct ErrorBuffer {
    bytes: [c_char; 1024],
}

impl ErrorBuffer {
    fn new() -> Self {
        Self { bytes: [0; 1024] }
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.bytes.as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn message(&self) -> String {
        unsafe { CStr::from_ptr(self.bytes.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

fn format_error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "GPU counters",
        detail: detail.into(),
    }
}
