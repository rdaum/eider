use micromeasure::{MeasurementBackend, MetricValue};
use nvfp4::CudaEvent;
use std::time::{Duration, Instant};

pub(crate) struct CudaEventBackend {
    start: CudaEvent,
    stop: CudaEvent,
    host_start: Option<Instant>,
    host_elapsed: Duration,
    device_ms: f64,
    bytes_per_op: u64,
    flops_per_op: u64,
}

impl CudaEventBackend {
    pub(crate) fn new(bytes_per_op: u64, flops_per_op: u64) -> Self {
        Self {
            start: CudaEvent::new().expect("create CUDA start event"),
            stop: CudaEvent::new().expect("create CUDA stop event"),
            host_start: None,
            host_elapsed: Duration::ZERO,
            device_ms: 0.0,
            bytes_per_op,
            flops_per_op,
        }
    }
}

impl MeasurementBackend for CudaEventBackend {
    fn begin(&mut self) {
        self.host_start = Some(Instant::now());
        self.start
            .record_default_stream()
            .expect("record CUDA start event");
    }

    fn end(&mut self) {
        self.stop
            .record_default_stream()
            .expect("record CUDA stop event");
        self.stop
            .synchronize()
            .expect("synchronize CUDA stop event");
        self.device_ms = self
            .start
            .elapsed_ms_until(&self.stop)
            .expect("compute CUDA event elapsed time") as f64;
        self.host_elapsed = self
            .host_start
            .take()
            .expect("CUDA event backend begin before end")
            .elapsed();
    }

    fn collect(
        &mut self,
        _host_elapsed: Duration,
        ops: u64,
        _chunk_index: usize,
        results: &mut micromeasure::bench::Results,
        metrics: &mut Vec<MetricValue>,
    ) {
        let device_duration = Duration::from_secs_f64(self.device_ms / 1_000.0);
        let device_seconds = device_duration.as_secs_f64().max(f64::MIN_POSITIVE);
        let host_overhead = self.host_elapsed.saturating_sub(device_duration);
        let total_bytes = self.bytes_per_op.saturating_mul(ops);
        let total_flops = self.flops_per_op.saturating_mul(ops);

        results.duration = device_duration;
        results.iterations = ops;
        results.chunks_executed = 1;

        metrics.push(
            MetricValue::duration_ms("cuda_event_ms", device_duration)
                .with_display_name("CUDA event"),
        );
        metrics.push(
            MetricValue::duration_ms("host_overhead_ms", host_overhead)
                .with_display_name("Host overhead"),
        );
        metrics.push(
            MetricValue::bandwidth_gib_s("gpu_gib_s", total_bytes, device_seconds)
                .with_display_name("GPU bandwidth"),
        );
        metrics.push(
            MetricValue::throughput_tflops("gpu_tflops", total_flops, device_seconds)
                .with_display_name("GPU throughput"),
        );
    }

    fn measurement_label(&self) -> &'static str {
        "timing + CUDA events"
    }

    fn emits_cpu_diagnostics(&self) -> bool {
        false
    }
}
