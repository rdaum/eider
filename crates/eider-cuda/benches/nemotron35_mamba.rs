use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, MambaSsmDtype,
    nemotron3_mamba_state_update_f32_chunks_into_on_stream,
    nemotron3_mamba_state_update_f32_chunks_snapshot_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;
const HEADS: usize = 128;
const DIM: usize = 64;
const GROUPS: usize = 8;
const STATE: usize = 128;
const INNER: usize = HEADS * DIM;
const CONV: usize = INNER + GROUPS * STATE * 2;
const PROJECTION: usize = INNER + CONV + HEADS;
struct Mamba<const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    projected: DeviceBuffer<f32>,
    convolved: DeviceBuffer<f32>,
    a_log: DeviceBuffer<u16>,
    d: DeviceBuffer<u16>,
    dt_bias: DeviceBuffer<u16>,
    norm: DeviceBuffer<u16>,
    states: [DeviceBuffer<u8>; 2],
    table: DeviceBuffer<DeviceAddress<u8>>,
    offsets: DeviceBuffer<u32>,
    lengths: DeviceBuffer<u32>,
    snapshots: DeviceBuffer<u8>,
    output: DeviceBuffer<f32>,
    dtype: MambaSsmDtype,
}
impl<const ROWS: usize> Mamba<ROWS> {
    fn new(dtype: MambaSsmDtype) -> Self {
        let initial = match dtype {
            MambaSsmDtype::F32 => 0.125_f32.to_le_bytes().repeat(INNER * STATE),
            MambaSsmDtype::Bf16 => 0x3e00_u16.to_le_bytes().repeat(INNER * STATE),
        };
        let states = [
            DeviceBuffer::from_host(&initial).unwrap(),
            DeviceBuffer::from_host(&initial).unwrap(),
        ];
        Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            projected: DeviceBuffer::from_host(
                &(0..ROWS * PROJECTION)
                    .map(|i| (i % 127) as f32 / 127. - 0.5)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            convolved: DeviceBuffer::from_host(
                &(0..ROWS * CONV)
                    .map(|i| (i * 17 % 257) as f32 / 257. - 0.5)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            a_log: DeviceBuffer::from_host(&[0; HEADS]).unwrap(),
            d: DeviceBuffer::from_host(&[0x3f00; HEADS]).unwrap(),
            dt_bias: DeviceBuffer::from_host(&[0xbf80; HEADS]).unwrap(),
            norm: DeviceBuffer::from_host(&[0x3f80; INNER]).unwrap(),
            table: DeviceBuffer::from_host(
                &states
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            states,
            offsets: DeviceBuffer::from_host(&[0]).unwrap(),
            lengths: DeviceBuffer::from_host(&[ROWS as u32]).unwrap(),
            snapshots: DeviceBuffer::zeroed(INNER * STATE * dtype.element_bytes()).unwrap(),
            output: DeviceBuffer::zeroed(ROWS * INNER).unwrap(),
            dtype,
        }
    }
    fn run(&mut self, reference: bool) {
        if reference {
            nemotron3_mamba_state_update_f32_chunks_snapshot_into_on_stream(
                &self.projected,
                &self.convolved,
                &self.a_log,
                &self.d,
                &self.dt_bias,
                &self.norm,
                &self.table,
                0,
                &self.offsets,
                &self.lengths,
                self.output.output(),
                self.snapshots.output(),
                1,
                ROWS,
                1,
                HEADS,
                DIM,
                GROUPS,
                STATE,
                self.dtype,
                1e-4,
                1e-5,
                &self.stream,
            )
            .unwrap();
        } else {
            nemotron3_mamba_state_update_f32_chunks_into_on_stream(
                &self.projected,
                &self.convolved,
                &self.a_log,
                &self.d,
                &self.dt_bias,
                &self.norm,
                &self.table,
                1,
                &self.offsets,
                &self.lengths,
                self.output.output(),
                1,
                ROWS,
                HEADS,
                DIM,
                GROUPS,
                STATE,
                self.dtype,
                1e-4,
                1e-5,
                &self.stream,
            )
            .unwrap();
        }
    }
    fn validate(&mut self) {
        for _ in 0..2 {
            self.run(true);
            let reference = self.output.copy_to_host(&self.stream).unwrap().to_vec();
            self.run(false);
            let actual = self.output.copy_to_host(&self.stream).unwrap();
            assert_eq!(&*actual, &reference, "chunk output");
            assert_eq!(
                self.states[0].copy_to_host(&self.stream).unwrap(),
                self.states[1].copy_to_host(&self.stream).unwrap(),
                "state across chunks"
            );
        }
    }
}
impl<const ROWS: usize> BenchContext for Mamba<ROWS> {
    fn prepare(_: usize) -> Self {
        let mut context = Self::new(MambaSsmDtype::F32);
        context.validate();
        context
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}
fn sample<const ROWS: usize, const REFERENCE: bool>(
    c: &mut Mamba<ROWS>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    c.start.record_on_stream(&c.stream).unwrap();
    for _ in 0..count {
        c.run(REFERENCE);
    }
    c.stop.record_on_stream(&c.stream).unwrap();
    c.stop.synchronize().unwrap();
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        c.start.elapsed_ms_until(&c.stop).unwrap() as f64 / count as f64,
        "ms/call",
    ))
}
fn main() {
    Mamba::<17>::new(MambaSsmDtype::Bf16).validate();
    Mamba::<17>::new(MambaSsmDtype::F32).validate();
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nemotron35-mamba".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(10),
                benchmark_duration: Duration::from_millis(30),
                min_samples: 3,
                max_samples: 3,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Mamba<128>>("Mamba 128 tokens", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("stored state with initial snapshot", sample::<128, true>);
                g.bench_sample("register state", sample::<128, false>);
            });
            runner.group::<Mamba<1024>>("Mamba 1024 tokens", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("stored state with initial snapshot", sample::<1024, true>);
                g.bench_sample("register state", sample::<1024, false>);
            });
        },
    );
}
