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
    coefficients: DeviceBuffer<f32>,
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
            coefficients: DeviceBuffer::zeroed(ROWS * HEADS * 2).unwrap(),
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
                eider_cuda::GemmPrecision::F32,
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
                self.coefficients.output(),
                1,
                ROWS,
                HEADS,
                DIM,
                GROUPS,
                STATE,
                self.dtype,
                eider_cuda::GemmPrecision::F32,
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
struct Ssd<const ROWS: usize> {
    recurrent: Mamba<ROWS>,
    workspace: eider_cuda::Nemotron3SsdWorkspace,
}
impl<const ROWS: usize> Ssd<ROWS> {
    fn run(&mut self) {
        let c = &mut self.recurrent;
        self.workspace
            .run(
                &c.projected,
                &c.convolved,
                &c.a_log,
                &c.d,
                &c.dt_bias,
                &c.norm,
                &c.table,
                1,
                &c.offsets,
                &c.lengths,
                1,
                ROWS,
                c.dtype,
                1e-5,
                c.output.output(),
                &c.stream,
            )
            .unwrap();
    }
}
impl<const ROWS: usize> BenchContext for Ssd<ROWS> {
    fn prepare(_: usize) -> Self {
        let bf = |x: f32| {
            f32::from_bits((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
        };
        let mut c = Mamba::new(MambaSsmDtype::Bf16);
        let x = bf(0.3);
        let b = bf(0.004);
        let cv = bf(0.01);
        let mut conv = vec![x; CONV];
        conv[INNER..INNER + GROUPS * STATE].fill(b);
        conv[INNER + GROUPS * STATE..].fill(cv);
        c.convolved = DeviceBuffer::from_host(&conv.repeat(ROWS)).unwrap();
        c.projected = DeviceBuffer::from_host(&vec![0.5; ROWS * PROJECTION]).unwrap();
        let mut context = Self {
            recurrent: c,
            workspace: eider_cuda::Nemotron3SsdWorkspace::new(ROWS, 1, HEADS, GROUPS).unwrap(),
        };
        context.run();
        // This uniform fixture has one scalar state per channel and an analytic chunk sum.
        let dt = (-0.5_f32).exp().ln_1p();
        let mut state = 0.125_f32;
        for begin in (0..ROWS).step_by(64) {
            let count = 64.min(ROWS - begin);
            let mut prefix = Vec::new();
            let mut sum = 0_f32;
            for _ in 0..count {
                sum -= dt;
                prefix.push(sum);
            }
            let mut contribution = 0_f32;
            for &value in &prefix {
                contribution = x.mul_add(bf(b * ((sum - value).exp() * dt)), contribution);
            }
            state = sum.exp().mul_add(state, contribution);
        }
        let bytes = context.recurrent.states[1]
            .copy_to_host(&context.recurrent.stream)
            .unwrap();
        for v in bytes.as_chunks::<2>().0 {
            let actual = f32::from_bits(u32::from(u16::from_ne_bytes(*v)) << 16);
            assert!(
                (actual - bf(state)).abs() < 2e-5,
                "SSD scalar state: {actual} vs {}",
                bf(state)
            );
        }
        for &value in context
            .recurrent
            .output
            .copy_to_host(&context.recurrent.stream)
            .unwrap()
            .iter()
        {
            assert!((value - 1.).abs() <= 0.008, "SSD normalized output {value}");
        }
        context
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}
fn sample_ssd<const ROWS: usize>(c: &mut Ssd<ROWS>, count: usize, _: usize) -> BenchSampleResult {
    c.recurrent
        .start
        .record_on_stream(&c.recurrent.stream)
        .unwrap();
    for _ in 0..count {
        c.run();
    }
    c.recurrent
        .stop
        .record_on_stream(&c.recurrent.stream)
        .unwrap();
    c.recurrent.stop.synchronize().unwrap();
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        c.recurrent
            .start
            .elapsed_ms_until(&c.recurrent.stop)
            .unwrap() as f64
            / count as f64,
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
            runner.group::<Ssd<1024>>("BF16 SSD 1024 tokens", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("chunked tensor cores", sample_ssd::<1024>);
            });
            runner.group::<Ssd<2048>>("BF16 SSD 2048 tokens", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("chunked tensor cores", sample_ssd::<2048>);
            });
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
