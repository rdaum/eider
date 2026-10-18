//! Dory causal attention, checked against the existing F32 GQA reference.
use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, DoryBranchPlan, dory_attention,
    dory_attention_workspace_len, prefill_gqa_attention_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, run_benchmark_main,
};
use std::time::Duration;
const Q: usize = 16;
const K: usize = 4;
struct Bench<const ROWS: usize, const START: usize, const WINDOW: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    ck: DeviceBuffer<u16>,
    cv: DeviceBuffer<u16>,
    partial: DeviceBuffer<f32>,
    out: DeviceBuffer<u16>,
    pos: DeviceBuffer<u32>,
}
impl<const ROWS: usize, const START: usize, const WINDOW: usize> Bench<ROWS, START, WINDOW> {
    fn run(&mut self) {
        dory_attention(
            &self.q,
            &self.k,
            &self.v,
            self.ck.output(),
            self.cv.output(),
            self.partial.output(),
            self.out.output(),
            &self.pos,
            ROWS,
            Q,
            K,
            START + ROWS,
            WINDOW,
            &self.stream,
        )
        .unwrap();
    }
}
impl<const ROWS: usize, const START: usize, const WINDOW: usize> BenchContext
    for Bench<ROWS, START, WINDOW>
{
    fn prepare(_: usize) -> Self {
        let host = |n: usize, m: usize| {
            (0..n)
                .map(|i| {
                    eider_cuda::format::bf16_to_f32(eider_cuda::format::f32_to_bf16(
                        ((i * m % 101) as f32 - 50.) / 100.,
                    ))
                })
                .collect::<Vec<_>>()
        };
        let bf = |v: &[f32]| {
            v.iter()
                .copied()
                .map(eider_cuda::format::f32_to_bf16)
                .collect::<Vec<_>>()
        };
        let keys = host((START + ROWS) * K * 256, 17);
        let values = host((START + ROWS) * K * 256, 7);
        let mut b = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            q: DeviceBuffer::from_host(&bf(&host(ROWS * Q * 256, 13))).unwrap(),
            k: DeviceBuffer::from_host(&bf(&keys[START * K * 256..])).unwrap(),
            v: DeviceBuffer::from_host(&bf(&values[START * K * 256..])).unwrap(),
            ck: DeviceBuffer::from_host(
                &keys
                    .iter()
                    .map(|&x| eider_cuda::format::f32_to_bf16(x))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            cv: DeviceBuffer::from_host(
                &values
                    .iter()
                    .map(|&x| eider_cuda::format::f32_to_bf16(x))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            partial: DeviceBuffer::zeroed(dory_attention_workspace_len(
                ROWS,
                Q,
                START + ROWS,
                WINDOW,
            ))
            .unwrap(),
            out: DeviceBuffer::zeroed(ROWS * Q * 256).unwrap(),
            pos: DeviceBuffer::from_host(&[START as u32]).unwrap(),
        };
        b.run();
        let mut expected = DeviceBuffer::zeroed(ROWS * Q * 256).unwrap();
        let expired = if WINDOW > 0 {
            assert_eq!(ROWS, 1);
            (START + ROWS).saturating_sub(WINDOW)
        } else {
            0
        };
        let reference_keys = DeviceBuffer::from_host(&keys[expired * K * 256..]).unwrap();
        let reference_values = DeviceBuffer::from_host(&values[expired * K * 256..]).unwrap();
        prefill_gqa_attention_f32_into_on_stream(
            &DeviceBuffer::from_host(&host(ROWS * Q * 256, 13)).unwrap(),
            &reference_keys,
            &reference_values,
            expected.output(),
            ROWS,
            START - expired,
            Q,
            K,
            256,
            &b.stream,
        )
        .unwrap();
        let expected = expected.copy_to_host(&b.stream).unwrap();
        let actual = b.out.copy_to_host(&b.stream).unwrap();
        let max = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (eider_cuda::format::bf16_to_f32(a) - b).abs())
            .fold(0., f32::max);
        assert!(max < 0.002, "attention max error {max}");
        b
    }
    fn chunk_size() -> Option<usize> {
        Some(3)
    }
}
fn sample<const ROWS: usize, const START: usize, const WINDOW: usize>(
    b: &mut Bench<ROWS, START, WINDOW>,
    n: usize,
    _: usize,
) -> BenchSampleResult {
    b.start.record_on_stream(&b.stream).unwrap();
    for _ in 0..n {
        b.run();
    }
    b.stop.record_on_stream(&b.stream).unwrap();
    b.stop.synchronize().unwrap();
    BenchSampleResult::operations(n as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        b.start.elapsed_ms_until(&b.stop).unwrap() as f64 / n as f64,
        "ms",
    ))
}
// Mixed lengths exercise query tails, page boundaries, and sibling isolation.
struct BranchBench<const PREFIX: usize, const WINDOW: usize, const FUSED: bool = false> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    plan: DoryBranchPlan,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    ck: DeviceBuffer<u16>,
    cv: DeviceBuffer<u16>,
    out: DeviceBuffer<u16>,
    scratch: DeviceBuffer<f32>,
    serial: Vec<SerialBranch>,
    prefix: Option<SerialBranch>,
    zero: DeviceBuffer<u32>,
    pos: DeviceBuffer<u32>,
}
struct SerialBranch {
    len: usize,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    ck: DeviceBuffer<u16>,
    cv: DeviceBuffer<u16>,
    out: DeviceBuffer<u16>,
    scratch: DeviceBuffer<f32>,
}
impl<const PREFIX: usize, const WINDOW: usize, const FUSED: bool>
    BranchBench<PREFIX, WINDOW, FUSED>
{
    fn packed(&mut self) {
        self.plan
            .attention(
                &self.q,
                &self.k,
                &self.v,
                self.ck.output(),
                self.cv.output(),
                self.scratch.output(),
                self.out.output(),
                Q,
                K,
                WINDOW,
                &self.stream,
            )
            .unwrap();
    }
    fn serial(&mut self) {
        if let Some(b) = &mut self.prefix {
            dory_attention(
                &b.q,
                &b.k,
                &b.v,
                b.ck.output(),
                b.cv.output(),
                b.scratch.output(),
                b.out.output(),
                &self.zero,
                PREFIX,
                Q,
                K,
                PREFIX,
                WINDOW,
                &self.stream,
            )
            .unwrap();
        }
        for b in &mut self.serial {
            dory_attention(
                &b.q,
                &b.k,
                &b.v,
                b.ck.output(),
                b.cv.output(),
                b.scratch.output(),
                b.out.output(),
                &self.pos,
                b.len,
                Q,
                K,
                PREFIX + b.len,
                WINDOW,
                &self.stream,
            )
            .unwrap();
        }
    }
}
impl<const PREFIX: usize, const WINDOW: usize, const FUSED: bool> BenchContext
    for BranchBench<PREFIX, WINDOW, FUSED>
{
    fn prepare(_: usize) -> Self {
        let lengths = [1, 7, 16, 33, 48, 65, 80, 95];
        let plan = DoryBranchPlan::new(PREFIX, &lengths, FUSED).unwrap();
        let rows = plan.rows();
        let make = |n: usize, m: usize| {
            (0..n)
                .map(|i| eider_cuda::format::f32_to_bf16(((i * m % 101) as f32 - 50.) / 100.))
                .collect::<Vec<_>>()
        };
        let q = make(rows * Q * 256, 13);
        let mut k = make(rows * K * 256, 17);
        let mut v = make(rows * K * 256, 7);
        let pk = make(PREFIX * K * 256, 23);
        let pv = make(PREFIX * K * 256, 29);
        let mut ck = vec![eider_cuda::format::f32_to_bf16(5.); plan.cache_capacity() * K * 256];
        let mut cv = ck.clone();
        if FUSED {
            k[..pk.len()].copy_from_slice(&pk);
            v[..pv.len()].copy_from_slice(&pv);
        } else {
            ck[..pk.len()].copy_from_slice(&pk);
            cv[..pv.len()].copy_from_slice(&pv);
        }
        let prefix = FUSED.then(|| SerialBranch {
            len: PREFIX,
            q: DeviceBuffer::from_host(&q[..PREFIX * Q * 256]).unwrap(),
            k: DeviceBuffer::from_host(&pk).unwrap(),
            v: DeviceBuffer::from_host(&pv).unwrap(),
            ck: DeviceBuffer::zeroed(PREFIX * K * 256).unwrap(),
            cv: DeviceBuffer::zeroed(PREFIX * K * 256).unwrap(),
            out: DeviceBuffer::zeroed(PREFIX * Q * 256).unwrap(),
            scratch: DeviceBuffer::zeroed(dory_attention_workspace_len(PREFIX, Q, PREFIX, WINDOW))
                .unwrap(),
        });
        let mut serial = Vec::new();
        let mut offset = if FUSED { PREFIX } else { 0 };
        for len in lengths {
            let mut keys = pk.clone();
            keys.extend_from_slice(&k[offset * K * 256..(offset + len) * K * 256]);
            let mut values = pv.clone();
            values.extend_from_slice(&v[offset * K * 256..(offset + len) * K * 256]);
            serial.push(SerialBranch {
                len,
                q: DeviceBuffer::from_host(&q[offset * Q * 256..(offset + len) * Q * 256]).unwrap(),
                k: DeviceBuffer::from_host(&k[offset * K * 256..(offset + len) * K * 256]).unwrap(),
                v: DeviceBuffer::from_host(&v[offset * K * 256..(offset + len) * K * 256]).unwrap(),
                ck: DeviceBuffer::from_host(&keys).unwrap(),
                cv: DeviceBuffer::from_host(&values).unwrap(),
                out: DeviceBuffer::zeroed(len * Q * 256).unwrap(),
                scratch: DeviceBuffer::zeroed(dory_attention_workspace_len(
                    len,
                    Q,
                    PREFIX + len,
                    WINDOW,
                ))
                .unwrap(),
            });
            offset += len;
        }
        let mut b = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            plan,
            q: DeviceBuffer::from_host(&q).unwrap(),
            k: DeviceBuffer::from_host(&k).unwrap(),
            v: DeviceBuffer::from_host(&v).unwrap(),
            ck: DeviceBuffer::from_host(&ck).unwrap(),
            cv: DeviceBuffer::from_host(&cv).unwrap(),
            out: DeviceBuffer::zeroed(rows * Q * 256).unwrap(),
            scratch: DeviceBuffer::zeroed(rows * Q).unwrap(),
            serial,
            prefix,
            zero: DeviceBuffer::from_host(&[0u32]).unwrap(),
            pos: DeviceBuffer::from_host(&[PREFIX as u32]).unwrap(),
        };
        b.serial();
        b.packed();
        let actual = b.out.copy_to_host(&b.stream).unwrap();
        let mut offset = 0;
        for branch in b.prefix.iter().chain(&b.serial) {
            let expected = branch.out.copy_to_host(&b.stream).unwrap();
            let max = actual[offset..offset + expected.len()]
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| {
                    (eider_cuda::format::bf16_to_f32(a) - eider_cuda::format::bf16_to_f32(b)).abs()
                })
                .fold(0., f32::max);
            assert!(
                max < 0.002,
                "packed attention prefix={PREFIX} window={WINDOW} branch_rows={} max_error={max}",
                branch.len
            );
            offset += expected.len();
        }
        // Check the fused branch readout against the existing selected-label path.
        let width = 2560;
        let hidden = DeviceBuffer::from_host(&make(rows * width, 11)).unwrap();
        let weights = DeviceBuffer::from_host(&make(64 * width, 19)).unwrap();
        let scales = DeviceBuffer::from_host(&vec![0.7f32; 64]).unwrap();
        let mut logits = DeviceBuffer::zeroed(64 * lengths.len()).unwrap();
        b.plan
            .logits(
                &hidden,
                &weights,
                &scales,
                logits.output(),
                width,
                &b.stream,
            )
            .unwrap();
        let actual = logits.copy_to_host(&b.stream).unwrap();
        let mut last = DeviceBuffer::zeroed(width).unwrap();
        let mut reference = DeviceBuffer::zeroed(64).unwrap();
        let mut offset = if FUSED { PREFIX } else { 0 };
        for (i, len) in lengths.into_iter().enumerate() {
            offset += len;
            eider_cuda::copy_bf16_row_to_f32_into_on_stream(
                rows,
                width,
                offset - 1,
                &hidden,
                last.output(),
                &b.stream,
            )
            .unwrap();
            eider_cuda::bf16_linear_logits_f32_into_on_stream(
                &last,
                &weights,
                reference.output(),
                64,
                width,
                &b.stream,
            )
            .unwrap();
            eider_cuda::dory_scale_logits(reference.output(), &scales, &b.stream).unwrap();
            let reference = reference.copy_to_host(&b.stream).unwrap();
            for (&a, &r) in actual[i * 64..(i + 1) * 64].iter().zip(reference.iter()) {
                assert!(
                    (a - r).abs() < 1e-5,
                    "branch selected-label readout {a} != {r}"
                );
            }
        }
        assert_eq!(
            &*b.ck.copy_prefix_to_host(pk.len(), &b.stream).unwrap(),
            &pk
        );
        assert_eq!(
            &*b.cv.copy_prefix_to_host(pv.len(), &b.stream).unwrap(),
            &pv
        );
        b
    }
    fn chunk_size() -> Option<usize> {
        Some(3)
    }
}
fn branch_sample<
    const PREFIX: usize,
    const WINDOW: usize,
    const FUSED: bool,
    const PACKED: bool,
>(
    b: &mut BranchBench<PREFIX, WINDOW, FUSED>,
    n: usize,
    _: usize,
) -> BenchSampleResult {
    b.start.record_on_stream(&b.stream).unwrap();
    for _ in 0..n {
        if PACKED {
            b.packed();
        } else {
            b.serial();
        }
    }
    b.stop.record_on_stream(&b.stream).unwrap();
    b.stop.synchronize().unwrap();
    BenchSampleResult::operations(n as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        b.start.elapsed_ms_until(&b.stop).unwrap() as f64 / n as f64,
        "ms",
    ))
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("dory-attention".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: false,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(50),
                benchmark_duration: Duration::from_millis(200),
                min_samples: 3,
                max_samples: 5,
            },
            ..Default::default()
        },
        |runner| {
            if DoryBranchPlan::supported() {
                runner.group::<BranchBench<163, 0, true>>("prefix plus 8 branches", |g| {
                    g.bench_sample("combined", branch_sample::<163, 0, true, true>);
                    g.bench_sample("separate", branch_sample::<163, 0, true, false>);
                });
                runner.group::<BranchBench<128, 64, true>>("prefix plus branches / sliding", |g| {
                    g.bench_sample("combined", branch_sample::<128, 64, true, true>);
                    g.bench_sample("separate", branch_sample::<128, 64, true, false>);
                });
                runner.group::<BranchBench<163, 0>>("8 branches / partial prefix page", |g| {
                    g.bench_sample("packed", branch_sample::<163, 0, false, true>);
                    g.bench_sample("serial", branch_sample::<163, 0, false, false>);
                });
                runner.group::<BranchBench<1024, 64>>("8 branches / sliding across pages", |g| {
                    g.bench_sample("packed", branch_sample::<1024, 64, false, true>);
                    g.bench_sample("serial", branch_sample::<1024, 64, false, false>);
                });
            }

            runner.group::<Bench<128, 1024, 0>>("128 queries / 1024 prefix", |group| {
                group.bench_sample("attention", sample);
            });
            runner.group::<Bench<256, 20480, 0>>("256 queries / 20480 prefix", |group| {
                group.bench_sample("attention", sample);
            });
            runner.group::<Bench<1, 20480, 0>>("decode / 20480 prefix", |group| {
                group.bench_sample("attention", sample);
            });
            runner.group::<Bench<1, 20480, 1024>>("decode / sliding 1024", |group| {
                group.bench_sample("attention", sample);
            });
        },
    );
}
