use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaEvent, CudaStream, DeviceBuffer, Fp4TnMatmulPlan, GemmShape,
    Nvfp4Matrix, TernaryG64ActivationWorkspace, TernaryG64Matrix, TernaryG64PackedLinear,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream, nvfp4_w4a16_matvec_f32_into_on_stream,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const HIDDEN: usize = 4_096;
const KV_WIDTH: usize = 1_024;
const QKV_WIDTH: usize = HIDDEN + 2 * KV_WIDTH;
const INTERMEDIATE: usize = 12_288;
const GATE_UP: usize = INTERMEDIATE * 2;
const WORKSPACE_LIMIT: u64 = 32 * 1024 * 1024;

struct TernaryG64LinearBench<const BATCH: usize, const ROWS: usize, const COLS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    matrix: TernaryG64Matrix,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    workspace: TernaryG64ActivationWorkspace,
}

struct TernaryG64Bf16Bench<const BATCH: usize, const ROWS: usize, const COLS: usize> {
    lt: CublasLt,
    plan: Bf16TnMatmulPlan,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    matrix: TernaryG64Matrix,
    input: DeviceBuffer<f32>,
    input_bf16: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
}

struct Nvfp4W4A16Bench<const ROWS: usize, const COLS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    packed_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    weight_scale_2: f32,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

struct Nvfp4W4A4Bench<const BATCH: usize, const ROWS: usize, const COLS: usize> {
    lt: CublasLt,
    plan: Fp4TnMatmulPlan,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    matrix: TernaryG64Matrix,
    input: DeviceBuffer<f32>,
    activation: Nvfp4Matrix,
    output: DeviceBuffer<f32>,
}

struct BonsaiLayerBench<Q, O, G, D> {
    qkv: Q,
    output: O,
    gate_up: G,
    down: D,
}

trait LayerProjection: Sized {
    fn prepare_projection() -> Self;
    fn time_once(&mut self) -> f64;
    fn resident_weight_bytes(&self) -> usize;
    fn retain_output(&self);
}

impl<Q, O, G, D> BenchContext for BonsaiLayerBench<Q, O, G, D>
where
    Q: LayerProjection,
    O: LayerProjection,
    G: LayerProjection,
    D: LayerProjection,
{
    fn prepare(_num_chunks: usize) -> Self {
        Self {
            qkv: Q::prepare_projection(),
            output: O::prepare_projection(),
            gate_up: G::prepare_projection(),
            down: D::prepare_projection(),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize>
    TernaryG64LinearBench<BATCH, ROWS, COLS>
{
    fn enqueue(&mut self) {
        self.matrix
            .run_f32_batch_into_on_stream(
                self.input.input(),
                self.output.output(),
                BATCH,
                &mut self.workspace,
                &self.stream,
            )
            .expect("ternary g64 projection");
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> BenchContext
    for TernaryG64LinearBench<BATCH, ROWS, COLS>
{
    fn prepare(_num_chunks: usize) -> Self {
        let packed = synthetic_linear(ROWS, COLS);
        let input_host = synthetic_input(BATCH, COLS);
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut context = Self {
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            matrix: TernaryG64Matrix::from_packed(&packed).expect("matrix"),
            input,
            output: DeviceBuffer::zeroed(BATCH * ROWS).expect("output"),
            workspace: TernaryG64ActivationWorkspace::new(BATCH, COLS).expect("workspace"),
            stream,
        };
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download");
        verify_selected_outputs(&packed, &input_host, &actual, BATCH);
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(if BATCH == 1 { 100 } else { 1 })
    }
}

impl<const ROWS: usize, const COLS: usize> Nvfp4W4A16Bench<ROWS, COLS> {
    fn enqueue(&mut self) {
        nvfp4_w4a16_matvec_f32_into_on_stream(
            &self.input,
            &self.packed_weight,
            &self.weight_scale,
            self.output.output(),
            ROWS,
            COLS,
            self.weight_scale_2,
            &self.stream,
        )
        .expect("NVFP4 W4A16 projection");
    }
}

impl<const ROWS: usize, const COLS: usize> BenchContext for Nvfp4W4A16Bench<ROWS, COLS> {
    fn prepare(_num_chunks: usize) -> Self {
        let packed = synthetic_linear(ROWS, COLS);
        let weight = synthetic_nvfp4(&packed);
        let input_host = synthetic_input(1, COLS);
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut context = Self {
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            packed_weight: DeviceBuffer::from_host(&weight.packed_weight)
                .expect("NVFP4 packed weight"),
            weight_scale: DeviceBuffer::from_host(&weight.weight_scale)
                .expect("NVFP4 weight scale"),
            weight_scale_2: weight.weight_scale_2,
            input: DeviceBuffer::from_host(&input_host).expect("input"),
            output: DeviceBuffer::zeroed(ROWS).expect("output"),
            stream,
        };
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download");
        verify_nvfp4_selected_outputs(&weight, &input_host, &actual, 1);
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(100)
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> Nvfp4W4A4Bench<BATCH, ROWS, COLS> {
    fn enqueue(&mut self) {
        self.matrix
            .run_f32_batch_nvfp4_into_on_stream(
                &self.lt,
                &self.plan,
                self.input.input(),
                &mut self.activation,
                self.output.inout(),
                BATCH,
                &self.stream,
            )
            .expect("NVFP4 W4A4 projection");
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> BenchContext
    for Nvfp4W4A4Bench<BATCH, ROWS, COLS>
{
    fn prepare(_num_chunks: usize) -> Self {
        let packed = synthetic_linear(ROWS, COLS);
        let host_weight = synthetic_nvfp4(&packed);
        let input_host = synthetic_input(BATCH, COLS);
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let lt = CublasLt::new().expect("cuBLASLt");
        let matrix = TernaryG64Matrix::from_packed_with_nvfp4_prefill(&packed)
            .expect("hybrid ternary/NVFP4 matrix");
        let activation =
            Nvfp4Matrix::zeroed_col_major(COLS, BATCH).expect("NVFP4 activation storage");
        let output = DeviceBuffer::zeroed(ROWS * BATCH).expect("NVFP4 output");
        let plan = matrix
            .new_f32_batch_nvfp4_plan(&lt, &activation, BATCH, WORKSPACE_LIMIT)
            .expect("NVFP4 W4A4 plan");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut context = Self {
            lt,
            plan,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            matrix,
            input,
            activation,
            output,
            stream,
        };

        let mut reference = DeviceBuffer::zeroed(BATCH * ROWS).expect("W4A16 reference output");
        let packed_weight =
            DeviceBuffer::from_host(&host_weight.packed_weight).expect("W4A16 reference weight");
        let weight_scale =
            DeviceBuffer::from_host(&host_weight.weight_scale).expect("W4A16 reference scales");
        nvfp4_w4a16_matvec_f32_batch_into_on_stream(
            &context.input,
            &packed_weight,
            &weight_scale,
            reference.output(),
            BATCH,
            ROWS,
            COLS,
            host_weight.weight_scale_2,
            &context.stream,
        )
        .expect("W4A16 reference projection");
        let reference = reference
            .copy_to_host(&context.stream)
            .expect("download W4A16 reference");
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download W4A4 output");
        verify_quality("NVFP4 W4A4 versus W4A16", &reference, &actual, 0.98, 0.30);
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize>
    TernaryG64Bf16Bench<BATCH, ROWS, COLS>
{
    fn enqueue(&mut self) {
        self.matrix
            .run_f32_batch_bf16_into_on_stream(
                &self.lt,
                &self.plan,
                self.input.input(),
                &mut self.input_bf16,
                self.output.output(),
                BATCH,
                &self.stream,
            )
            .expect("ternary g64 BF16 projection");
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> BenchContext
    for TernaryG64Bf16Bench<BATCH, ROWS, COLS>
{
    fn prepare(_num_chunks: usize) -> Self {
        let packed = synthetic_linear(ROWS, COLS);
        let input_host = synthetic_input(BATCH, COLS);
        let lt = CublasLt::new().expect("cuBLASLt");
        let plan = Bf16TnMatmulPlan::new(&lt, GemmShape::new(ROWS, BATCH, COLS), 32 * 1024 * 1024)
            .expect("BF16 plan");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut context = Self {
            lt,
            plan,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            matrix: TernaryG64Matrix::from_packed_with_bf16_prefill(&packed).expect("matrix"),
            input,
            input_bf16: DeviceBuffer::zeroed(BATCH * COLS).expect("BF16 input"),
            output: DeviceBuffer::zeroed(BATCH * ROWS).expect("output"),
            stream,
        };
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download");
        verify_bf16_selected_outputs(&packed, &input_host, &actual, BATCH);
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> LayerProjection
    for TernaryG64LinearBench<BATCH, ROWS, COLS>
{
    fn prepare_projection() -> Self {
        <Self as BenchContext>::prepare(1)
    }

    fn time_once(&mut self) -> f64 {
        self.start.record_on_stream(&self.stream).expect("start");
        self.enqueue();
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64
    }

    fn resident_weight_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }

    fn retain_output(&self) {
        black_box(self.output.cuda_address());
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> LayerProjection
    for TernaryG64Bf16Bench<BATCH, ROWS, COLS>
{
    fn prepare_projection() -> Self {
        <Self as BenchContext>::prepare(1)
    }

    fn time_once(&mut self) -> f64 {
        self.start.record_on_stream(&self.stream).expect("start");
        self.enqueue();
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64
    }

    fn resident_weight_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }

    fn retain_output(&self) {
        black_box(self.output.cuda_address());
    }
}

impl<const ROWS: usize, const COLS: usize> LayerProjection for Nvfp4W4A16Bench<ROWS, COLS> {
    fn prepare_projection() -> Self {
        <Self as BenchContext>::prepare(1)
    }

    fn time_once(&mut self) -> f64 {
        self.start.record_on_stream(&self.stream).expect("start");
        self.enqueue();
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64
    }

    fn resident_weight_bytes(&self) -> usize {
        self.packed_weight.device_bytes() + self.weight_scale.device_bytes()
    }

    fn retain_output(&self) {
        black_box(self.output.cuda_address());
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> LayerProjection
    for Nvfp4W4A4Bench<BATCH, ROWS, COLS>
{
    fn prepare_projection() -> Self {
        <Self as BenchContext>::prepare(1)
    }

    fn time_once(&mut self) -> f64 {
        self.start.record_on_stream(&self.stream).expect("start");
        self.enqueue();
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64
    }

    fn resident_weight_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }

    fn retain_output(&self) {
        black_box(self.output.cuda_address());
    }
}

fn synthetic_linear(rows: usize, cols: usize) -> TernaryG64PackedLinear {
    let groups = rows * (cols / 64);
    let mut raw = Vec::with_capacity(groups * 18);
    for group in 0..groups {
        let scale_bits = [0x3800u16, 0x3400, 0x3c00][group % 3];
        raw.extend_from_slice(&scale_bits.to_le_bytes());
        for packed in 0..16 {
            let mut byte = 0u8;
            for within in 0..4 {
                byte |= (((group * 7 + packed * 3 + within) % 3) as u8) << (within * 2);
            }
            raw.push(byte);
        }
    }
    TernaryG64PackedLinear::from_gguf_q2_0_g64("synthetic", rows, cols, &raw)
        .expect("synthetic g64 weight")
}

fn synthetic_input(batch: usize, cols: usize) -> Vec<f32> {
    (0..batch * cols)
        .map(|index| ((index * 37 % 509) as f32 - 254.0) / 73.0)
        .collect()
}

fn synthetic_nvfp4(packed: &TernaryG64PackedLinear) -> ModelOptNvfp4Linear {
    let packed_cols = packed.in_features / 4;
    let groups = packed.in_features / 64;
    let values = (0..packed.out_features * packed.in_features)
        .map(|index| {
            let row = index / packed.in_features;
            let col = index % packed.in_features;
            let byte = packed.packed_weight[row * packed_cols + col / 4];
            let code = (byte >> ((col % 4) * 2)) & 0x03;
            let value = (f32::from(code) - 1.0) * packed.group_scales[row * groups + col / 64];
            eider_cuda::format::f32_to_bf16(value)
        })
        .collect::<Vec<_>>();
    let weight = ModelOptNvfp4Linear::quantize_bf16(
        "synthetic-ternary",
        packed.out_features,
        packed.in_features,
        &values,
    )
    .expect("convert ternary weight to NVFP4");
    verify_nvfp4_weight_quality(packed, &weight);
    weight
}

fn verify_nvfp4_weight_quality(packed: &TernaryG64PackedLinear, weight: &ModelOptNvfp4Linear) {
    let rows = [0, packed.out_features / 2, packed.out_features - 1];
    for row in rows {
        let expected = (0..packed.in_features)
            .map(|col| packed.weight(row, col).expect("ternary weight"))
            .collect::<Vec<_>>();
        let actual = (0..packed.in_features)
            .map(|col| nvfp4_weight_value(weight, row, col))
            .collect::<Vec<_>>();
        verify_quality(
            &format!("NVFP4 conversion row {row}"),
            &expected,
            &actual,
            0.995,
            0.05,
        );
    }
}

fn nvfp4_weight_value(weight: &ModelOptNvfp4Linear, row: usize, col: usize) -> f32 {
    let flat = row * weight.in_features + col;
    let byte = weight.packed_weight[flat / 2];
    let code = if flat.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let blocks_per_row = weight.in_features / 16;
    let scale =
        eider_cuda::format::e4m3_value(weight.weight_scale[row * blocks_per_row + col / 16]);
    eider_cuda::format::e2m1_value(code) * scale * weight.weight_scale_2
}

fn verify_nvfp4_selected_outputs(
    weight: &ModelOptNvfp4Linear,
    input: &[f32],
    actual: &[f32],
    batch_rows: usize,
) {
    let batches = [0, batch_rows / 2, batch_rows - 1];
    let rows = [0, weight.out_features / 2, weight.out_features - 1];
    for batch in batches {
        for row in rows {
            let expected = (0..weight.in_features)
                .map(|col| {
                    input[batch * weight.in_features + col] * nvfp4_weight_value(weight, row, col)
                })
                .sum::<f32>();
            let observed = actual[batch * weight.out_features + row];
            let tolerance = 0.002 * expected.abs().max(1.0);
            assert!(
                (observed - expected).abs() <= tolerance,
                "NVFP4 W4A16 batch={batch} row={row} observed={observed} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

fn verify_quality(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    minimum_cosine: f64,
    maximum_nrmse: f64,
) {
    let mut dot = 0.0f64;
    let mut expected_squared = 0.0f64;
    let mut actual_squared = 0.0f64;
    let mut error_squared = 0.0f64;
    for (&expected, &actual) in expected.iter().zip(actual) {
        let expected = f64::from(expected);
        let actual = f64::from(actual);
        dot += expected * actual;
        expected_squared += expected * expected;
        actual_squared += actual * actual;
        error_squared += (expected - actual) * (expected - actual);
    }
    let cosine = dot / (expected_squared.sqrt() * actual_squared.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (error_squared / expected_squared.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("validated {label}: cosine={cosine:.6} nrmse={nrmse:.6}");
    assert!(
        cosine >= minimum_cosine && nrmse <= maximum_nrmse,
        "{label}: cosine={cosine:.6} nrmse={nrmse:.6}"
    );
}

fn verify_selected_outputs(
    packed: &TernaryG64PackedLinear,
    input: &[f32],
    actual: &[f32],
    batch_rows: usize,
) {
    let batches = [0, batch_rows / 2, batch_rows - 1];
    let rows = [0, packed.out_features / 2, packed.out_features - 1];
    let groups = packed.in_features / 64;
    for batch in batches {
        for row in rows {
            let input_row = &input[batch * packed.in_features..(batch + 1) * packed.in_features];
            let mut expected = 0.0f32;
            for group in 0..groups {
                let start = group * 64;
                let values = &input_row[start..start + 64];
                let maximum = values
                    .iter()
                    .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
                let input_scale = maximum / 127.0;
                let quantize_scale = if maximum == 0.0 { 0.0 } else { 127.0 / maximum };
                let mut integer_sum = 0i32;
                for (offset, value) in values.iter().enumerate() {
                    let col = start + offset;
                    let byte = packed.packed_weight[row * (packed.in_features / 4) + col / 4];
                    let code = (byte >> ((col % 4) * 2)) & 0x03;
                    let quantized = (value * quantize_scale)
                        .round_ties_even()
                        .clamp(-127.0, 127.0) as i32;
                    integer_sum += (i32::from(code) - 1) * quantized;
                }
                expected +=
                    integer_sum as f32 * input_scale * packed.group_scales[row * groups + group];
            }
            let observed = actual[batch * packed.out_features + row];
            let tolerance = 2.0e-3f32.max(expected.abs() * 2.0e-5);
            assert!(
                (observed - expected).abs() <= tolerance,
                "ternary g64 batch={batch} row={row} observed={observed} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

fn verify_bf16_selected_outputs(
    packed: &TernaryG64PackedLinear,
    input: &[f32],
    actual: &[f32],
    batch_rows: usize,
) {
    let batches = [0, batch_rows / 2, batch_rows - 1];
    let rows = [0, packed.out_features / 2, packed.out_features - 1];
    for batch in batches {
        for row in rows {
            let expected = (0..packed.in_features)
                .map(|col| {
                    let input = eider_cuda::format::bf16_to_f32(eider_cuda::format::f32_to_bf16(
                        input[batch * packed.in_features + col],
                    ));
                    let weight = eider_cuda::format::bf16_to_f32(eider_cuda::format::f32_to_bf16(
                        packed.weight(row, col).expect("weight"),
                    ));
                    input * weight
                })
                .sum::<f32>();
            let observed = actual[batch * packed.out_features + row];
            let tolerance = 0.002 * expected.abs().max(1.0);
            assert!(
                (observed - expected).abs() <= tolerance,
                "ternary BF16 batch={batch} row={row} observed={observed} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

fn sample<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    context: &mut TernaryG64LinearBench<BATCH, ROWS, COLS>,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64
        / chunk_size as f64;
    black_box(context.output.cuda_address());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            context.matrix.device_bytes() as i64,
            "bytes",
        ))
}

fn sample_bf16<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    context: &mut TernaryG64Bf16Bench<BATCH, ROWS, COLS>,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64
        / chunk_size as f64;
    black_box(context.output.cuda_address());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            context.matrix.device_bytes() as i64,
            "bytes",
        ))
}

fn sample_nvfp4_w4a16<const ROWS: usize, const COLS: usize>(
    context: &mut Nvfp4W4A16Bench<ROWS, COLS>,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64
        / chunk_size as f64;
    black_box(context.output.cuda_address());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            (context.packed_weight.device_bytes() + context.weight_scale.device_bytes()) as i64,
            "bytes",
        ))
}

fn sample_nvfp4_w4a4<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    context: &mut Nvfp4W4A4Bench<BATCH, ROWS, COLS>,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64
        / chunk_size as f64;
    black_box(context.output.cuda_address());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            context.matrix.device_bytes() as i64,
            "bytes",
        ))
}

type DecodeW2A8Layer = BonsaiLayerBench<
    TernaryG64LinearBench<1, QKV_WIDTH, HIDDEN>,
    TernaryG64LinearBench<1, HIDDEN, HIDDEN>,
    TernaryG64LinearBench<1, GATE_UP, HIDDEN>,
    TernaryG64LinearBench<1, HIDDEN, INTERMEDIATE>,
>;
type DecodeW4A16Layer = BonsaiLayerBench<
    Nvfp4W4A16Bench<QKV_WIDTH, HIDDEN>,
    Nvfp4W4A16Bench<HIDDEN, HIDDEN>,
    Nvfp4W4A16Bench<GATE_UP, HIDDEN>,
    Nvfp4W4A16Bench<HIDDEN, INTERMEDIATE>,
>;
type PrefillW2A8Layer = BonsaiLayerBench<
    TernaryG64LinearBench<256, QKV_WIDTH, HIDDEN>,
    TernaryG64LinearBench<256, HIDDEN, HIDDEN>,
    TernaryG64LinearBench<256, GATE_UP, HIDDEN>,
    TernaryG64LinearBench<256, HIDDEN, INTERMEDIATE>,
>;
type PrefillBf16Layer = BonsaiLayerBench<
    TernaryG64Bf16Bench<256, QKV_WIDTH, HIDDEN>,
    TernaryG64Bf16Bench<256, HIDDEN, HIDDEN>,
    TernaryG64Bf16Bench<256, GATE_UP, HIDDEN>,
    TernaryG64Bf16Bench<256, HIDDEN, INTERMEDIATE>,
>;
type PrefillW4A4Layer = BonsaiLayerBench<
    Nvfp4W4A4Bench<256, QKV_WIDTH, HIDDEN>,
    Nvfp4W4A4Bench<256, HIDDEN, HIDDEN>,
    Nvfp4W4A4Bench<256, GATE_UP, HIDDEN>,
    Nvfp4W4A4Bench<256, HIDDEN, INTERMEDIATE>,
>;

fn sample_layer<Q, O, G, D>(
    context: &mut BonsaiLayerBench<Q, O, G, D>,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult
where
    Q: LayerProjection,
    O: LayerProjection,
    G: LayerProjection,
    D: LayerProjection,
{
    let mut elapsed_ms = 0.0;
    for _ in 0..chunk_size {
        elapsed_ms += context.qkv.time_once();
        elapsed_ms += context.output.time_once();
        elapsed_ms += context.gate_up.time_once();
        elapsed_ms += context.down.time_once();
    }
    elapsed_ms /= chunk_size as f64;
    context.qkv.retain_output();
    context.output.retain_output();
    context.gate_up.retain_output();
    context.down.retain_output();
    let resident_weight_bytes = context.qkv.resident_weight_bytes()
        + context.output.resident_weight_bytes()
        + context.gate_up.resident_weight_bytes()
        + context.down.resident_weight_bytes();
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            resident_weight_bytes as i64,
            "bytes",
        ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("ternary-g64-w2a8".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(100),
                benchmark_duration: Duration::from_millis(500),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            register::<1, QKV_WIDTH, HIDDEN>(runner, "qkv_m1_n6144_k4096", "Ternary Bonsai decode");
            register::<1, HIDDEN, HIDDEN>(runner, "output_m1_n4096_k4096", "Ternary Bonsai decode");
            register::<1, GATE_UP, HIDDEN>(
                runner,
                "gate_up_m1_n24576_k4096",
                "Ternary Bonsai decode",
            );
            register::<1, HIDDEN, INTERMEDIATE>(
                runner,
                "down_m1_n4096_k12288",
                "Ternary Bonsai decode",
            );
            register_nvfp4_decode::<QKV_WIDTH, HIDDEN>(runner, "qkv_m1_n6144_k4096_w4a16");
            register_nvfp4_decode::<HIDDEN, HIDDEN>(runner, "output_m1_n4096_k4096_w4a16");
            register_nvfp4_decode::<GATE_UP, HIDDEN>(runner, "gate_up_m1_n24576_k4096_w4a16");
            register_nvfp4_decode::<HIDDEN, INTERMEDIATE>(runner, "down_m1_n4096_k12288_w4a16");

            register::<256, QKV_WIDTH, HIDDEN>(
                runner,
                "qkv_m256_n6144_k4096",
                "Ternary Bonsai prefill",
            );
            register::<256, HIDDEN, HIDDEN>(
                runner,
                "output_m256_n4096_k4096",
                "Ternary Bonsai prefill",
            );
            register::<256, GATE_UP, HIDDEN>(
                runner,
                "gate_up_m256_n24576_k4096",
                "Ternary Bonsai prefill",
            );
            register::<256, HIDDEN, INTERMEDIATE>(
                runner,
                "down_m256_n4096_k12288",
                "Ternary Bonsai prefill",
            );
            register_bf16::<256, QKV_WIDTH, HIDDEN>(runner, "qkv_m256_n6144_k4096_bf16");
            register_bf16::<256, HIDDEN, HIDDEN>(runner, "output_m256_n4096_k4096_bf16");
            register_bf16::<256, GATE_UP, HIDDEN>(runner, "gate_up_m256_n24576_k4096_bf16");
            register_bf16::<256, HIDDEN, INTERMEDIATE>(runner, "down_m256_n4096_k12288_bf16");
            register_nvfp4_prefill::<256, QKV_WIDTH, HIDDEN>(runner, "qkv_m256_n6144_k4096_w4a4");
            register_nvfp4_prefill::<256, HIDDEN, HIDDEN>(runner, "output_m256_n4096_k4096_w4a4");
            register_nvfp4_prefill::<256, GATE_UP, HIDDEN>(
                runner,
                "gate_up_m256_n24576_k4096_w4a4",
            );
            register_nvfp4_prefill::<256, HIDDEN, INTERMEDIATE>(
                runner,
                "down_m256_n4096_k12288_w4a4",
            );
            register_layer_bakeoff(runner);
        },
    );
}

fn register_layer_bakeoff(runner: &mut micromeasure::BenchmarkRunner) {
    let ternary_weight_bytes = ternary_layer_weight_bytes();
    let nvfp4_weight_bytes = nvfp4_layer_weight_bytes();
    let bf16_weight_bytes = bf16_layer_weight_bytes();

    runner.group::<DecodeW2A8Layer>("Ternary Bonsai decode layer", |group| {
        group
            .throughput(Throughput::bytes(ternary_weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample("decode_layer_w2a8", sample_layer);
    });
    runner.group::<DecodeW4A16Layer>("Ternary Bonsai NVFP4 decode layer", |group| {
        group
            .throughput(Throughput::bytes(nvfp4_weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample("decode_layer_w4a16", sample_layer);
    });
    runner.group::<PrefillW2A8Layer>("Ternary Bonsai direct prefill layer", |group| {
        group
            .throughput(Throughput::bytes(ternary_weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample("prefill_layer_w2a8", sample_layer);
    });
    runner.group::<PrefillBf16Layer>("Ternary Bonsai BF16 prefill layer", |group| {
        group
            .throughput(Throughput::bytes(bf16_weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample("prefill_layer_bf16", sample_layer);
    });
    runner.group::<PrefillW4A4Layer>("Ternary Bonsai NVFP4 prefill layer", |group| {
        group
            .throughput(Throughput::bytes(nvfp4_weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample("prefill_layer_w4a4", sample_layer);
    });
}

const fn layer_weights() -> usize {
    QKV_WIDTH * HIDDEN + HIDDEN * HIDDEN + GATE_UP * HIDDEN + HIDDEN * INTERMEDIATE
}

const fn ternary_layer_weight_bytes() -> usize {
    layer_weights() / 4 + layer_weights() / 64 * 4
}

const fn nvfp4_layer_weight_bytes() -> usize {
    layer_weights() / 2 + layer_weights() / 16
}

const fn bf16_layer_weight_bytes() -> usize {
    layer_weights() * 2
}

fn register_nvfp4_decode<const ROWS: usize, const COLS: usize>(
    runner: &mut micromeasure::BenchmarkRunner,
    name: &'static str,
) {
    let weight_bytes = ROWS * COLS / 2 + ROWS * (COLS / 16);
    runner.group::<Nvfp4W4A16Bench<ROWS, COLS>>("Ternary Bonsai NVFP4 decode", |group| {
        group
            .throughput(Throughput::bytes(weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample(name, sample_nvfp4_w4a16::<ROWS, COLS>);
    });
}

fn register_nvfp4_prefill<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    runner: &mut micromeasure::BenchmarkRunner,
    name: &'static str,
) {
    let weight_bytes = ROWS * COLS / 2 + ROWS * (COLS / 16);
    runner.group::<Nvfp4W4A4Bench<BATCH, ROWS, COLS>>("Ternary Bonsai NVFP4 prefill", |group| {
        group
            .throughput(Throughput::bytes(weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample(name, sample_nvfp4_w4a4::<BATCH, ROWS, COLS>);
    });
}

fn register_bf16<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    runner: &mut micromeasure::BenchmarkRunner,
    name: &'static str,
) {
    runner.group::<TernaryG64Bf16Bench<BATCH, ROWS, COLS>>(
        "Ternary Bonsai BF16 prefill",
        |group| {
            group
                .throughput(Throughput::bytes((ROWS * COLS * 2) as u64))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample(name, sample_bf16::<BATCH, ROWS, COLS>);
        },
    );
}

fn register<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    runner: &mut micromeasure::BenchmarkRunner,
    name: &'static str,
    group_name: &'static str,
) {
    let weight_bytes = ROWS * COLS / 4 + ROWS * (COLS / 64) * 4;
    runner.group::<TernaryG64LinearBench<BATCH, ROWS, COLS>>(group_name, |group| {
        group
            .throughput(Throughput::bytes(weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample(name, sample::<BATCH, ROWS, COLS>);
    });
}
