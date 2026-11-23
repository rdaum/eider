//! Inkling 2 recurrence, attention preparation, convolution, and expert operators.
//!
//! F32 device buffers carry the model's explicitly rounded BF16 values. Stable
//! per-sequence convolution and KV state remains owned by the inference family.
use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::kernels::sm121_w4a16::MoeRouteGroups;
use crate::{Error, Result};
use std::ffi::c_void;

pub mod draft;

unsafe extern "C" {
    fn infer_inkling2_embed(
        embedding: *const u16,
        tokens: *const u32,
        gamma: *const f32,
        out: *mut f32,
        first: u32,
        rows: u32,
        width: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_rms(
        x: *const f32,
        gamma: *const f32,
        out: *mut f32,
        rows: u32,
        width: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_attn_res(
        unused: *const *const f32,
        s0: *const f32,
        s1: *const f32,
        s2: *const f32,
        s3: *const f32,
        s4: *const f32,
        addend: *const f32,
        query: *const f32,
        out: *mut f32,
        rows: u32,
        sources: u32,
        width: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_controller_norm(
        destination: *const f32,
        message: *const f32,
        normed: *mut f32,
        rows: u32,
        width: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_controller_mix(
        destination: *const f32,
        message: *const f32,
        gain: *const f32,
        forget: *const f32,
        out: *mut f32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_sconv(
        x: *const f32,
        weight: *const f32,
        dynamic: *const f32,
        state: *const f32,
        out: *mut f32,
        rows: u32,
        hidden: u32,
        width: u32,
        nheads: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_sconv_commit(
        x: *const f32,
        state: *mut f32,
        rows: u32,
        hidden: u32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_l2_clip(
        x: *const f32,
        scale: *const f32,
        out: *mut f32,
        rows: u32,
        heads: u32,
        dim: u32,
        threshold: f32,
        eps: f32,
        position: f32,
        device_position: *const u32,
        tau_floor: u32,
        tau_alpha: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_rel_projection(
        r: *const f32,
        projection: *const u16,
        out: *mut f32,
        rows: u32,
        heads: u32,
        extent: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_router(
        logits: *const f32,
        bias: *const f32,
        ids: *mut u32,
        weights: *mut f32,
        rows: u32,
        experts: u32,
        route_scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_capped_swiglu(
        gate_up: *const f32,
        gammas: *const f32,
        out: *mut f32,
        routes: u32,
        intermediate: u32,
        gate_cap: f32,
        up_cap: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_grouped_mxfp8_expert(
        sorted: *const u32,
        item_expert: *const u32,
        item_start: *const u32,
        item_count: *const u32,
        items: *const u32,
        max_items: u32,
        input: *const f32,
        weights: *const u8,
        scales: *const u8,
        out: *mut f32,
        top_k: u32,
        per_route_input: bool,
        outputs: u32,
        inputs: u32,
        tiled: bool,
        max_columns: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_bf16_matvec(
        x: *const f32,
        weight: *const u16,
        y: *mut f32,
        rows: u32,
        outputs: u32,
        inputs: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_routed_sum(
        routes: *const f32,
        out: *mut f32,
        rows: u32,
        top_k: u32,
        hidden: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_inkling2_finish_logits(
        logits: *mut f32,
        count: u32,
        real_vocab: u32,
        softcap: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Routed batches at least this large use the activation-staging expert kernel.
const TILED_ROUTES: usize = 64;

fn invalid(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Inkling2 operator",
        detail: detail.into(),
    }
}

fn launch_dimensions(rows: usize, width: usize) -> Result<(u32, u32)> {
    let count = rows.checked_mul(width);
    if rows == 0
        || width == 0
        || rows > u32::MAX as usize
        || width > u32::MAX as usize
        || count.is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(invalid("invalid launch dimensions"));
    }
    Ok((rows as u32, width as u32))
}

/// Gathers `rows` tokens starting at `first` and applies the embedding RMS norm.
#[allow(clippy::too_many_arguments)]
pub fn embed(
    embedding: &DeviceBuffer<u16>,
    vocabulary: usize,
    tokens: &DeviceBuffer<u32>,
    gamma: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    first: usize,
    rows: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let width = gamma.len();
    let (rows, width) = launch_dimensions(rows, width)?;
    if first.saturating_add(rows as usize) > tokens.len()
        || embedding.len() != vocabulary.saturating_mul(width as usize)
        || out.len() < rows as usize * width as usize
        || vocabulary == 0
        || first > u32::MAX as usize
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid("embedding geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_embed",
            infer_inkling2_embed(
                embedding.ptr,
                tokens.ptr,
                gamma.ptr,
                out.buffer_mut().ptr,
                first as u32,
                rows,
                width,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies RMS normalization with BF16 rounding and an optional learned gamma.
pub fn rms_norm(
    x: &DeviceBuffer<f32>,
    gamma: Option<&DeviceBuffer<f32>>,
    mut out: DeviceOutput<'_, f32>,
    rows: usize,
    width: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let (rows, width) = launch_dimensions(rows, width)?;
    if x.len() < rows as usize * width as usize
        || out.len() < rows as usize * width as usize
        || gamma.is_some_and(|g| g.len() != width as usize)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid("RMS norm geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_rms",
            infer_inkling2_rms(
                x.ptr,
                gamma.map_or(std::ptr::null(), |g| g.ptr),
                out.buffer_mut().ptr,
                rows,
                width,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Reads one pseudo-query over up to five normalized Attention Residual sources.
/// If supplied, `addend` contributes only to the last source.
#[allow(clippy::too_many_arguments)]
pub fn attention_residual(
    sources: &[&DeviceBuffer<f32>],
    addend: Option<&DeviceBuffer<f32>>,
    query: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    rows: usize,
    width: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let (rows, width) = launch_dimensions(rows, width)?;
    let count = rows as usize * width as usize;
    if sources.is_empty()
        || sources.len() > 5
        || sources.iter().any(|source| source.len() < count)
        || addend.is_some_and(|addend| addend.len() < count)
        || query.len() != width as usize
        || out.len() < count
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid("Attention Residual geometry mismatch"));
    }
    let mut pointers = [std::ptr::null(); 5];
    for (pointer, source) in pointers.iter_mut().zip(sources) {
        *pointer = source.ptr;
    }
    unsafe {
        check_cuda(
            "infer_inkling2_attn_res",
            infer_inkling2_attn_res(
                std::ptr::null(),
                pointers[0],
                pointers[1],
                pointers[2],
                pointers[3],
                pointers[4],
                addend.map_or(std::ptr::null(), |x| x.ptr),
                query.ptr,
                out.buffer_mut().ptr,
                rows,
                sources.len() as u32,
                width,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Produces BF16 RMS-normalized destination/message pairs for a loop controller.
pub fn controller_norm(
    destination: &DeviceBuffer<f32>,
    message: &DeviceBuffer<f32>,
    mut normed: DeviceOutput<'_, f32>,
    rows: usize,
    width: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let (rows, width) = launch_dimensions(rows, width)?;
    let count = rows as usize * width as usize;
    if destination.len() < count || message.len() < count || normed.len() < 2 * count {
        return Err(invalid("controller normalization geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_controller_norm",
            infer_inkling2_controller_norm(
                destination.ptr,
                message.ptr,
                normed.buffer_mut().ptr,
                rows,
                width,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies the learned gain/forget interpolation in F32 without FP contraction.
pub fn controller_mix(
    destination: &DeviceBuffer<f32>,
    message: &DeviceBuffer<f32>,
    gain: &DeviceBuffer<f32>,
    forget: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    count: usize,
    stream: &CudaStream,
) -> Result<()> {
    if count == 0
        || count > u32::MAX as usize
        || [destination, message, gain, forget]
            .iter()
            .any(|buffer| buffer.len() < count)
        || out.len() < count
    {
        return Err(invalid("controller mixing geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_controller_mix",
            infer_inkling2_controller_mix(
                destination.ptr,
                message.ptr,
                gain.ptr,
                forget.ptr,
                out.buffer_mut().ptr,
                count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Evaluates `rows` causal depthwise-convolution rows without modifying the
/// history state, which holds the inputs before the first row.
#[allow(clippy::too_many_arguments)]
pub fn short_conv(
    x: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    dynamic: Option<&DeviceBuffer<f32>>,
    state: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    rows: usize,
    hidden: usize,
    width: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    let (rows, hidden) = launch_dimensions(rows, hidden)?;
    let width = u32::try_from(width).map_err(|_| invalid("invalid convolution width"))?;
    let heads = u32::try_from(heads).map_err(|_| invalid("invalid convolution heads"))?;
    let count = rows as usize * hidden as usize;
    if !(2..=32).contains(&width)
        || heads == 0
        || !hidden.is_multiple_of(heads)
        || x.len() < count
        || out.len() < count
        || weight.len() < hidden as usize * width as usize
        || state.len() < hidden as usize * (width as usize - 1)
        || dynamic
            .is_some_and(|d| d.len() < count / hidden as usize * width as usize * heads as usize)
    {
        return Err(invalid(format!(
            "short-convolution rows={rows} hidden={hidden} width={width} heads={heads}; x={} out={} weight={} state={}",
            x.len(),
            out.len(),
            weight.len(),
            state.len()
        )));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_sconv",
            infer_inkling2_sconv(
                x.ptr,
                weight.ptr,
                dynamic.map_or(std::ptr::null(), |d| d.ptr),
                state.ptr,
                out.buffer_mut().ptr,
                rows,
                hidden,
                width,
                heads,
                stream.as_raw(),
            ),
        )
    }
}

/// Retains only the raw inputs needed by the next short-convolution call.
pub fn commit_short_conv(
    x: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    rows: usize,
    hidden: usize,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let (rows, hidden) = launch_dimensions(rows, hidden)?;
    let width = u32::try_from(width).map_err(|_| invalid("invalid convolution width"))?;
    if !(2..=32).contains(&width)
        || x.len() < rows as usize * hidden as usize
        || state.len() < hidden as usize * (width as usize - 1)
    {
        return Err(invalid("short-convolution commit geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_sconv_commit",
            infer_inkling2_sconv_commit(
                x.ptr,
                state.buffer_mut().ptr,
                rows,
                hidden,
                width,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies learned per-channel scaling, L2 clipping, and optional log-context
/// tau. A `device_position` replaces `position` when the launch executes.
#[allow(clippy::too_many_arguments)]
pub fn l2_clip(
    x: &DeviceBuffer<f32>,
    scale: Option<&DeviceBuffer<f32>>,
    mut out: DeviceOutput<'_, f32>,
    rows: usize,
    heads: usize,
    dim: usize,
    threshold: f32,
    eps: f32,
    position: usize,
    device_position: Option<&DeviceBuffer<u32>>,
    tau: Option<(usize, f32)>,
    stream: &CudaStream,
) -> Result<()> {
    let count = heads.checked_mul(dim).and_then(|n| n.checked_mul(rows));
    if rows == 0
        || rows > u32::MAX as usize
        || heads == 0
        || dim == 0
        || dim > 128
        || !dim.is_power_of_two()
        || heads > u32::MAX as usize
        || count.is_none_or(|n| n > u32::MAX as usize)
        || x.len() < count.unwrap_or(0)
        || out.len() < count.unwrap_or(0)
        || scale.is_some_and(|s| s.len() != heads * dim)
        || !threshold.is_finite()
        || threshold <= 0.0
        || !eps.is_finite()
        || eps <= 0.0
        || position.saturating_add(rows) >= u32::MAX as usize
        || device_position.is_some_and(|p| p.len() != 1)
        || tau.is_some_and(|(floor, alpha)| {
            floor == 0 || floor > u32::MAX as usize || !alpha.is_finite()
        })
    {
        return Err(invalid("QK L2 clipping geometry mismatch"));
    }
    let (floor, alpha) = tau.unwrap_or((0, 0.0));
    unsafe {
        check_cuda(
            "infer_inkling2_l2_clip",
            infer_inkling2_l2_clip(
                x.ptr,
                scale.map_or(std::ptr::null(), |s| s.ptr),
                out.buffer_mut().ptr,
                rows as u32,
                heads as u32,
                dim as u32,
                threshold,
                eps,
                position as f32,
                device_position.map_or(std::ptr::null(), |p| p.ptr.cast_const()),
                floor as u32,
                alpha,
                stream.as_raw(),
            ),
        )
    }
}

/// Projects each row and head's 16 relative features onto its distance-bias table.
pub fn relative_projection(
    r: &DeviceBuffer<f32>,
    projection: &DeviceBuffer<u16>,
    mut out: DeviceOutput<'_, f32>,
    rows: usize,
    heads: usize,
    extent: usize,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || heads == 0
        || extent == 0
        || r.len() < rows * heads * 16
        || projection.len() != 16 * extent
        || out.len() < rows * heads * extent
        || rows * heads * extent > u32::MAX as usize
    {
        return Err(invalid("relative-projection geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_rel_projection",
            infer_inkling2_rel_projection(
                r.ptr,
                projection.ptr,
                out.buffer_mut().ptr,
                rows as u32,
                heads as u32,
                extent as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Selects eight biased sigmoid routes, then normalizes their log-sigmoid weights.
pub fn router(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    ids: &mut DeviceBuffer<u32>,
    weights: &mut DeviceBuffer<f32>,
    rows: usize,
    route_scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let experts = bias.len();
    if rows == 0
        || experts == 0
        || experts > 128
        || rows > u32::MAX as usize
        || experts > u32::MAX as usize
        || logits.len() < rows * experts
        || ids.len() < rows * 8
        || weights.len() < rows * 8
        || !route_scale.is_finite()
        || route_scale <= 0.0
    {
        return Err(invalid("router geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_router",
            infer_inkling2_router(
                logits.ptr,
                bias.ptr,
                ids.ptr,
                weights.ptr,
                rows as u32,
                experts as u32,
                route_scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies Inkling's interleaved capped SwiGLU and per-route gamma before down.
#[allow(clippy::too_many_arguments)]
pub fn capped_swiglu(
    gate_up: &DeviceBuffer<f32>,
    gammas: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    routes: usize,
    intermediate: usize,
    gate_cap: f32,
    up_cap: f32,
    stream: &CudaStream,
) -> Result<()> {
    if routes == 0
        || intermediate == 0
        || routes > u32::MAX as usize
        || intermediate > u32::MAX as usize
        || gate_up.len() < routes * intermediate * 2
        || gammas.len() < routes
        || out.len() < routes * intermediate
        || !gate_cap.is_finite()
        || gate_cap <= 0.0
        || !up_cap.is_finite()
        || up_cap <= 0.0
    {
        return Err(invalid("capped-SwiGLU geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_capped_swiglu",
            infer_inkling2_capped_swiglu(
                gate_up.ptr,
                gammas.ptr,
                out.buffer_mut().ptr,
                routes as u32,
                intermediate as u32,
                gate_cap,
                up_cap,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs MXFP8 expert weights `[experts, outputs, inputs]` (E4M3 values with one
/// E8M0 scale per 32 inputs) for grouped routes. Route `r` reads input row
/// `r / top_k`, or row `r` with `per_route_input`, and writes BF16-rounded
/// output row `r`. Decode-sized batches split K across warps; larger batches
/// stage route activations.
#[allow(clippy::too_many_arguments)]
pub fn grouped_mxfp8_expert(
    groups: &MoeRouteGroups,
    input: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    mut out: DeviceOutput<'_, f32>,
    routes: usize,
    top_k: usize,
    outputs: usize,
    inputs: usize,
    per_route_input: bool,
    stream: &CudaStream,
) -> Result<()> {
    let input_rows = if per_route_input {
        routes
    } else {
        routes.div_ceil(top_k.max(1))
    };
    if routes == 0
        || top_k == 0
        || outputs == 0
        || inputs == 0
        || !outputs.is_multiple_of(16)
        || !inputs.is_multiple_of(32)
        || outputs > u32::MAX as usize
        || inputs > u32::MAX as usize
        || input.len() < input_rows * inputs
        || out.len() < routes * outputs
        || weights.len() != groups.experts() * outputs * inputs
        || scales.len() != weights.len() / 32
    {
        return Err(invalid("grouped MXFP8 expert geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_grouped_mxfp8_expert",
            infer_inkling2_grouped_mxfp8_expert(
                groups.sorted.ptr,
                groups.item_expert.ptr,
                groups.item_start.ptr,
                groups.item_count.ptr,
                groups.items.ptr,
                groups.max_items(routes) as u32,
                input.ptr,
                weights.ptr,
                scales.ptr,
                out.buffer_mut().ptr,
                top_k as u32,
                per_route_input,
                outputs as u32,
                inputs as u32,
                routes >= TILED_ROUTES,
                // A token selects distinct experts; larger items loop.
                routes.div_ceil(top_k).min(u32::MAX as usize) as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Largest batch accepted by [`bf16_matvec`].
pub const MATVEC_ROWS: usize = 8;

/// Projects up to [`MATVEC_ROWS`] F32 rows through a BF16 `[outputs, inputs]`
/// weight with F32 accumulation, one warp per output feature.
pub fn bf16_matvec(
    x: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    mut y: DeviceOutput<'_, f32>,
    rows: usize,
    outputs: usize,
    inputs: usize,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || rows > MATVEC_ROWS
        || outputs == 0
        || inputs == 0
        || !inputs.is_multiple_of(8)
        || outputs > u32::MAX as usize
        || inputs > u32::MAX as usize
        || x.len() < rows * inputs
        || y.len() < rows * outputs
        || weight.len() != outputs * inputs
    {
        return Err(invalid("BF16 matrix-vector geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_bf16_matvec",
            infer_inkling2_bf16_matvec(
                x.ptr,
                weight.ptr,
                y.buffer_mut().ptr,
                rows as u32,
                outputs as u32,
                inputs as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Sums already gamma-weighted routed down outputs for each row.
pub fn routed_sum(
    routes: &DeviceBuffer<f32>,
    mut out: DeviceOutput<'_, f32>,
    rows: usize,
    top_k: usize,
    hidden: usize,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || top_k == 0
        || hidden == 0
        || rows * hidden > u32::MAX as usize
        || routes.len() < rows * top_k * hidden
        || out.len() < rows * hidden
    {
        return Err(invalid("routed-sum geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_routed_sum",
            infer_inkling2_routed_sum(
                routes.ptr,
                out.buffer_mut().ptr,
                rows as u32,
                top_k as u32,
                hidden as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies final logit softcapping and masks the checkpoint's padded vocabulary.
pub fn finish_logits(
    logits: &mut DeviceBuffer<f32>,
    real_vocab: usize,
    softcap: f32,
    stream: &CudaStream,
) -> Result<()> {
    if logits.len() > u32::MAX as usize
        || real_vocab > logits.len()
        || !softcap.is_finite()
        || softcap <= 0.0
    {
        return Err(invalid("logit-finish geometry mismatch"));
    }
    unsafe {
        check_cuda(
            "infer_inkling2_finish_logits",
            infer_inkling2_finish_logits(
                logits.ptr,
                logits.len() as u32,
                real_vocab as u32,
                softcap,
                stream.as_raw(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DeviceAddress, RelativeBias, RelativeKv, SinkAttentionCache, SinkAttentionWorkspace,
    };

    fn bf(x: f32) -> f32 {
        let bits = x.to_bits();
        f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) & 0xffff0000)
    }

    /// Allows one BF16 rounding step of the expected value plus `floor`.
    fn close_bf16(actual: &[f32], expected: &[f32], floor: f32) -> Result<()> {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= floor + expected.abs() / 128.0,
                "index {index}: actual {actual}, expected {expected}"
            );
        }
        Ok(())
    }

    fn close(actual: &[f32], expected: &[f32], tolerance: f32) -> Result<()> {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual {actual}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn attention_residual_normalizes_addend_and_computes_softmax() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let first = DeviceBuffer::from_host(&(0..8).map(|x| x as f32 - 2.0).collect::<Vec<_>>())?;
        let second =
            DeviceBuffer::from_host(&(0..8).map(|x| 1.0 - x as f32 / 4.0).collect::<Vec<_>>())?;
        let addend = DeviceBuffer::from_host(&[0.5; 8])?;
        let query =
            DeviceBuffer::from_host(&(0..8).map(|x| (x + 1) as f32 / 16.0).collect::<Vec<_>>())?;
        let mut output = DeviceBuffer::zeroed(8)?;
        attention_residual(
            &[&first, &second],
            Some(&addend),
            &query,
            output.output(),
            1,
            8,
            1e-6,
            &stream,
        )?;
        let second_values = (0..8).map(|x| 1.5 - x as f32 / 4.0).collect::<Vec<_>>();
        let sources = [
            (0..8).map(|x| x as f32 - 2.0).collect::<Vec<_>>(),
            second_values,
        ];
        let query = (0..8).map(|x| (x + 1) as f32 / 16.0).collect::<Vec<_>>();
        let norms = sources.clone().map(|source| {
            (source.iter().map(|x| x * x).sum::<f32>() / 8.0 + 1e-6)
                .sqrt()
                .recip()
        });
        let dots = [0, 1].map(|index| {
            sources[index]
                .iter()
                .zip(&query)
                .map(|(x, query)| x * query / 8f32.sqrt())
                .sum::<f32>()
                * norms[index]
        });
        let maximum = dots[0].max(dots[1]);
        let weights = dots.map(|x| (x - maximum).exp());
        let total = weights[0] + weights[1];
        let expected = (0..8)
            .map(|index| {
                (weights[0] * norms[0] * sources[0][index]
                    + weights[1] * norms[1] * sources[1][index])
                    / total
            })
            .collect::<Vec<_>>();
        close(&output.copy_to_host(&stream)?, &expected, 2e-6)?;
        Ok(())
    }

    #[test]
    fn small_relative_heads_clip_and_commit_convolution_history() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let input = (0..16).map(|x| x as f32 / 4.0).collect::<Vec<_>>();
        let x = DeviceBuffer::from_host(&input)?;
        let mut clipped = DeviceBuffer::zeroed(16)?;
        l2_clip(
            &x,
            None,
            clipped.output(),
            1,
            1,
            16,
            1.0,
            1e-6,
            0,
            None,
            None,
            &stream,
        )?;
        let norm = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected = input
            .iter()
            .map(|x| bf(x / (norm + 1e-6)))
            .collect::<Vec<_>>();
        close(&clipped.copy_to_host(&stream)?, &expected, 0.0001)?;

        let raw = DeviceBuffer::from_host(&[1.0, 2.0, 2.0, 4.0])?;
        let weights = DeviceBuffer::from_host(&[0.25; 8])?;
        let mut state = DeviceBuffer::zeroed(6)?;
        let mut output = DeviceBuffer::zeroed(4)?;
        short_conv(
            &raw,
            &weights,
            None,
            &state,
            output.output(),
            1,
            2,
            4,
            1,
            &stream,
        )?;
        let scale = 5.0f32.sqrt();
        close(
            &output.copy_to_host(&stream)?[..2],
            &[bf(1.25 / scale), bf(2.5 / scale)],
            0.001,
        )?;
        commit_short_conv(&raw, state.inout(), 1, 2, 4, &stream)?;
        close(
            &state.copy_to_host(&stream)?,
            &[0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
            0.0,
        )?;
        let current = DeviceBuffer::from_host(&[2.0, 4.0])?;
        short_conv(
            &current,
            &weights,
            None,
            &state,
            output.output(),
            1,
            2,
            4,
            1,
            &stream,
        )?;
        close(
            &output.copy_to_host(&stream)?[..2],
            &[bf(2.75 / scale), bf(5.5 / scale)],
            0.001,
        )?;
        commit_short_conv(&current, state.inout(), 1, 2, 4, &stream)?;
        close(
            &state.copy_to_host(&stream)?,
            &[0.0, 1.0, 2.0, 0.0, 2.0, 4.0],
            0.0,
        )?;
        Ok(())
    }

    #[test]
    fn dynamic_short_conv_scales_both_weight_branches_and_residual() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let raw = DeviceBuffer::from_host(&[1.0, 2.0])?;
        // Global dynamic taps are indexed newest-to-oldest, unlike static taps.
        let weights = DeviceBuffer::from_host(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
        let dynamic = DeviceBuffer::zeroed(4)?;
        let state = DeviceBuffer::zeroed(6)?;
        let mut output = DeviceBuffer::zeroed(2)?;
        short_conv(
            &raw,
            &weights,
            Some(&dynamic),
            &state,
            output.output(),
            1,
            2,
            4,
            1,
            &stream,
        )?;
        let scale = 3.0f32;
        close(
            &output.copy_to_host(&stream)?,
            &[bf((1.0 + 1.0) / scale), bf((2.0 + 2.0 * 5.0) / scale)],
            0.001,
        )?;
        Ok(())
    }

    #[test]
    fn router_normalizes_log_sigmoid_weights() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let logit = |row: usize, e: usize| {
            let e = if row == 0 { e } else { 15 - e };
            e as f32 / 8.0 - 1.0
        };
        let values = (0..2)
            .flat_map(|row| (0..16).map(move |e| logit(row, e)))
            .collect::<Vec<_>>();
        let bias = DeviceBuffer::from_host(&[
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])?;
        let logits = DeviceBuffer::from_host(&values)?;
        let mut ids = DeviceBuffer::zeroed(16)?;
        let mut weights = DeviceBuffer::zeroed(16)?;
        router(&logits, &bias, &mut ids, &mut weights, 2, 8.0, &stream)?;
        let expected_ids = [[0, 15, 14, 13, 12, 11, 10, 9], [0, 1, 2, 3, 4, 5, 6, 7]];
        assert_eq!(ids.copy_to_host(&stream)?.as_slice(), expected_ids.concat());
        let mut expected = Vec::new();
        for (row, selected) in expected_ids.iter().enumerate() {
            let probs = selected
                .iter()
                .map(|&e| 1.0 / (1.0 + (-logit(row, e as usize)).exp()))
                .collect::<Vec<f32>>();
            let total = probs.iter().sum::<f32>();
            expected.extend(probs.iter().map(|x| 8.0 * x / total));
        }
        close(&weights.copy_to_host(&stream)?, &expected, 2e-6)?;
        Ok(())
    }

    #[test]
    fn chunked_short_conv_matches_single_row_steps() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let rows = [[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]];
        let weights = DeviceBuffer::from_host(&[0.5, -0.25, 1.0, 0.75, 0.1, 0.2, -0.3, 0.4])?;
        let dynamic =
            DeviceBuffer::from_host(&(0..12).map(|x| x as f32 / 16.0 - 0.3).collect::<Vec<_>>())?;
        for dynamic in [None, Some(&dynamic)] {
            let mut chunk_state = DeviceBuffer::from_host(&[0.5, -1.0, 2.0, 1.5, 0.25, -0.5])?;
            let mut step_state = DeviceBuffer::from_host(&[0.5, -1.0, 2.0, 1.5, 0.25, -0.5])?;
            let chunk = DeviceBuffer::from_host(&rows.concat())?;
            let mut chunk_out = DeviceBuffer::zeroed(6)?;
            short_conv(
                &chunk,
                &weights,
                dynamic,
                &chunk_state,
                chunk_out.output(),
                3,
                2,
                4,
                1,
                &stream,
            )?;
            commit_short_conv(&chunk, chunk_state.inout(), 3, 2, 4, &stream)?;
            let mut expected = Vec::new();
            for (row, values) in rows.iter().enumerate() {
                let x = DeviceBuffer::from_host(values)?;
                let mut out = DeviceBuffer::zeroed(2)?;
                let taps = dynamic
                    .map(|d| {
                        DeviceBuffer::from_host(&d.copy_to_host(&stream)?[row * 4..row * 4 + 4])
                    })
                    .transpose()?;
                short_conv(
                    &x,
                    &weights,
                    taps.as_ref(),
                    &step_state,
                    out.output(),
                    1,
                    2,
                    4,
                    1,
                    &stream,
                )?;
                commit_short_conv(&x, step_state.inout(), 1, 2, 4, &stream)?;
                expected.extend_from_slice(&out.copy_to_host(&stream)?);
            }
            close(&chunk_out.copy_to_host(&stream)?, &expected, 0.0)?;
            close(
                &chunk_state.copy_to_host(&stream)?,
                &step_state.copy_to_host(&stream)?,
                0.0,
            )?;
        }
        Ok(())
    }

    #[test]
    fn bf16_matvec_matches_cpu_for_each_batch_row() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let (outputs, inputs) = (37, 264);
        let weights = (0..outputs * inputs)
            .map(|i| bf(((i * 37 % 101) as f32 - 50.0) / 64.0))
            .collect::<Vec<_>>();
        let words = weights
            .iter()
            .map(|x| (x.to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let weight = DeviceBuffer::from_host(&words)?;
        for rows in [1, 3, MATVEC_ROWS] {
            let input = (0..rows * inputs)
                .map(|i| bf(((i * 13 % 29) as f32 - 14.0) / 8.0))
                .collect::<Vec<_>>();
            let x = DeviceBuffer::from_host(&input)?;
            let mut y = DeviceBuffer::zeroed(rows * outputs)?;
            bf16_matvec(&x, &weight, y.output(), rows, outputs, inputs, &stream)?;
            let expected = (0..rows)
                .flat_map(|r| {
                    let input = &input;
                    let weights = &weights;
                    (0..outputs).map(move |o| {
                        weights[o * inputs..][..inputs]
                            .iter()
                            .zip(&input[r * inputs..][..inputs])
                            .map(|(w, x)| w * x)
                            .sum::<f32>()
                    })
                })
                .collect::<Vec<_>>();
            close(&y.copy_to_host(&stream)?, &expected, 1e-3)?;
        }
        Ok(())
    }

    #[test]
    fn grouped_mxfp8_experts_match_cpu_for_shared_and_per_route_inputs() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        // 120 routes select the tiled kernel and give expert 1 two work items;
        // 96 inputs end inside a staged K chunk.
        let (experts, outputs, inputs, top_k, rows) = (3, 32, 96, 2, 60);
        let routes = rows * top_k;
        let e4m3 = |code: u8| {
            let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
            let exponent = (code >> 3) & 0xf;
            let mantissa = f32::from(code & 7);
            sign * if exponent == 0 {
                mantissa / 8.0 * 2f32.powi(-6)
            } else {
                (1.0 + mantissa / 8.0) * 2f32.powi(i32::from(exponent) - 7)
            }
        };
        // Avoid the E4M3 NaN codes 0x7f and 0xff.
        let codes = (0..experts * outputs * inputs)
            .map(|i| ((i * 37 + i / 7) % 127) as u8 | if i % 3 == 0 { 0x80 } else { 0 })
            .collect::<Vec<u8>>();
        let scale_codes = (0..codes.len() / 32)
            .map(|i| 112 + (i % 9) as u8)
            .collect::<Vec<u8>>();
        let weights = codes
            .iter()
            .enumerate()
            .map(|(i, &code)| e4m3(code) * 2f32.powi(i32::from(scale_codes[i / 32]) - 127))
            .collect::<Vec<_>>();
        let weights_device = DeviceBuffer::from_host(&codes)?;
        let scales_device = DeviceBuffer::from_host(&scale_codes)?;
        // Expert 1 receives more than one 32-route work item.
        let ids = (0..routes)
            .map(|r| {
                if r % 5 == 0 {
                    0
                } else if r % 7 == 0 {
                    2
                } else {
                    1
                }
            })
            .collect::<Vec<u32>>();
        let ids_device = DeviceBuffer::from_host(&ids)?;
        let mut groups = MoeRouteGroups::new(routes, experts)?;
        groups.group_on_stream(&ids_device, routes, &stream)?;
        for per_route_input in [false, true] {
            let input_rows = if per_route_input { routes } else { rows };
            let input = (0..input_rows * inputs)
                .map(|i| bf(((i * 13 % 29) as f32 - 14.0) / 8.0))
                .collect::<Vec<_>>();
            let input_device = DeviceBuffer::from_host(&input)?;
            let mut output = DeviceBuffer::zeroed(routes * outputs)?;
            grouped_mxfp8_expert(
                &groups,
                &input_device,
                &weights_device,
                &scales_device,
                output.output(),
                routes,
                top_k,
                outputs,
                inputs,
                per_route_input,
                &stream,
            )?;
            let mut expected = Vec::with_capacity(routes * outputs);
            for (route, &expert) in ids.iter().enumerate() {
                let row = if per_route_input {
                    route
                } else {
                    route / top_k
                };
                for out in 0..outputs {
                    let w = &weights[(expert as usize * outputs + out) * inputs..][..inputs];
                    let x = &input[row * inputs..][..inputs];
                    expected.push(bf(w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()));
                }
            }
            close(&output.copy_to_host(&stream)?, &expected, 0.05)?;
            // Eight rows select the single-group kernel; these ids repeat
            // experts within a token, so its items still loop.
            let narrow = 8 * top_k;
            let mut narrow_groups = MoeRouteGroups::new(narrow, experts)?;
            narrow_groups.group_on_stream(&ids_device, narrow, &stream)?;
            grouped_mxfp8_expert(
                &narrow_groups,
                &input_device,
                &weights_device,
                &scales_device,
                output.output(),
                narrow,
                top_k,
                outputs,
                inputs,
                per_route_input,
                &stream,
            )?;
            close(
                &output.copy_to_host(&stream)?[..narrow * outputs],
                &expected[..narrow * outputs],
                0.05,
            )?;
            // Fewer routes select the split-K kernel.
            let few = TILED_ROUTES - top_k;
            let mut few_groups = MoeRouteGroups::new(few, experts)?;
            few_groups.group_on_stream(&ids_device, few, &stream)?;
            grouped_mxfp8_expert(
                &few_groups,
                &input_device,
                &weights_device,
                &scales_device,
                output.output(),
                few,
                top_k,
                outputs,
                inputs,
                per_route_input,
                &stream,
            )?;
            close(
                &output.copy_to_host(&stream)?[..few * outputs],
                &expected[..few * outputs],
                0.05,
            )?;
        }
        Ok(())
    }

    #[test]
    fn relative_attention_applies_lse_sink_gate_per_head() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let mut cache = SinkAttentionCache::new(2, 1)?;
        let mut workspace = SinkAttentionWorkspace::new(1, 2, 16)?;
        let query = DeviceBuffer::zeroed(2 * 128)?;
        let key = DeviceBuffer::zeroed(128)?;
        let gate = DeviceBuffer::zeroed(2 * 128)?;
        let sinks = DeviceBuffer::zeroed(2)?;
        let mut out = DeviceBuffer::zeroed(2 * 128)?;
        let v1 = DeviceBuffer::from_host(&vec![1.0; 128])?;
        // Projection dimension 0 biases distance 0 and dimension 1 distance 1;
        // head 1 has a zero R row and so no bias.
        let mut projection = vec![0u16; 16 * 4];
        projection[0] = 0x3f80;
        projection[4 + 1] = 0x3f80;
        let projection = DeviceBuffer::from_host(&projection)?;
        let mut scratch = DeviceBuffer::zeroed(2 * 4)?;
        let unit = |dimension: usize| {
            DeviceBuffer::from_host(
                &(0..32)
                    .map(|i| if i == dimension { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )
        };
        let r1 = unit(0)?;
        workspace.run_relative(
            RelativeKv::Ring(&mut cache),
            &query,
            &key,
            &v1,
            RelativeBias {
                r: &r1,
                projection: &projection,
                scratch: &mut scratch,
                extent: 4,
            },
            &gate,
            &sinks,
            &mut out,
            1,
            None,
            &stream,
        )?;
        let first = [vec![bf(0.5 / (1.0 + (-1f32).exp())); 128], vec![0.25; 128]].concat();
        close(&out.copy_to_host(&stream)?, &first, 0.0001)?;
        let v2 = DeviceBuffer::from_host(&vec![2.0; 128])?;
        let r2 = unit(1)?;
        workspace.run_relative(
            RelativeKv::Ring(&mut cache),
            &query,
            &key,
            &v2,
            RelativeBias {
                r: &r2,
                projection: &projection,
                scratch: &mut scratch,
                extent: 4,
            },
            &gate,
            &sinks,
            &mut out,
            1,
            None,
            &stream,
        )?;
        let expected_head = 0.5;
        close(
            &out.copy_to_host(&stream)?[0..128],
            &vec![expected_head; 128],
            0.0002,
        )?;
        close(
            &out.copy_to_host(&stream)?[128..256],
            &vec![0.5; 128],
            0.0002,
        )?;
        Ok(())
    }

    #[test]
    fn chunked_relative_attention_matches_single_row_steps() -> Result<()> {
        crate::set_cuda_device(0)?;
        let stream = CudaStream::new_non_blocking()?;
        let (heads, capacity, extent) = (2, 96, 8);
        let value =
            |seed: usize, i: usize| bf((((i * 2654435761 + seed) >> 7) % 97) as f32 / 48.0 - 1.0);
        let rows_for = |seed: usize, rows: usize, width: usize| {
            (0..rows * width)
                .map(|i| value(seed, i))
                .collect::<Vec<_>>()
        };
        // The 110-row chunk exceeds the 96-position local ring.
        let chunks = [3usize, 110];
        let total = chunks.iter().sum::<usize>();
        let q = rows_for(1, total, heads * 128);
        let k = rows_for(2, total, 128);
        let v = rows_for(3, total, 128);
        let rel = rows_for(4, total, heads * 16);
        let projection = DeviceBuffer::from_host(
            &rows_for(7, 16, extent)
                .iter()
                .map(|x| (x.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
        )?;
        let gate = rows_for(5, total, heads * 128);
        let sinks = rows_for(6, total, heads);
        let slice = |values: &[f32], width: usize, first: usize, rows: usize| {
            DeviceBuffer::from_host(&values[first * width..(first + rows) * width])
        };
        let run = |workspace: &mut SinkAttentionWorkspace, sizes: &[usize]| -> Result<Vec<f32>> {
            let mut cache = SinkAttentionCache::new(capacity, 1)?;
            let mut outputs = Vec::new();
            let mut first = 0;
            for &rows in sizes {
                let mut out = DeviceBuffer::zeroed(rows * heads * 128)?;
                workspace.run_relative(
                    RelativeKv::Ring(&mut cache),
                    &slice(&q, heads * 128, first, rows)?,
                    &slice(&k, 128, first, rows)?,
                    &slice(&v, 128, first, rows)?,
                    RelativeBias {
                        r: &slice(&rel, heads * 16, first, rows)?,
                        projection: &projection,
                        scratch: &mut DeviceBuffer::zeroed(rows * heads * extent)?,
                        extent,
                    },
                    &slice(&gate, heads * 128, first, rows)?,
                    &slice(&sinks, heads, first, rows)?,
                    &mut out,
                    rows,
                    None,
                    &stream,
                )?;
                outputs.extend_from_slice(&out.copy_to_host(&stream)?);
                first += rows;
            }
            Ok(outputs)
        };
        let expected = run(&mut SinkAttentionWorkspace::new(1, heads, 256)?, &[1; 113])?;
        // Chunks of 64 or more rows use the tensor-core kernel, whose BF16
        // queries and probabilities differ from the F32 row path by rounding.
        let chunked = run(&mut SinkAttentionWorkspace::new(128, heads, 256)?, &chunks)?;
        close_bf16(&chunked, &expected, 1e-3)?;
        // A 60-row chunk keeps the row-grouped kernel, which a one-row
        // workspace evaluates in one-row groups with identical arithmetic.
        let grouped = run(
            &mut SinkAttentionWorkspace::new(1, heads, 256)?,
            &[3, 60, 50],
        )?;
        close(&grouped, &expected, 0.0)?;

        // Global pages have no window: compare with a ring holding all rows.
        let full = {
            let mut cache = SinkAttentionCache::new(128, 1)?;
            let mut workspace = SinkAttentionWorkspace::new(1, heads, 256)?;
            let mut outputs = Vec::new();
            for first in 0..total {
                let mut out = DeviceBuffer::zeroed(heads * 128)?;
                workspace.run_relative(
                    RelativeKv::Ring(&mut cache),
                    &slice(&q, heads * 128, first, 1)?,
                    &slice(&k, 128, first, 1)?,
                    &slice(&v, 128, first, 1)?,
                    RelativeBias {
                        r: &slice(&rel, heads * 16, first, 1)?,
                        projection: &projection,
                        scratch: &mut DeviceBuffer::zeroed(heads * extent)?,
                        extent,
                    },
                    &slice(&gate, heads * 128, first, 1)?,
                    &slice(&sinks, heads, first, 1)?,
                    &mut out,
                    1,
                    None,
                    &stream,
                )?;
                outputs.extend_from_slice(&out.copy_to_host(&stream)?);
            }
            outputs
        };
        // One page bundle: [layer 1][K/V][128 tokens][1 head][128].
        let page = DeviceBuffer::<u16>::zeroed(2 * 128 * 128)?;
        let table = DeviceBuffer::<DeviceAddress<u16>>::from_host(&[page.cuda_address()])?;
        let mut workspace = SinkAttentionWorkspace::new(128, heads, 256)?;
        let mut paged = Vec::new();
        let mut first = 0;
        for rows in chunks {
            let mut out = DeviceBuffer::zeroed(rows * heads * 128)?;
            workspace.run_relative(
                RelativeKv::Pages {
                    table: &table,
                    layer_offset: 0,
                    position: first,
                    kv_heads: 1,
                },
                &slice(&q, heads * 128, first, rows)?,
                &slice(&k, 128, first, rows)?,
                &slice(&v, 128, first, rows)?,
                RelativeBias {
                    r: &slice(&rel, heads * 16, first, rows)?,
                    projection: &projection,
                    scratch: &mut DeviceBuffer::zeroed(rows * heads * extent)?,
                    extent,
                },
                &slice(&gate, heads * 128, first, rows)?,
                &slice(&sinks, heads, first, rows)?,
                &mut out,
                rows,
                None,
                &stream,
            )?;
            paged.extend_from_slice(&out.copy_to_host(&stream)?);
            first += rows;
        }
        close_bf16(&paged, &full, 1e-3)?;
        Ok(())
    }
}
