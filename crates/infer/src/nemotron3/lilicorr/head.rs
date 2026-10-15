//! Candidate lattice and anchor-conditioned transition factors.

use super::config::LiLiCorrConfig;
use super::linear::{Linear, LinearWorkspace, load_f32, norm, read_bf16};
use eider_cuda::{
    CudaStream, DeviceBuffer, Result, add_f32_prefix_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, lilicorr,
    round_f32_to_bf16_prefix_in_place_on_stream,
};
use eider_format::ModelOptCheckpoint;

struct Layer {
    attention_norm: DeviceBuffer<f32>,
    qkv: Linear,
    output: Linear,
    mlp_norm: DeviceBuffer<f32>,
    up: Linear,
    down: Linear,
}

pub(super) struct Head {
    token: Linear,
    pass: Linear,
    context: Linear,
    feature_norm: DeviceBuffer<f32>,
    feature_bias: DeviceBuffer<f32>,
    feature_up: Linear,
    feature_down: Linear,
    slot: DeviceBuffer<f32>,
    rank: DeviceBuffer<f32>,
    relative: DeviceBuffer<f32>,
    same: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    output_norm: DeviceBuffer<f32>,
    anchor_norm: DeviceBuffer<f32>,
    factor_self: Linear,
    factor_anchor: Linear,
    factor_cross: Linear,
    outgoing: Linear,
    incoming: Linear,
    anchor_outgoing: Linear,
}

pub(super) struct HeadWorkspace {
    token_embeddings: DeviceBuffer<f32>,
    token: DeviceBuffer<f32>,
    pass: DeviceBuffer<f32>,
    feature_input: DeviceBuffer<f32>,
    feature_up: DeviceBuffer<f32>,
    feature: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    update: DeviceBuffer<f32>,
    mlp: DeviceBuffer<f32>,
    anchor_projected: DeviceBuffer<f32>,
    anchor: DeviceBuffer<f32>,
    cross: DeviceBuffer<f32>,
    cross_projection: DeviceBuffer<f32>,
    factor: DeviceBuffer<f32>,
    anchor_factor: DeviceBuffer<f32>,
    outgoing: DeviceBuffer<f32>,
    incoming: DeviceBuffer<f32>,
    anchor_outgoing: DeviceBuffer<f32>,
    pub scores: DeviceBuffer<f32>,
}

impl HeadWorkspace {
    pub(super) fn new(c: &LiLiCorrConfig) -> Result<Self> {
        let d = &c.dflash_config;
        let n = c.draft_tokens() * d.lilicorr_candidate_topk;
        let h = d.lilicorr_hidden_size;
        Ok(Self {
            token_embeddings: DeviceBuffer::zeroed(n * c.hidden_size)?,
            token: DeviceBuffer::zeroed(n * h)?,
            pass: DeviceBuffer::zeroed(c.draft_tokens() * h)?,
            feature_input: DeviceBuffer::zeroed(n * 5)?,
            feature_up: DeviceBuffer::zeroed(n * h)?,
            feature: DeviceBuffer::zeroed(n * h)?,
            hidden: DeviceBuffer::zeroed(n * h)?,
            normalized: DeviceBuffer::zeroed(n * h)?,
            qkv: DeviceBuffer::zeroed(n * 3 * h)?,
            attention: DeviceBuffer::zeroed(n * h)?,
            update: DeviceBuffer::zeroed(n * h)?,
            mlp: DeviceBuffer::zeroed(n * 2 * h)?,
            anchor_projected: DeviceBuffer::zeroed(h)?,
            anchor: DeviceBuffer::zeroed(h)?,
            cross: DeviceBuffer::zeroed(n * h)?,
            cross_projection: DeviceBuffer::zeroed(n * h)?,
            factor: DeviceBuffer::zeroed(n * h)?,
            anchor_factor: DeviceBuffer::zeroed(h)?,
            outgoing: DeviceBuffer::zeroed(n * d.lilicorr_factor_dim)?,
            incoming: DeviceBuffer::zeroed(n * d.lilicorr_factor_dim)?,
            anchor_outgoing: DeviceBuffer::zeroed(d.lilicorr_factor_dim)?,
            scores: DeviceBuffer::zeroed(n * d.lilicorr_candidate_topk)?,
        })
    }
}

impl Head {
    pub(super) fn device_bytes(&self) -> usize {
        [
            &self.token,
            &self.pass,
            &self.context,
            &self.feature_up,
            &self.feature_down,
            &self.factor_self,
            &self.factor_anchor,
            &self.factor_cross,
            &self.outgoing,
            &self.incoming,
            &self.anchor_outgoing,
        ]
        .iter()
        .map(|l| l.device_bytes())
        .sum::<usize>()
            + [
                &self.feature_norm,
                &self.feature_bias,
                &self.slot,
                &self.rank,
                &self.relative,
                &self.same,
                &self.output_norm,
                &self.anchor_norm,
            ]
            .iter()
            .map(|b| b.device_bytes())
            .sum::<usize>()
            + self
                .layers
                .iter()
                .map(|l| {
                    l.attention_norm.device_bytes()
                        + l.mlp_norm.device_bytes()
                        + [&l.qkv, &l.output, &l.up, &l.down]
                            .iter()
                            .map(|l| l.device_bytes())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    pub(super) fn load(checkpoint: &ModelOptCheckpoint, c: &LiLiCorrConfig) -> Result<Self> {
        let d = &c.dflash_config;
        let h = d.lilicorr_hidden_size;
        let mut layers = Vec::with_capacity(d.lilicorr_num_layers);
        for i in 0..d.lilicorr_num_layers {
            let p = format!("lilicorr.layers.{i}");
            layers.push(Layer {
                attention_norm: load_f32(checkpoint, &format!("{p}.attn_norm.weight"), &[h])?,
                qkv: Linear::from_bf16(
                    &read_bf16(checkpoint, &format!("{p}.attn.in_proj_weight"), &[3 * h, h])?,
                    Some(load_f32(
                        checkpoint,
                        &format!("{p}.attn.in_proj_bias"),
                        &[3 * h],
                    )?),
                    3 * h,
                    h,
                )?,
                output: Linear::load(checkpoint, &format!("{p}.attn.out_proj"), h, h, true)?,
                mlp_norm: load_f32(checkpoint, &format!("{p}.mlp_norm.weight"), &[h])?,
                up: Linear::load(checkpoint, &format!("{p}.mlp.up_proj"), 2 * h, h, true)?,
                down: Linear::load(checkpoint, &format!("{p}.mlp.down_proj"), h, 2 * h, true)?,
            });
        }
        // Match the reference's split projection and its intermediate BF16 sums.
        let weight = read_bf16(checkpoint, "lilicorr.factor_input_proj.weight", &[h, 3 * h])?;
        let parts = (0..3)
            .map(|part| {
                weight
                    .chunks_exact(3 * h)
                    .flat_map(|row| row[part * h..(part + 1) * h].iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(Self {
            token: Linear::load(checkpoint, "lilicorr.token_proj", h, c.hidden_size, true)?,
            pass: Linear::load(
                checkpoint,
                "lilicorr.pass_hidden_proj",
                h,
                c.hidden_size,
                true,
            )?,
            context: Linear::load(checkpoint, "lilicorr.context_proj", h, c.hidden_size, true)?,
            feature_norm: load_f32(checkpoint, "lilicorr.feature_norm.weight", &[5])?,
            feature_bias: load_f32(checkpoint, "lilicorr.feature_norm.bias", &[5])?,
            feature_up: Linear::load(checkpoint, "lilicorr.feature_mlp.up_proj", h, 5, true)?,
            feature_down: Linear::load(checkpoint, "lilicorr.feature_mlp.down_proj", h, h, true)?,
            slot: load_f32(
                checkpoint,
                "lilicorr.slot_embedding",
                &[1, 1, c.draft_tokens(), 1, h],
            )?,
            rank: load_f32(
                checkpoint,
                "lilicorr.rank_embedding",
                &[1, 1, 1, d.lilicorr_candidate_topk, h],
            )?,
            relative: load_f32(
                checkpoint,
                "lilicorr.relative_slot_bias",
                &[d.lilicorr_num_heads, 2 * c.block_size - 1],
            )?,
            same: load_f32(
                checkpoint,
                "lilicorr.same_slot_bias",
                &[d.lilicorr_num_heads],
            )?,
            layers,
            output_norm: load_f32(checkpoint, "lilicorr.output_norm.weight", &[h])?,
            anchor_norm: load_f32(checkpoint, "lilicorr.anchor_norm.weight", &[h])?,
            factor_self: Linear::from_bf16(
                &parts[0],
                Some(load_f32(
                    checkpoint,
                    "lilicorr.factor_input_proj.bias",
                    &[h],
                )?),
                h,
                h,
            )?,
            factor_anchor: Linear::from_bf16(&parts[1], None, h, h)?,
            factor_cross: Linear::from_bf16(&parts[2], None, h, h)?,
            outgoing: Linear::load(
                checkpoint,
                "lilicorr.out_head",
                d.lilicorr_factor_dim,
                h,
                true,
            )?,
            incoming: Linear::load(
                checkpoint,
                "lilicorr.in_head",
                d.lilicorr_factor_dim,
                h,
                true,
            )?,
            anchor_outgoing: Linear::load(
                checkpoint,
                "lilicorr.anchor_out_head",
                d.lilicorr_factor_dim,
                h,
                true,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn score(
        &self,
        c: &LiLiCorrConfig,
        embeddings: &DeviceBuffer<u16>,
        ids: &DeviceBuffer<u32>,
        logp: &DeviceBuffer<f32>,
        pass_hidden: &DeviceBuffer<f32>,
        anchor: &DeviceBuffer<f32>,
        slots: usize,
        ws: &mut HeadWorkspace,
        linear: &mut LinearWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let d = &c.dflash_config;
        let k = d.lilicorr_candidate_topk;
        let h = d.lilicorr_hidden_size;
        let n = slots * k;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            c.vocab_size,
            c.hidden_size,
            embeddings,
            ids,
            ws.token_embeddings.output(),
            n,
            stream,
        )?;
        self.token
            .run(&ws.token_embeddings, &mut ws.token, n, linear, stream)?;
        self.pass
            .run(pass_hidden, &mut ws.pass, slots, linear, stream)?;
        lilicorr::features(
            logp,
            &self.feature_norm,
            &self.feature_bias,
            &mut ws.feature_input,
            slots,
            k,
            stream,
        )?;
        self.feature_up
            .run(&ws.feature_input, &mut ws.feature_up, n, linear, stream)?;
        lilicorr::bias_activation(&mut ws.feature_up, None, n, h, true, stream)?;
        self.feature_down
            .run(&ws.feature_up, &mut ws.feature, n, linear, stream)?;
        lilicorr::hidden(
            &ws.token,
            &ws.pass,
            &ws.feature,
            &self.slot,
            &self.rank,
            &mut ws.hidden,
            slots,
            k,
            h,
            stream,
        )?;
        self.context
            .run(anchor, &mut ws.anchor_projected, 1, linear, stream)?;
        norm(
            &ws.anchor_projected,
            &self.anchor_norm,
            &mut ws.anchor,
            1,
            h,
            c.rms_norm_eps,
            stream,
        )?;
        for layer in &self.layers {
            norm(
                &ws.hidden,
                &layer.attention_norm,
                &mut ws.normalized,
                n,
                h,
                c.rms_norm_eps,
                stream,
            )?;
            layer
                .qkv
                .run(&ws.normalized, &mut ws.qkv, n, linear, stream)?;
            lilicorr::attention(
                &ws.qkv,
                &self.relative,
                &self.same,
                &mut ws.attention,
                slots,
                c.draft_tokens(),
                k,
                d.lilicorr_num_heads,
                stream,
            )?;
            layer
                .output
                .run(&ws.attention, &mut ws.update, n, linear, stream)?;
            add_f32_prefix_into_on_stream(
                &ws.hidden,
                &ws.update,
                ws.normalized.output(),
                n * h,
                stream,
            )?;
            round_f32_to_bf16_prefix_in_place_on_stream(ws.normalized.inout(), n * h, stream)?;
            std::mem::swap(&mut ws.hidden, &mut ws.normalized);
            norm(
                &ws.hidden,
                &layer.mlp_norm,
                &mut ws.normalized,
                n,
                h,
                c.rms_norm_eps,
                stream,
            )?;
            layer
                .up
                .run(&ws.normalized, &mut ws.mlp, n, linear, stream)?;
            lilicorr::bias_activation(&mut ws.mlp, None, n, 2 * h, true, stream)?;
            layer.down.run(&ws.mlp, &mut ws.update, n, linear, stream)?;
            add_f32_prefix_into_on_stream(
                &ws.hidden,
                &ws.update,
                ws.normalized.output(),
                n * h,
                stream,
            )?;
            round_f32_to_bf16_prefix_in_place_on_stream(ws.normalized.inout(), n * h, stream)?;
            std::mem::swap(&mut ws.hidden, &mut ws.normalized);
        }
        norm(
            &ws.hidden,
            &self.output_norm,
            &mut ws.normalized,
            n,
            h,
            c.rms_norm_eps,
            stream,
        )?;
        lilicorr::cross(&ws.normalized, &ws.anchor, &mut ws.cross, n, h, stream)?;
        self.factor_self
            .run(&ws.normalized, &mut ws.factor, n, linear, stream)?;
        self.factor_anchor
            .run(&ws.anchor, &mut ws.anchor_factor, 1, linear, stream)?;
        self.factor_cross
            .run(&ws.cross, &mut ws.cross_projection, n, linear, stream)?;
        lilicorr::factors(
            &ws.factor,
            &ws.anchor_factor,
            &ws.cross_projection,
            &mut ws.hidden,
            n,
            h,
            stream,
        )?;
        self.outgoing
            .run(&ws.hidden, &mut ws.outgoing, n, linear, stream)?;
        self.incoming
            .run(&ws.hidden, &mut ws.incoming, n, linear, stream)?;
        self.anchor_outgoing
            .run(&ws.anchor, &mut ws.anchor_outgoing, 1, linear, stream)?;
        lilicorr::normalize(
            &mut ws.outgoing,
            n,
            d.lilicorr_factor_dim,
            d.lilicorr_vector_eps,
            stream,
        )?;
        lilicorr::normalize(
            &mut ws.incoming,
            n,
            d.lilicorr_factor_dim,
            d.lilicorr_vector_eps,
            stream,
        )?;
        lilicorr::normalize(
            &mut ws.anchor_outgoing,
            1,
            d.lilicorr_factor_dim,
            d.lilicorr_vector_eps,
            stream,
        )?;
        lilicorr::edges(
            &ws.outgoing,
            &ws.incoming,
            &ws.anchor_outgoing,
            &mut ws.scores,
            slots,
            k,
            d.lilicorr_factor_dim,
            d.lilicorr_logit_scale,
            stream,
        )
    }
}
