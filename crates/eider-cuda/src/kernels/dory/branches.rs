//! Packed prefix and suffix queries with separate causal histories.
use super::*;

unsafe extern "C" {
    fn infer_dory_branch_attention(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        ck: *mut u16,
        cv: *mut u16,
        scratch: *mut f32,
        out: *mut u16,
        destinations: *const u32,
        offsets: *const u32,
        lengths: *const u32,
        pages: *const u32,
        prefix: u32,
        prefix_rows: u32,
        branches: u32,
        rows: u32,
        max_rows: u32,
        stride: u32,
        qh: u32,
        kh: u32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_dory_branch_logits(
        hidden: *const u16,
        weight: *const u16,
        scales: *const f32,
        offsets: *const u32,
        out: *mut f32,
        width: u32,
        labels: u32,
        branches: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// One packed wave of suffixes, optionally including the shared prefix queries.
/// All buffers and their consumers must use the same ordered CUDA stream.
pub struct DoryBranchPlan {
    prefix: usize,
    prefix_rows: usize,
    rows: usize,
    max_rows: usize,
    capacity: usize,
    stride: usize,
    positions: DeviceBuffer<u32>,
    destinations: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    lengths: DeviceBuffer<u32>,
    pages: DeviceBuffer<u32>,
}
impl DoryBranchPlan {
    /// Whether this build includes paged FlashAttention for packed decisions.
    pub const fn supported() -> bool {
        cfg!(eider_flash_attention)
    }

    /// Builds mappings for suffixes, each starting at `prefix`.
    /// `include_prefix` prepends the shared prefix queries to this batch.
    /// Cache space after the prefix is scratch and may be overwritten.
    pub fn new(prefix: usize, lengths: &[usize], include_prefix: bool) -> Result<Self> {
        let prefix_rows = if include_prefix { prefix } else { 0 };
        let rows = lengths
            .iter()
            .try_fold(prefix_rows, |a, &n| a.checked_add(n))
            .ok_or_else(|| invalid("branch row count overflow"))?;
        if !Self::supported()
            || prefix > 1_048_576
            || rows == 0
            || rows > 65535
            || lengths.is_empty()
            || lengths.len() > 64
            || lengths.contains(&0)
        {
            return Err(invalid("branch dimensions or FlashAttention support"));
        }
        let max_suffix = *lengths.iter().max().unwrap();
        let max_rows = max_suffix.max(prefix_rows);
        let stride = (prefix + max_suffix).div_ceil(64);
        let prefix_segments = usize::from(prefix_rows > 0);
        let mut pages = vec![0; stride * (lengths.len() + prefix_segments)];
        let mut positions = Vec::with_capacity(rows);
        let mut destinations = Vec::with_capacity(rows);
        let mut offsets = vec![0];
        let mut used = Vec::with_capacity(lengths.len() + prefix_segments);
        if prefix_rows > 0 {
            for (page, entry) in pages[..prefix.div_ceil(64)].iter_mut().enumerate() {
                *entry = page as u32;
            }
            positions.extend(0..prefix as u32);
            destinations.extend(0..prefix as u32);
            offsets.push(prefix as u32);
            used.push(prefix as u32);
        }
        // Preserve the prefix's partial page as the source for every branch.
        let mut next_page = prefix.div_ceil(64);
        for (branch, &len) in lengths.iter().enumerate() {
            let table = &mut pages
                [(branch + prefix_segments) * stride..(branch + prefix_segments + 1) * stride];
            for (page, entry) in table[..prefix / 64].iter_mut().enumerate() {
                *entry = page as u32;
            }
            let count = (prefix % 64 + len).div_ceil(64);
            for page in 0..count {
                table[prefix / 64 + page] = (next_page + page) as u32;
            }
            for row in 0..len {
                positions.push((prefix + row) as u32);
                destinations.push((next_page * 64 + prefix % 64 + row) as u32);
            }
            offsets.push(positions.len() as u32);
            used.push((prefix + len) as u32);
            next_page += count;
        }
        Ok(Self {
            prefix,
            prefix_rows,
            rows,
            max_rows,
            capacity: next_page * 64,
            stride,
            positions: DeviceBuffer::from_host(&positions)?,
            destinations: DeviceBuffer::from_host(&destinations)?,
            offsets: DeviceBuffer::from_host(&offsets)?,
            lengths: DeviceBuffer::from_host(&used)?,
            pages: DeviceBuffer::from_host(&pages)?,
        })
    }
    /// Packed query rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of independent suffixes.
    pub fn branches(&self) -> usize {
        self.lengths.len() - usize::from(self.prefix_rows > 0)
    }
    /// Minimum physical KV capacity, including page padding.
    pub fn cache_capacity(&self) -> usize {
        self.capacity
    }
    /// Absolute RoPE positions for the packed queries.
    pub fn positions(&self) -> &DeviceBuffer<u32> {
        &self.positions
    }

    /// Appends K/V, clones the partial prefix page, and attends per segment.
    /// Full prefix pages are shared. No branch can address another suffix's pages.
    pub fn attention(
        &self,
        query: &DeviceBuffer<u16>,
        key: &DeviceBuffer<u16>,
        value: &DeviceBuffer<u16>,
        mut cache_key: DeviceOutput<'_, u16>,
        mut cache_value: DeviceOutput<'_, u16>,
        mut scratch: DeviceOutput<'_, f32>,
        mut output: DeviceOutput<'_, u16>,
        qheads: usize,
        kvheads: usize,
        window: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let qn = self.rows.saturating_mul(qheads).saturating_mul(256);
        let kn = self.rows.saturating_mul(kvheads).saturating_mul(256);
        let cn = self.capacity.saturating_mul(kvheads).saturating_mul(256);
        if kvheads == 0
            || qheads == 0
            || !qheads.is_multiple_of(kvheads)
            || cn > u32::MAX as usize
            || qn > u32::MAX as usize
            || window > i32::MAX as usize
            || query.len() < qn
            || key.len() < kn
            || value.len() < kn
            || output.len() < qn
            || cache_key.len() < cn
            || cache_value.len() < cn
            || scratch.len() < self.rows * qheads
        {
            return Err(invalid("branch attention buffers"));
        }
        unsafe {
            check_cuda(
                "infer_dory_branch_attention",
                infer_dory_branch_attention(
                    query.as_const_ptr().cast(),
                    key.as_const_ptr().cast(),
                    value.as_const_ptr().cast(),
                    cache_key.as_mut_ptr().cast(),
                    cache_value.as_mut_ptr().cast(),
                    scratch.as_mut_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    self.destinations.as_const_ptr().cast(),
                    self.offsets.as_const_ptr().cast(),
                    self.lengths.as_const_ptr().cast(),
                    self.pages.as_const_ptr().cast(),
                    self.prefix as u32,
                    self.prefix_rows as u32,
                    self.lengths.len() as u32,
                    self.rows as u32,
                    self.max_rows as u32,
                    self.stride as u32,
                    qheads as u32,
                    kvheads as u32,
                    window as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Projects each suffix's final hidden row onto the selected labels.
    /// Applies the same BF16 rounding and checkpoint scales as the full head.
    pub fn logits(
        &self,
        hidden: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        scales: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        width: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let labels = scales.len();
        if width == 0
            || width > 8192
            || labels == 0
            || labels > 64
            || hidden.len() < self.rows * width
            || weight.len() != labels * width
            || output.len() < self.branches() * labels
        {
            return Err(invalid("branch readout buffers"));
        }
        unsafe {
            check_cuda(
                "infer_dory_branch_logits",
                infer_dory_branch_logits(
                    hidden.as_const_ptr().cast(),
                    weight.as_const_ptr().cast(),
                    scales.as_const_ptr().cast(),
                    self.offsets
                        .as_const_ptr()
                        .cast::<u32>()
                        .add(usize::from(self.prefix_rows > 0)),
                    output.as_mut_ptr().cast(),
                    width as u32,
                    labels as u32,
                    self.branches() as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
}
