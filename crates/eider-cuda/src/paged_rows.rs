//! Direct-I/O backed embedding-row gathers for coherent unified memory.

#[cfg(not(feature = "cuda-oxide"))]
use crate::cuda::check_cuda;
use crate::cuda::{CudaEvent, CudaStream, DeviceOutput, PageableHostBuffer};
use crate::error::{Error, Result};
#[cfg(not(feature = "cuda-oxide"))]
use crate::ffi;
#[cfg(feature = "cuda-oxide")]
use crate::kernels::core_oxide;
use eider_format::SafeTensorShard;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const DIRECT_IO_ALIGNMENT: usize = 4096;
const PAGE_SLOT_BYTES: usize = 2 * DIRECT_IO_ALIGNMENT;
const MAX_DIRECT_IO_WORKERS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PagedRowShard {
    first_row: usize,
    rows: usize,
    file_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectRowRead {
    file_offset: u64,
    bytes: usize,
    slot: usize,
    row_offset: usize,
}

/// Statistics for one logical row batch populated from direct storage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PagedRowReadStats {
    /// Requested rows, including duplicates.
    pub logical_rows: usize,
    /// Distinct rows issued to storage.
    pub unique_rows: usize,
    /// Aligned bytes requested from storage.
    pub bytes_read: usize,
    /// Wall time spent planning and reading the batch.
    pub elapsed: Duration,
}

/// Storage type for one paged row table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PagedRowFormat {
    /// BF16 values requiring no tensor scale.
    Bf16,
    /// E4M3 values with one positive scale for the complete table.
    Fp8E4m3 {
        /// Multiplier applied after converting each E4M3 value.
        scale: f32,
    },
}

impl PagedRowFormat {
    fn dtype(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Fp8E4m3 { .. } => "F8_E4M3",
        }
    }

    fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4m3 { .. } => 1,
        }
    }

    fn validate(self) -> Result<Self> {
        if let Self::Fp8E4m3 { scale } = self
            && (!scale.is_finite() || scale <= 0.0)
        {
            return Err(Error::Shape {
                label: "paged FP8 row scale",
                expected: "positive finite scale".to_string(),
                actual: scale.to_string(),
            });
        }
        Ok(self)
    }
}

/// One row table split across numbered tensors in a safetensors shard.
pub struct PagedRowSource {
    readers: DirectRowReaderPool,
    shards: Vec<PagedRowShard>,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    format: PagedRowFormat,
}

impl PagedRowSource {
    /// Opens `prefix{index}suffix` tensors as one logical row table.
    pub fn open_numbered(
        shard: &SafeTensorShard,
        prefix: &str,
        suffix: &str,
        shard_count: usize,
        cols: usize,
        format: PagedRowFormat,
    ) -> Result<Self> {
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_DIRECT_IO_WORKERS);
        Self::open_numbered_with_workers(shard, prefix, suffix, shard_count, cols, format, workers)
    }

    /// Opens numbered tensors with an explicit direct-I/O worker count.
    pub fn open_numbered_with_workers(
        shard: &SafeTensorShard,
        prefix: &str,
        suffix: &str,
        shard_count: usize,
        cols: usize,
        format: PagedRowFormat,
        workers: usize,
    ) -> Result<Self> {
        if shard_count == 0 || cols == 0 || workers == 0 {
            return Err(Error::Shape {
                label: "paged row source",
                expected: "positive shard count, columns, and workers".to_string(),
                actual: format!("shards={shard_count} cols={cols} workers={workers}"),
            });
        }
        let format = format.validate()?;
        let row_bytes = cols
            .checked_mul(format.element_bytes())
            .ok_or_else(|| Error::Shape {
                label: "paged row bytes",
                expected: "cols * element bytes without overflow".to_string(),
                actual: cols.to_string(),
            })?;
        if row_bytes > DIRECT_IO_ALIGNMENT {
            return Err(Error::Shape {
                label: "paged row bytes",
                expected: format!("at most {DIRECT_IO_ALIGNMENT} bytes"),
                actual: row_bytes.to_string(),
            });
        }

        let mut shards = Vec::with_capacity(shard_count);
        let mut first_row = 0usize;
        for index in 0..shard_count {
            let name = format!("{prefix}{index}{suffix}");
            let info = shard.require_tensor(&name)?;
            if info.dtype != format.dtype() || info.shape.len() != 2 || info.shape[1] != cols {
                return Err(Error::Shape {
                    label: "paged row tensor",
                    expected: format!("dtype={} shape=[rows, {cols}]", format.dtype()),
                    actual: format!("{name}: dtype={} shape={:?}", info.dtype, info.shape),
                });
            }
            let rows = info.shape[0];
            let range = shard.tensor_file_range(&name)?;
            let expected_bytes = rows.checked_mul(row_bytes).ok_or_else(|| Error::Shape {
                label: "paged row tensor bytes",
                expected: "rows * row bytes without overflow".to_string(),
                actual: format!("rows={rows} row_bytes={row_bytes}"),
            })?;
            if range.end - range.start != expected_bytes as u64 {
                return Err(Error::Shape {
                    label: "paged row tensor bytes",
                    expected: expected_bytes.to_string(),
                    actual: (range.end - range.start).to_string(),
                });
            }
            shards.push(PagedRowShard {
                first_row,
                rows,
                file_offset: range.start,
            });
            first_row = first_row.checked_add(rows).ok_or_else(|| Error::Shape {
                label: "paged row count",
                expected: "row sum without overflow".to_string(),
                actual: format!("current={first_row} next={rows}"),
            })?;
        }
        let path = shard.path().to_path_buf();
        let direct_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .map_err(|error| row_io_error("open direct", &path, error))?;
        Ok(Self {
            readers: DirectRowReaderPool::new(direct_file, path, workers)?,
            shards,
            rows: first_row,
            cols,
            row_bytes,
            format,
        })
    }

    /// Total logical rows across the numbered tensors.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Width of each row.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Reads unique logical rows into final GPU-visible system-memory pages.
    pub fn read_rows(
        &mut self,
        row_ids: &[u32],
        batch: &mut PagedRowBatch,
    ) -> Result<PagedRowReadStats> {
        if row_ids.is_empty()
            || row_ids.len() > batch.capacity
            || batch.cols != self.cols
            || batch.format != self.format
        {
            return Err(Error::Shape {
                label: "paged row batch",
                expected: format!(
                    "1..={} rows with {} columns and {:?}",
                    batch.capacity, self.cols, self.format
                ),
                actual: format!(
                    "rows={} cols={} format={:?}",
                    row_ids.len(),
                    batch.cols,
                    batch.format
                ),
            });
        }
        let started = Instant::now();
        let mut unique_slots = HashMap::with_capacity(row_ids.len());
        let mut reads = Vec::with_capacity(row_ids.len());
        for (output_row, &row_id) in row_ids.iter().enumerate() {
            let row_id = row_id as usize;
            if row_id >= self.rows {
                return Err(Error::Shape {
                    label: "paged row ID",
                    expected: format!("row < {}", self.rows),
                    actual: row_id.to_string(),
                });
            }
            let slot = if let Some(&slot) = unique_slots.get(&row_id) {
                slot
            } else {
                let slot = reads.len();
                let absolute = self.row_file_offset(row_id)?;
                let aligned = absolute / DIRECT_IO_ALIGNMENT as u64 * DIRECT_IO_ALIGNMENT as u64;
                let row_offset = (absolute - aligned) as usize;
                let bytes = align_up(row_offset + self.row_bytes, DIRECT_IO_ALIGNMENT)?;
                reads.push(DirectRowRead {
                    file_offset: aligned,
                    bytes,
                    slot,
                    row_offset,
                });
                unique_slots.insert(row_id, slot);
                slot
            };
            let offset = slot
                .checked_mul(PAGE_SLOT_BYTES)
                .and_then(|value| value.checked_add(reads[slot].row_offset))
                .ok_or_else(|| Error::Shape {
                    label: "paged row offset",
                    expected: "slot offset without overflow".to_string(),
                    actual: format!("slot={slot} row_offset={}", reads[slot].row_offset),
                })?;
            batch.offsets.as_mut_slice()[output_row] =
                u32::try_from(offset).map_err(|_| Error::Shape {
                    label: "paged row offset",
                    expected: "offset fitting u32".to_string(),
                    actual: offset.to_string(),
                })?;
        }

        let active_bytes = reads.len() * PAGE_SLOT_BYTES;
        self.readers.read_rows(
            &reads,
            &mut batch.pages.as_mut_slice()[..active_bytes],
            self.row_bytes,
        )?;
        batch.row_count = row_ids.len();
        Ok(PagedRowReadStats {
            logical_rows: row_ids.len(),
            unique_rows: reads.len(),
            bytes_read: reads.iter().map(|read| read.bytes).sum(),
            elapsed: started.elapsed(),
        })
    }

    fn row_file_offset(&self, row: usize) -> Result<u64> {
        let shard = self
            .shards
            .iter()
            .rev()
            .find(|shard| row >= shard.first_row)
            .ok_or_else(|| Error::Shape {
                label: "paged row shard",
                expected: "row covered by a shard".to_string(),
                actual: row.to_string(),
            })?;
        let local_row = row - shard.first_row;
        if local_row >= shard.rows {
            return Err(Error::Shape {
                label: "paged row shard",
                expected: format!("local row < {}", shard.rows),
                actual: local_row.to_string(),
            });
        }
        let row_offset = local_row
            .checked_mul(self.row_bytes)
            .ok_or_else(|| Error::Shape {
                label: "paged row file offset",
                expected: "local row * row bytes without overflow".to_string(),
                actual: format!("local_row={local_row} row_bytes={}", self.row_bytes),
            })?;
        shard
            .file_offset
            .checked_add(row_offset as u64)
            .ok_or_else(|| Error::Shape {
                label: "paged row file offset",
                expected: "tensor offset + row offset without overflow".to_string(),
                actual: format!(
                    "tensor={} local_row={} row_bytes={}",
                    shard.file_offset, local_row, self.row_bytes
                ),
            })
    }
}

/// Reusable direct-read pages and stable row-offset storage.
pub struct PagedRowBatch {
    pages: PageableHostBuffer<u8>,
    offsets: PageableHostBuffer<u32>,
    capacity: usize,
    cols: usize,
    row_count: usize,
    format: PagedRowFormat,
}

impl PagedRowBatch {
    /// Allocates stable direct-I/O pages for at most `capacity` gathered rows.
    pub fn new(capacity: usize, cols: usize, format: PagedRowFormat) -> Result<Self> {
        if capacity == 0 || cols == 0 {
            return Err(Error::Shape {
                label: "paged row batch",
                expected: "positive capacity and columns".to_string(),
                actual: format!("capacity={capacity} cols={cols}"),
            });
        }
        let page_bytes = capacity
            .checked_mul(PAGE_SLOT_BYTES)
            .ok_or_else(|| Error::Shape {
                label: "paged row page storage",
                expected: "capacity * page slot bytes without overflow".to_string(),
                actual: capacity.to_string(),
            })?;
        if page_bytes > u32::MAX as usize {
            return Err(Error::Shape {
                label: "paged row page storage",
                expected: "page offsets fitting u32".to_string(),
                actual: page_bytes.to_string(),
            });
        }
        Ok(Self {
            pages: PageableHostBuffer::zeroed(page_bytes)?,
            offsets: PageableHostBuffer::zeroed(capacity)?,
            capacity,
            cols,
            row_count: 0,
            format: format.validate()?,
        })
    }

    /// Maximum number of logical rows accepted by this batch.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Logical rows populated by the latest successful read.
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Resident bytes used by direct-I/O pages and row offsets.
    pub fn storage_bytes(&self) -> usize {
        self.pages.bytes() + self.offsets.bytes()
    }

    /// Gathers the most recently read rows into contiguous F32 rows.
    pub fn gather_into_on_stream(
        &self,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let values = self
            .row_count
            .checked_mul(self.cols)
            .ok_or_else(|| Error::Shape {
                label: "paged row gather output",
                expected: "rows * cols without overflow".to_string(),
                actual: format!("rows={} cols={}", self.row_count, self.cols),
            })?;
        if self.row_count == 0 || output.len() < values {
            return Err(Error::Shape {
                label: "paged row gather output",
                expected: format!("at least {values} values for a non-empty batch"),
                actual: output.len().to_string(),
            });
        }
        match self.format {
            PagedRowFormat::Bf16 => unsafe {
                #[cfg(feature = "cuda-oxide")]
                {
                    core_oxide::paged_bf16_rows_to_f32(
                        self.pages.as_ptr(),
                        self.offsets.as_ptr(),
                        output.as_mut_ptr().cast(),
                        self.row_count as u32,
                        self.cols as u32,
                        stream.as_raw(),
                    )
                }
                #[cfg(not(feature = "cuda-oxide"))]
                {
                    check_cuda(
                        "infer_paged_bf16_rows_to_f32_on_stream",
                        ffi::infer_paged_bf16_rows_to_f32_on_stream(
                            self.pages.as_ptr(),
                            self.offsets.as_ptr(),
                            output.as_mut_ptr().cast(),
                            self.row_count as u32,
                            self.cols as u32,
                            stream.as_raw(),
                        ),
                    )
                }
            },
            PagedRowFormat::Fp8E4m3 { scale } => unsafe {
                #[cfg(feature = "cuda-oxide")]
                {
                    core_oxide::paged_fp8_rows_to_f32(
                        self.pages.as_ptr(),
                        self.offsets.as_ptr(),
                        output.as_mut_ptr().cast(),
                        self.row_count as u32,
                        self.cols as u32,
                        scale,
                        stream.as_raw(),
                    )
                }
                #[cfg(not(feature = "cuda-oxide"))]
                {
                    check_cuda(
                        "infer_paged_fp8_rows_to_f32_on_stream",
                        ffi::infer_paged_fp8_rows_to_f32_on_stream(
                            self.pages.as_ptr(),
                            self.offsets.as_ptr(),
                            output.as_mut_ptr().cast(),
                            self.row_count as u32,
                            self.cols as u32,
                            scale,
                            stream.as_raw(),
                        ),
                    )
                }
            },
        }
    }
}

struct PagedRowBatchSlot {
    index: usize,
    batch: PagedRowBatch,
}

struct PagedRowReadRequest {
    row_ids: Vec<u32>,
    slot: PagedRowBatchSlot,
}

struct PagedRowReadResponse {
    row_ids: Vec<u32>,
    slot: PagedRowBatchSlot,
    result: Result<PagedRowReadStats>,
}

/// Double-buffered asynchronous reader for a paged row source.
///
/// Call [`Self::begin_rows`] before unrelated GPU work, then call
/// [`Self::gather_into_on_stream`] at the first consumer. The gather waits only
/// for any storage work that did not overlap. A per-batch CUDA event prevents
/// the reader threads from overwriting pages still visible to the GPU.
pub struct PagedRowReader {
    requests: Option<SyncSender<PagedRowReadRequest>>,
    responses: Receiver<PagedRowReadResponse>,
    worker: Option<JoinHandle<()>>,
    free: Vec<PagedRowBatchSlot>,
    busy: Vec<PagedRowBatchSlot>,
    ready: Option<(PagedRowBatchSlot, PagedRowReadStats)>,
    reuse_events: Vec<CudaEvent>,
    row_id_buffers: Vec<Vec<u32>>,
    row_capacity: usize,
    cols: usize,
    storage_bytes: usize,
    pending: bool,
}

impl PagedRowReader {
    /// Starts a dedicated reader around `source` with two reusable batches.
    pub fn new(source: PagedRowSource, row_capacity: usize) -> Result<Self> {
        if row_capacity == 0 {
            return Err(Error::Shape {
                label: "paged asynchronous row reader",
                expected: "positive row capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let cols = source.cols();
        let mut free = Vec::with_capacity(2);
        let mut reuse_events = Vec::with_capacity(2);
        let mut storage_bytes = 0usize;
        for index in 0..2 {
            let batch = PagedRowBatch::new(row_capacity, cols, source.format)?;
            storage_bytes = storage_bytes.saturating_add(batch.storage_bytes());
            free.push(PagedRowBatchSlot { index, batch });
            reuse_events.push(CudaEvent::new_sync()?);
        }

        let (request_tx, request_rx) = mpsc::sync_channel::<PagedRowReadRequest>(1);
        let (response_tx, responses) = mpsc::channel::<PagedRowReadResponse>();
        let worker = std::thread::Builder::new()
            .name("eider-paged-rows".to_string())
            .spawn(move || {
                let mut source = source;
                while let Ok(mut request) = request_rx.recv() {
                    let result = source.read_rows(&request.row_ids, &mut request.slot.batch);
                    if response_tx
                        .send(PagedRowReadResponse {
                            row_ids: request.row_ids,
                            slot: request.slot,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|error| Error::Format {
                label: "paged asynchronous row reader",
                detail: format!("spawn coordinator: {error}"),
            })?;
        Ok(Self {
            requests: Some(request_tx),
            responses,
            worker: Some(worker),
            free,
            busy: Vec::with_capacity(2),
            ready: None,
            reuse_events,
            row_id_buffers: vec![Vec::with_capacity(row_capacity)],
            row_capacity,
            cols,
            storage_bytes,
            pending: false,
        })
    }

    /// Starts reading one row batch without waiting for storage completion.
    pub fn begin_rows(&mut self, row_ids: &[u32]) -> Result<()> {
        if row_ids.is_empty() || row_ids.len() > self.row_capacity {
            return Err(Error::Shape {
                label: "paged asynchronous rows",
                expected: format!("1..={} rows", self.row_capacity),
                actual: row_ids.len().to_string(),
            });
        }
        if self.pending || self.ready.is_some() {
            return Err(Error::Format {
                label: "paged asynchronous row reader",
                detail: "the previous row batch has not been gathered".to_string(),
            });
        }

        let slot = if let Some(slot) = self.free.pop() {
            slot
        } else {
            let slot = self.busy.pop().ok_or_else(|| Error::Format {
                label: "paged asynchronous row reader",
                detail: "no reusable batch is available".to_string(),
            })?;
            self.reuse_events[slot.index].synchronize()?;
            slot
        };
        let mut owned_ids = self
            .row_id_buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.row_capacity));
        owned_ids.extend_from_slice(row_ids);
        let request = PagedRowReadRequest {
            row_ids: owned_ids,
            slot,
        };
        let Some(sender) = &self.requests else {
            return Err(Error::Format {
                label: "paged asynchronous row reader",
                detail: "reader coordinator has stopped".to_string(),
            });
        };
        if let Err(error) = sender.send(request) {
            let mut request = error.0;
            request.row_ids.clear();
            self.row_id_buffers.push(request.row_ids);
            self.free.push(request.slot);
            return Err(Error::Format {
                label: "paged asynchronous row reader",
                detail: "reader coordinator has stopped".to_string(),
            });
        }
        self.pending = true;
        Ok(())
    }

    /// Waits for the active storage read while retaining its batch for gather.
    pub fn wait_ready(&mut self) -> Result<PagedRowReadStats> {
        if let Some((_, stats)) = self.ready.as_ref() {
            return Ok(*stats);
        }
        if !self.pending {
            return Err(Error::Format {
                label: "paged asynchronous row reader",
                detail: "no row batch is in flight".to_string(),
            });
        }
        let mut response = self.responses.recv().map_err(|error| Error::Format {
            label: "paged asynchronous row reader",
            detail: format!("reader coordinator stopped: {error}"),
        })?;
        self.pending = false;
        response.row_ids.clear();
        self.row_id_buffers.push(response.row_ids);
        match response.result {
            Ok(stats) => {
                self.ready = Some((response.slot, stats));
                Ok(stats)
            }
            Err(error) => {
                self.free.push(response.slot);
                Err(error)
            }
        }
    }

    /// Waits for the active read and gathers it into contiguous F32 rows.
    pub fn gather_into_on_stream(
        &mut self,
        output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<PagedRowReadStats> {
        let stats = self.wait_ready()?;
        let (slot, _) = self.ready.take().expect("wait_ready populated batch");
        if let Err(error) = slot.batch.gather_into_on_stream(output, stream) {
            self.free.push(slot);
            return Err(error);
        }
        if let Err(error) = self.reuse_events[slot.index].record_on_stream(stream) {
            stream.synchronize()?;
            self.free.push(slot);
            return Err(error);
        }
        self.busy.push(slot);
        Ok(stats)
    }

    /// Resident bytes used by both reusable direct-I/O batches.
    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }

    /// Width of each gathered row.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl Drop for PagedRowReader {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        for slot in &self.busy {
            let _ = self.reuse_events[slot.index].synchronize();
        }
    }
}

struct DirectRowReaderPool {
    senders: Vec<SyncSender<DirectReadMessage>>,
    completions: Receiver<Result<()>>,
    workers: Vec<JoinHandle<()>>,
}

struct DirectReadJob {
    destination: *mut u8,
    destination_bytes: usize,
    file_offset: u64,
    row_offset: usize,
    row_bytes: usize,
}

// Each job points into one uniquely borrowed batch slot. `read_rows` waits for
// every dispatched job before returning the borrow to its caller.
unsafe impl Send for DirectReadJob {}

enum DirectReadMessage {
    Read(DirectReadJob),
    Shutdown,
}

impl DirectRowReaderPool {
    fn new(file: File, path: PathBuf, worker_count: usize) -> Result<Self> {
        let worker_count = worker_count.max(1);
        let (completion_tx, completions) = mpsc::channel();
        let mut senders = Vec::with_capacity(worker_count);
        let mut workers = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let worker_file = file
                .try_clone()
                .map_err(|error| row_io_error("clone direct file", &path, error))?;
            let worker_path = path.clone();
            let worker_completion_tx = completion_tx.clone();
            let (sender, receiver) = mpsc::sync_channel(1);
            let worker = std::thread::Builder::new()
                .name(format!("eider-paged-row-{index}"))
                .spawn(move || {
                    while let Ok(message) = receiver.recv() {
                        let DirectReadMessage::Read(job) = message else {
                            break;
                        };
                        let result = read_direct_job(&worker_file, &worker_path, job);
                        if worker_completion_tx.send(result).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|error| Error::Format {
                    label: "paged BF16 row reader",
                    detail: format!("spawn worker {index}: {error}"),
                })?;
            senders.push(sender);
            workers.push(worker);
        }
        Ok(Self {
            senders,
            completions,
            workers,
        })
    }

    fn read_rows(
        &mut self,
        reads: &[DirectRowRead],
        pages: &mut [u8],
        row_bytes: usize,
    ) -> Result<()> {
        let mut dispatched = 0usize;
        let mut dispatch_error = None;
        for read in reads {
            let destination_offset = read.slot * PAGE_SLOT_BYTES;
            let destination = pages[destination_offset..].as_mut_ptr();
            let address = destination as usize;
            if !address.is_multiple_of(DIRECT_IO_ALIGNMENT) {
                dispatch_error = Some(Error::Format {
                    label: "paged BF16 direct read",
                    detail: format!(
                        "destination address 0x{address:x} is not {DIRECT_IO_ALIGNMENT}-byte aligned"
                    ),
                });
                break;
            }
            let job = DirectReadJob {
                destination,
                destination_bytes: read.bytes,
                file_offset: read.file_offset,
                row_offset: read.row_offset,
                row_bytes,
            };
            let worker = dispatched % self.senders.len();
            if self.senders[worker]
                .send(DirectReadMessage::Read(job))
                .is_err()
            {
                dispatch_error = Some(Error::Format {
                    label: "paged BF16 row reader",
                    detail: format!("worker {worker} stopped"),
                });
                break;
            }
            dispatched += 1;
        }

        let mut read_error = None;
        for _ in 0..dispatched {
            match self.completions.recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) if read_error.is_none() => read_error = Some(error),
                Ok(Err(_)) => {}
                Err(error) if read_error.is_none() => {
                    read_error = Some(Error::Format {
                        label: "paged BF16 row reader",
                        detail: format!("completion channel stopped: {error}"),
                    });
                }
                Err(_) => {}
            }
        }
        if let Some(error) = dispatch_error.or(read_error) {
            return Err(error);
        }
        Ok(())
    }
}

impl Drop for DirectRowReaderPool {
    fn drop(&mut self) {
        for sender in &self.senders {
            let _ = sender.send(DirectReadMessage::Shutdown);
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn read_direct_job(file: &File, path: &Path, job: DirectReadJob) -> Result<()> {
    let destination =
        unsafe { std::slice::from_raw_parts_mut(job.destination, job.destination_bytes) };
    let bytes = pread_direct(file, destination, job.file_offset)
        .map_err(|error| row_io_error("read direct", path, error))?;
    let required = job.row_offset + job.row_bytes;
    if bytes < required {
        return Err(Error::Format {
            label: "paged BF16 direct read",
            detail: format!(
                "short read at {}: {bytes} bytes, need {required}",
                job.file_offset
            ),
        });
    }
    Ok(())
}

fn pread_direct(file: &File, destination: &mut [u8], offset: u64) -> std::io::Result<usize> {
    loop {
        let bytes = unsafe {
            libc::pread(
                file.as_raw_fd(),
                destination.as_mut_ptr().cast(),
                destination.len(),
                offset as libc::off_t,
            )
        };
        if bytes >= 0 {
            return Ok(bytes as usize);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| Error::Shape {
            label: "paged BF16 direct-read alignment",
            expected: "aligned value without overflow".to_string(),
            actual: format!("value={value} alignment={alignment}"),
        })
}

fn row_io_error(operation: &'static str, path: &Path, error: std::io::Error) -> Error {
    Error::Format {
        label: "paged row source",
        detail: format!("{operation} {}: {error}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::{PagedRowBatch, PagedRowFormat, PagedRowReader, PagedRowSource};
    use crate::format::{bf16_to_f32, e4m3_value, f32_to_bf16};
    use crate::{CudaStream, DeviceBuffer};
    use eider_format::SafeTensorShard;
    use serde_json::json;
    use serial_test::serial;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    const COLS: usize = 16;

    #[serial]
    #[test]
    fn direct_paged_rows_match_numbered_bf16_tensors() {
        let path = fixture_path();
        let rows = (0..5)
            .flat_map(|row| (0..COLS).map(move |col| f32_to_bf16(row as f32 + col as f32 / 32.0)))
            .collect::<Vec<_>>();
        write_fixture(&path, &rows);

        let shard = SafeTensorShard::open(&path).expect("fixture shard");
        let mut source = PagedRowSource::open_numbered(
            &shard,
            "table.shard_",
            ".weight",
            2,
            COLS,
            PagedRowFormat::Bf16,
        )
        .expect("paged source");
        assert_eq!(source.rows(), 5);
        assert_eq!(source.cols(), COLS);

        let ids = [4, 0, 4, 2];
        let mut batch = PagedRowBatch::new(ids.len(), COLS, PagedRowFormat::Bf16).expect("batch");
        let stats = source.read_rows(&ids, &mut batch).expect("direct rows");
        assert_eq!(stats.logical_rows, 4);
        assert_eq!(stats.unique_rows, 3);
        assert_eq!(batch.row_count(), ids.len());

        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut output = DeviceBuffer::zeroed(ids.len() * COLS).expect("output");
        batch
            .gather_into_on_stream(output.output(), &stream)
            .expect("gather");
        let actual = output.copy_to_host(&stream).expect("readback");
        let expected = ids
            .iter()
            .flat_map(|&row| {
                let row = row as usize;
                rows[row * COLS..(row + 1) * COLS]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);

        fs::remove_file(path).expect("remove fixture");
    }

    #[serial]
    #[test]
    fn asynchronous_reader_reuses_batches_after_cuda_gather() {
        let path = fixture_path();
        let rows = (0..5)
            .flat_map(|row| (0..COLS).map(move |col| f32_to_bf16(row as f32 + col as f32 / 32.0)))
            .collect::<Vec<_>>();
        write_fixture(&path, &rows);

        let shard = SafeTensorShard::open(&path).expect("fixture shard");
        let source = PagedRowSource::open_numbered(
            &shard,
            "table.shard_",
            ".weight",
            2,
            COLS,
            PagedRowFormat::Bf16,
        )
        .expect("paged source");
        let mut reader = PagedRowReader::new(source, 2).expect("asynchronous reader");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut output = DeviceBuffer::zeroed(2 * COLS).expect("output");

        for ids in [[4, 0], [1, 3], [2, 2]] {
            reader.begin_rows(&ids).expect("begin rows");
            let stats = reader
                .gather_into_on_stream(output.output(), &stream)
                .expect("gather rows");
            assert_eq!(stats.logical_rows, 2);
        }
        let actual = output.copy_to_host(&stream).expect("readback");
        let expected = [2usize, 2]
            .into_iter()
            .flat_map(|row| {
                rows[row * COLS..(row + 1) * COLS]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);

        drop(reader);
        fs::remove_file(path).expect("remove fixture");
    }

    #[serial]
    #[test]
    fn direct_paged_rows_scale_fp8_tensors() {
        let path = fixture_path();
        let rows = (0..5 * COLS)
            .map(|index| [0x00, 0x28, 0x30, 0x38, 0xb8][index % 5])
            .collect::<Vec<_>>();
        write_fp8_fixture(&path, &rows);
        let shard = SafeTensorShard::open(&path).expect("fixture shard");
        let format = PagedRowFormat::Fp8E4m3 { scale: 0.25 };
        let mut source =
            PagedRowSource::open_numbered(&shard, "table.shard_", ".weight", 2, COLS, format)
                .expect("paged source");
        let ids = [4, 0, 2];
        let mut batch = PagedRowBatch::new(ids.len(), COLS, format).expect("batch");
        source.read_rows(&ids, &mut batch).expect("direct rows");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut output = DeviceBuffer::zeroed(ids.len() * COLS).expect("output");
        batch
            .gather_into_on_stream(output.output(), &stream)
            .expect("gather");
        let actual = output.copy_to_host(&stream).expect("readback");
        let expected = ids
            .iter()
            .flat_map(|&row| {
                let row = row as usize;
                rows[row * COLS..(row + 1) * COLS]
                    .iter()
                    .map(|&code| e4m3_value(code) * 0.25)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);
        fs::remove_file(path).expect("remove fixture");
    }

    fn fixture_path() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "eider-paged-bf16-{}-{unique}.safetensors",
            std::process::id()
        ))
    }

    fn write_fixture(path: &PathBuf, rows: &[u16]) {
        let first_bytes = 2 * COLS * 2;
        let total_bytes = rows.len() * 2;
        let header = json!({
            "table.shard_0.weight": {
                "dtype": "BF16",
                "shape": [2, COLS],
                "data_offsets": [0, first_bytes]
            },
            "table.shard_1.weight": {
                "dtype": "BF16",
                "shape": [3, COLS],
                "data_offsets": [first_bytes, total_bytes]
            }
        });
        let mut header = serde_json::to_vec(&header).expect("header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = File::create(path).expect("create fixture");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("header length");
        file.write_all(&header).expect("header");
        for &value in rows {
            file.write_all(&value.to_le_bytes()).expect("row value");
        }
        file.set_len(8192).expect("direct-I/O padding");
        file.sync_all().expect("sync fixture");
    }

    fn write_fp8_fixture(path: &PathBuf, rows: &[u8]) {
        let first_bytes = 2 * COLS;
        let header = json!({
            "table.shard_0.weight": {
                "dtype": "F8_E4M3",
                "shape": [2, COLS],
                "data_offsets": [0, first_bytes]
            },
            "table.shard_1.weight": {
                "dtype": "F8_E4M3",
                "shape": [3, COLS],
                "data_offsets": [first_bytes, rows.len()]
            }
        });
        let mut header = serde_json::to_vec(&header).expect("header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = File::create(path).expect("create fixture");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("header length");
        file.write_all(&header).expect("header");
        file.write_all(rows).expect("row values");
        file.set_len(8192).expect("direct-I/O padding");
        file.sync_all().expect("sync fixture");
    }
}
