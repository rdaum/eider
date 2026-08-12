//! Minimal CUDA backend with a device-resident logical page table.
//!
//! The example intentionally uses the CUDA Driver API directly so the cache
//! crate does not depend on Eider's CUDA implementation. One kernel launch
//! writes a logical append across every physical page in its reservation.

use std::error::Error;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::mem::size_of;
use std::ptr::null_mut;

use sequence_cache::{
    AdmissionOutcome, AdmissionRequest, BackendAppendCommit, BackendAppendPage, CacheConfig,
    PageAllocation, PageBackend, RetireError, RetireOutcome, SequenceCache, SequenceId,
};

const PAGE_TOKENS: usize = 4;
const MAX_PHYSICAL_PAGES: usize = 8;
const MAX_SEQUENCE_TOKENS: usize = 16;
const THREADS: u32 = 128;

type CuDevice = c_int;
type CuDevicePtr = u64;
type CuResult = c_int;
type CuContext = *mut c_void;
type CuStream = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CU_STREAM_NON_BLOCKING: c_uint = 1;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> CuResult;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> CuResult;
    fn cuDevicePrimaryCtxRetain(context: *mut CuContext, device: CuDevice) -> CuResult;
    fn cuDevicePrimaryCtxRelease(device: CuDevice) -> CuResult;
    fn cuCtxSetCurrent(context: CuContext) -> CuResult;
    fn cuStreamCreate(stream: *mut CuStream, flags: c_uint) -> CuResult;
    fn cuStreamDestroy_v2(stream: CuStream) -> CuResult;
    fn cuStreamSynchronize(stream: CuStream) -> CuResult;
    fn cuMemAlloc_v2(pointer: *mut CuDevicePtr, bytes: usize) -> CuResult;
    fn cuMemFree_v2(pointer: CuDevicePtr) -> CuResult;
    fn cuMemcpyHtoD_v2(destination: CuDevicePtr, source: *const c_void, bytes: usize) -> CuResult;
    fn cuMemcpyDtoH_v2(destination: *mut c_void, source: CuDevicePtr, bytes: usize) -> CuResult;
    fn cuMemcpyDtoD_v2(destination: CuDevicePtr, source: CuDevicePtr, bytes: usize) -> CuResult;
    fn cuModuleLoadData(module: *mut CuModule, image: *const c_void) -> CuResult;
    fn cuModuleUnload(module: CuModule) -> CuResult;
    fn cuModuleGetFunction(
        function: *mut CuFunction,
        module: CuModule,
        name: *const c_char,
    ) -> CuResult;
    fn cuLaunchKernel(
        function: CuFunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory_bytes: c_uint,
        stream: CuStream,
        kernel_parameters: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;
}

const WRITE_PAGED_ROWS_PTX: &str = r#"
.version 7.0
.target sm_52
.address_size 64

.visible .entry write_paged_rows(
    .param .u64 table_param,
    .param .u32 start_param,
    .param .u32 rows_param,
    .param .u32 page_tokens_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [table_param];
    ld.param.u32 %r1, [start_param];
    ld.param.u32 %r2, [rows_param];
    ld.param.u32 %r3, [page_tokens_param];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r2;
    @%p1 bra done;

    add.u32 %r8, %r1, %r7;
    div.u32 %r9, %r8, %r3;
    rem.u32 %r10, %r8, %r3;
    mul.wide.u32 %rd2, %r9, 8;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u64 %rd4, [%rd3];
    mul.wide.u32 %rd5, %r10, 4;
    add.s64 %rd6, %rd4, %rd5;
    add.u32 %r11, %r8, 1;
    cvt.rn.f32.u32 %f1, %r11;
    st.global.f32 [%rd6], %f1;

done:
    ret;
}
"#;

#[derive(Debug)]
enum CudaBackendError {
    Driver { call: &'static str, code: CuResult },
    OutOfPages,
    StalePage(usize),
    InvalidGeometry(&'static str),
}

impl fmt::Display for CudaBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver { call, code } => {
                write!(formatter, "{call} failed with CUDA error {code}")
            }
            Self::OutOfPages => formatter.write_str("CUDA page pool is exhausted"),
            Self::StalePage(slot) => write!(formatter, "CUDA page slot {slot} is not live"),
            Self::InvalidGeometry(detail) => formatter.write_str(detail),
        }
    }
}

impl Error for CudaBackendError {}

fn check_cuda(call: &'static str, result: CuResult) -> Result<(), CudaBackendError> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaBackendError::Driver { call, code: result })
    }
}

struct DeviceAllocation {
    pointer: CuDevicePtr,
}

impl DeviceAllocation {
    fn new(bytes: usize) -> Result<Self, CudaBackendError> {
        let mut pointer = 0;
        unsafe {
            check_cuda("cuMemAlloc", cuMemAlloc_v2(&mut pointer, bytes))?;
        }
        Ok(Self { pointer })
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if self.pointer != 0 {
            unsafe {
                let _ = cuMemFree_v2(self.pointer);
            }
        }
    }
}

struct CudaDevice {
    device: CuDevice,
    context: CuContext,
    stream: CuStream,
    module: CuModule,
    write_paged_rows: CuFunction,
}

impl CudaDevice {
    fn new() -> Result<Self, CudaBackendError> {
        let mut device = 0;
        let mut context = null_mut();
        let mut stream = null_mut();
        let mut module = null_mut();
        let mut write_paged_rows = null_mut();
        let ptx = CString::new(WRITE_PAGED_ROWS_PTX)
            .expect("embedded PTX must not contain an interior NUL");
        let kernel_name = c"write_paged_rows";

        unsafe {
            check_cuda("cuInit", cuInit(0))?;
            check_cuda("cuDeviceGet", cuDeviceGet(&mut device, 0))?;
            check_cuda(
                "cuDevicePrimaryCtxRetain",
                cuDevicePrimaryCtxRetain(&mut context, device),
            )?;
            if let Err(error) = check_cuda("cuCtxSetCurrent", cuCtxSetCurrent(context)) {
                let _ = cuDevicePrimaryCtxRelease(device);
                return Err(error);
            }
            if let Err(error) = check_cuda(
                "cuStreamCreate",
                cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING),
            ) {
                let _ = cuDevicePrimaryCtxRelease(device);
                return Err(error);
            }
            if let Err(error) = check_cuda(
                "cuModuleLoadData",
                cuModuleLoadData(&mut module, ptx.as_ptr().cast()),
            ) {
                let _ = cuStreamDestroy_v2(stream);
                let _ = cuDevicePrimaryCtxRelease(device);
                return Err(error);
            }
            if let Err(error) = check_cuda(
                "cuModuleGetFunction",
                cuModuleGetFunction(&mut write_paged_rows, module, kernel_name.as_ptr()),
            ) {
                let _ = cuModuleUnload(module);
                let _ = cuStreamDestroy_v2(stream);
                let _ = cuDevicePrimaryCtxRelease(device);
                return Err(error);
            }
        }

        Ok(Self {
            device,
            context,
            stream,
            module,
            write_paged_rows,
        })
    }

    fn synchronize(&self) -> Result<(), CudaBackendError> {
        unsafe { check_cuda("cuStreamSynchronize", cuStreamSynchronize(self.stream)) }
    }

    fn launch_paged_write(
        &self,
        page_table: &CudaPageTable,
        start_position: usize,
        rows: usize,
    ) -> Result<(), CudaBackendError> {
        let mut table = page_table.device_pointer();
        let mut start = u32::try_from(start_position)
            .map_err(|_| CudaBackendError::InvalidGeometry("start position exceeds u32"))?;
        let mut rows = u32::try_from(rows)
            .map_err(|_| CudaBackendError::InvalidGeometry("row count exceeds u32"))?;
        let mut page_tokens = PAGE_TOKENS as u32;
        let mut parameters = [
            (&mut table as *mut CuDevicePtr).cast::<c_void>(),
            (&mut start as *mut u32).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut page_tokens as *mut u32).cast::<c_void>(),
        ];

        unsafe {
            check_cuda(
                "cuLaunchKernel(write_paged_rows)",
                cuLaunchKernel(
                    self.write_paged_rows,
                    rows.div_ceil(THREADS),
                    1,
                    1,
                    THREADS,
                    1,
                    1,
                    0,
                    self.stream,
                    parameters.as_mut_ptr(),
                    null_mut(),
                ),
            )
        }
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = cuCtxSetCurrent(self.context);
            let _ = cuModuleUnload(self.module);
            let _ = cuStreamDestroy_v2(self.stream);
            let _ = cuDevicePrimaryCtxRelease(self.device);
        }
    }
}

struct CudaPageTable {
    device_tables: [DeviceAllocation; 2],
    published_addresses: Vec<CuDevicePtr>,
    staging_addresses: Vec<CuDevicePtr>,
    active: usize,
    position: usize,
}

impl CudaPageTable {
    fn new(max_position: usize) -> Result<Self, CudaBackendError> {
        let page_capacity = max_position.div_ceil(PAGE_TOKENS);
        let table_bytes = page_capacity * size_of::<CuDevicePtr>();
        Ok(Self {
            device_tables: [
                DeviceAllocation::new(table_bytes)?,
                DeviceAllocation::new(table_bytes)?,
            ],
            published_addresses: Vec::with_capacity(page_capacity),
            staging_addresses: Vec::with_capacity(page_capacity),
            active: 0,
            position: 0,
        })
    }

    fn managed_bytes(&self) -> usize {
        let capacity = self.published_addresses.capacity();
        4 * capacity * size_of::<CuDevicePtr>()
    }

    fn device_pointer(&self) -> CuDevicePtr {
        self.device_tables[self.active].pointer
    }

    fn publish(
        &mut self,
        addresses: &[CuDevicePtr],
        position: usize,
        device: &CudaDevice,
    ) -> Result<(), CudaBackendError> {
        if addresses.len() > self.staging_addresses.capacity() {
            return Err(CudaBackendError::InvalidGeometry(
                "logical page table exceeds its admitted capacity",
            ));
        }

        self.staging_addresses.clear();
        self.staging_addresses.extend_from_slice(addresses);
        let inactive = 1 - self.active;
        let bytes = size_of_val(self.staging_addresses.as_slice());
        // The table is double-buffered. Wait before reusing the inactive copy,
        // then publish from ordinary host memory with a synchronous transfer.
        // Model execution remains asynchronous on the CUDA stream.
        device.synchronize()?;
        if bytes != 0 {
            unsafe {
                check_cuda(
                    "cuMemcpyHtoD(page table)",
                    cuMemcpyHtoD_v2(
                        self.device_tables[inactive].pointer,
                        self.staging_addresses.as_ptr().cast(),
                        bytes,
                    ),
                )?;
            }
        }

        std::mem::swap(&mut self.published_addresses, &mut self.staging_addresses);
        self.active = inactive;
        self.position = position;
        Ok(())
    }

    fn read_rows(&self, rows: usize, device: &CudaDevice) -> Result<Vec<f32>, CudaBackendError> {
        if rows > self.position {
            return Err(CudaBackendError::InvalidGeometry(
                "cannot read beyond the committed position",
            ));
        }
        let mut values = vec![0.0; rows];
        device.synchronize()?;
        for (page_index, destination) in values.chunks_mut(PAGE_TOKENS).enumerate() {
            let source = *self
                .published_addresses
                .get(page_index)
                .ok_or(CudaBackendError::InvalidGeometry("page table is too short"))?;
            unsafe {
                check_cuda(
                    "cuMemcpyDtoH(page rows)",
                    cuMemcpyDtoH_v2(
                        destination.as_mut_ptr().cast(),
                        source,
                        size_of_val(destination),
                    ),
                )?;
            }
        }
        Ok(values)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaPage {
    slot: usize,
}

struct CudaExecution<'a> {
    device: &'a CudaDevice,
    page_table: &'a mut CudaPageTable,
}

struct CudaPageBackend {
    allocations: Vec<DeviceAllocation>,
    live: Vec<bool>,
    free_slots: Vec<usize>,
    max_pages: usize,
}

impl CudaPageBackend {
    fn new(max_pages: usize) -> Self {
        Self {
            allocations: Vec::new(),
            live: Vec::new(),
            free_slots: Vec::new(),
            max_pages,
        }
    }

    fn is_live(&self, page: CudaPage) -> bool {
        self.live.get(page.slot).copied().unwrap_or(false)
    }

    fn pointer(&self, page: CudaPage) -> Result<CuDevicePtr, CudaBackendError> {
        if !self.is_live(page) {
            return Err(CudaBackendError::StalePage(page.slot));
        }
        Ok(self.allocations[page.slot].pointer)
    }

    fn validate_pages<'a>(
        &self,
        pages: impl IntoIterator<Item = &'a CudaPage>,
    ) -> Result<(), CudaBackendError> {
        for page in pages {
            self.pointer(*page)?;
        }
        Ok(())
    }

    fn allocate(&mut self) -> Result<PageAllocation<CudaPage>, CudaBackendError> {
        if let Some(slot) = self.free_slots.pop() {
            self.live[slot] = true;
            return Ok(PageAllocation {
                page: CudaPage { slot },
                recycled: true,
            });
        }
        if self.allocations.len() == self.max_pages {
            return Err(CudaBackendError::OutOfPages);
        }
        let slot = self.allocations.len();
        self.allocations
            .push(DeviceAllocation::new(PAGE_TOKENS * size_of::<f32>())?);
        self.live.push(true);
        Ok(PageAllocation {
            page: CudaPage { slot },
            recycled: false,
        })
    }

    fn release(&mut self, page: CudaPage) -> Result<(), CudaBackendError> {
        if !self.is_live(page) {
            return Err(CudaBackendError::StalePage(page.slot));
        }
        self.live[page.slot] = false;
        self.free_slots.push(page.slot);
        Ok(())
    }

    fn addresses(&self, pages: &[&CudaPage]) -> Result<Vec<CuDevicePtr>, CudaBackendError> {
        pages.iter().map(|page| self.pointer(**page)).collect()
    }
}

impl PageBackend for CudaPageBackend {
    type Page = CudaPage;
    type Context<'a> = CudaExecution<'a>;
    // The kernel writes only previously invalid rows. Hiding an aborted suffix
    // is sufficient because a later append overwrites those rows.
    type AppendTransaction = ();
    type Error = CudaBackendError;

    fn page_bytes(&self) -> usize {
        PAGE_TOKENS * size_of::<f32>()
    }

    fn page_capacity(&self) -> Option<usize> {
        Some(self.max_pages)
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.allocate()
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.release(page)
            .expect("an unpublished CUDA page must remain live");
    }

    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        _start_position: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction, Self::Error> {
        self.validate_pages(pages.iter().map(BackendAppendPage::page))?;
        if pages
            .iter()
            .any(|page| page.page_offset() + page.rows() > PAGE_TOKENS)
        {
            return Err(CudaBackendError::InvalidGeometry(
                "append segment exceeds a physical CUDA page",
            ));
        }
        Ok(())
    }

    fn abort_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(restored_pages.iter().copied())?;
        self.validate_pages(released_pages.iter().copied())?;
        context.device.synchronize()?;
        let addresses = self.addresses(restored_pages)?;
        context
            .page_table
            .publish(&addresses, restored_position, context.device)?;
        for page in released_pages {
            self.release(**page)
                .expect("validated CUDA reservation page remains live");
        }
        Ok(())
    }

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        if valid_tokens > PAGE_TOKENS {
            return Err(CudaBackendError::InvalidGeometry(
                "copy exceeds a physical CUDA page",
            ));
        }
        context.device.synchronize()?;
        let source_pointer = self.pointer(*source)?;
        let allocation = self.allocate()?;
        let destination_pointer = self.pointer(allocation.page)?;
        let copy = unsafe {
            check_cuda(
                "cuMemcpyDtoD(partial page)",
                cuMemcpyDtoD_v2(
                    destination_pointer,
                    source_pointer,
                    valid_tokens * size_of::<f32>(),
                ),
            )
        };
        if let Err(error) = copy {
            self.release(allocation.page)
                .expect("failed copy destination must remain live");
            return Err(error);
        }
        Ok(allocation)
    }

    fn commit_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(commit.committed_pages().iter().copied())?;
        self.validate_pages(commit.released_pages().iter().copied())?;
        context.device.synchronize()?;
        let addresses = self.addresses(commit.committed_pages())?;
        context
            .page_table
            .publish(&addresses, commit.position(), context.device)?;
        for page in commit.released_pages() {
            self.release(**page)
                .expect("validated CUDA reservation page remains live");
        }
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(pages.iter().copied())?;
        let addresses = self.addresses(pages)?;
        context
            .page_table
            .publish(&addresses, position, context.device)
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        if let Some(error) = pages.iter().find_map(|page| self.pointer(*page).err()) {
            return Err(RetireError { error, pages });
        }
        if let Err(error) = context.device.synchronize() {
            return Err(RetireError { error, pages });
        }
        for page in &pages {
            self.release(*page)
                .expect("validated CUDA retirement page remains live");
        }
        Ok(RetireOutcome::default())
    }

    fn retirement_is_immediate(&self) -> bool {
        true
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize, Self::Error> {
        Ok(0)
    }
}

type CudaSequenceCache = SequenceCache<CudaPageBackend, ()>;

fn admitted(outcome: AdmissionOutcome) -> Result<SequenceId, Box<dyn Error>> {
    match outcome {
        AdmissionOutcome::Admitted(sequence) => Ok(sequence),
        AdmissionOutcome::WouldBlock => Err("cache admission unexpectedly blocked".into()),
    }
}

fn append(
    cache: &mut CudaSequenceCache,
    sequence: SequenceId,
    rows: usize,
    context: &mut CudaExecution<'_>,
) -> Result<usize, Box<dyn Error>> {
    let reservation = cache.reserve_append(sequence, rows, context)?;
    let start_position = reservation.start_position();
    let segment_count = reservation.segments().len();
    cache.with_append_pages(&reservation, |_backend, pages| {
        debug_assert_eq!(
            pages
                .iter()
                .map(|page| page.segment().rows())
                .sum::<usize>(),
            rows
        );
        context
            .device
            .launch_paged_write(context.page_table, start_position, rows)
    })?;
    cache.commit_append(reservation, rows, context)?;
    Ok(segment_count)
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = CudaDevice::new()?;
    let mut page_table = CudaPageTable::new(MAX_SEQUENCE_TOKENS)?;
    let backend = CudaPageBackend::new(MAX_PHYSICAL_PAGES);
    let page_bytes = backend.page_bytes();
    let config = CacheConfig {
        page_tokens: PAGE_TOKENS,
        max_managed_bytes: page_bytes * MAX_PHYSICAL_PAGES + page_table.managed_bytes(),
        max_snapshot_bytes: 0,
        max_prefix_entries: None,
        emergency_bytes: 0,
    };
    let mut cache = SequenceCache::new(config, backend)?;
    let request = AdmissionRequest {
        max_position: MAX_SEQUENCE_TOKENS,
        private_state_bytes: 0,
        page_table_bytes: page_table.managed_bytes(),
        allow_emergency: false,
    };
    let mut context = CudaExecution {
        device: &device,
        page_table: &mut page_table,
    };
    let sequence = admitted(
        cache.admit(None, request, &mut context, |snapshot, position| {
            assert!(snapshot.is_none());
            assert_eq!(position, 0);
            Ok(())
        })?,
    )?;

    // Leave a partial tail, then cross two page boundaries in one launch.
    assert_eq!(append(&mut cache, sequence, 2, &mut context)?, 1);
    let segments = append(&mut cache, sequence, 9, &mut context)?;
    assert_eq!(segments, 3);
    assert_eq!(context.page_table.position, 11);
    assert_eq!(
        context.page_table.read_rows(11, context.device)?,
        (1..=11).map(|value| value as f32).collect::<Vec<_>>()
    );
    cache.validate()?;

    println!(
        "wrote 9 rows through {segments} CUDA pages in one kernel launch; committed position: {}",
        context.page_table.position
    );
    cache.finish(sequence, &mut context)?;
    Ok(())
}
