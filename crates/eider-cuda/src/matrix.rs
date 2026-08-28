//! Matrix owner types for the current cuBLASLt FP4 path.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceInput, DeviceOutput, HostRead};
use crate::error::{Error, Result};
use crate::format;

/// Logical matrix dimensions and leading dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixShape {
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

impl MatrixShape {
    /// Creates a column-major matrix shape with `ld = rows`.
    pub const fn column_major(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            ld: rows,
        }
    }

    /// Returns the logical element count.
    pub fn len(self) -> usize {
        self.rows * self.cols
    }

    /// Returns true when this shape has no logical elements.
    pub fn is_empty(self) -> bool {
        self.rows == 0 || self.cols == 0
    }
}

/// Device-resident packed NVFP4 matrix for cuBLASLt FP4 GEMM.
///
/// Values are stored as packed E2M1 nibbles. Scales are stored in cuBLASLt's
/// tiled UE4M3 `VEC16` layout.
pub struct Nvfp4Matrix {
    pub(crate) values: DeviceBuffer<u8>,
    pub(crate) scales: DeviceBuffer<u8>,
    /// Logical row count in the cuBLASLt matrix layout.
    pub rows: usize,
    /// Logical column count in the cuBLASLt matrix layout.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

/// Input-role view of an NVFP4 matrix.
pub struct Nvfp4MatrixInput<'a> {
    values: DeviceInput<'a, u8>,
    scales: DeviceInput<'a, u8>,
    /// Logical row count in the cuBLASLt matrix layout.
    pub rows: usize,
    /// Logical column count in the cuBLASLt matrix layout.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

impl Nvfp4MatrixInput<'_> {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns the packed E2M1 payload pointer.
    pub(crate) fn values_ptr(&self) -> *const u8 {
        self.values.as_const_ptr().cast()
    }

    /// Returns the UE4M3 scale metadata pointer.
    pub(crate) fn scales_ptr(&self) -> *const u8 {
        self.scales.as_const_ptr().cast()
    }
}

/// Output-role view of an NVFP4 matrix.
pub struct Nvfp4MatrixOutput<'a> {
    values: DeviceOutput<'a, u8>,
    scales: DeviceOutput<'a, u8>,
    /// Logical row count in the cuBLASLt matrix layout.
    pub rows: usize,
    /// Logical column count in the cuBLASLt matrix layout.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

impl Nvfp4MatrixOutput<'_> {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns the packed E2M1 payload output pointer.
    pub(crate) fn values_mut_ptr(&mut self) -> *mut u8 {
        self.values.as_mut_ptr().cast()
    }

    /// Returns the UE4M3 scale metadata output pointer.
    pub(crate) fn scales_mut_ptr(&mut self) -> *mut u8 {
        self.scales.as_mut_ptr().cast()
    }
}

impl Nvfp4Matrix {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Allocates a zero-initialized column-major NVFP4 matrix.
    ///
    /// This is primarily useful for graph-captured inference paths that need
    /// stable activation operand storage before runtime quantization fills it.
    pub fn zeroed_col_major(rows: usize, cols: usize) -> Result<Self> {
        Self::from_device_col_major_parts(
            rows,
            cols,
            DeviceBuffer::zeroed((rows * cols).div_ceil(2))?,
            DeviceBuffer::zeroed(format::ue4m3_scale_layout_len(cols, rows))?,
        )
    }

    pub(crate) fn from_device_col_major_parts(
        rows: usize,
        cols: usize,
        values: DeviceBuffer<u8>,
        scales: DeviceBuffer<u8>,
    ) -> Result<Self> {
        let expected_values = (rows * cols).div_ceil(2);
        if values.len() != expected_values {
            return Err(Error::Shape {
                label: "NVFP4 device packed values",
                expected: format!("{expected_values} bytes"),
                actual: format!("{} bytes", values.len()),
            });
        }

        let expected_scales = format::ue4m3_scale_layout_len(cols, rows);
        if scales.len() != expected_scales {
            return Err(Error::Shape {
                label: "NVFP4 device scales",
                expected: format!("{expected_scales} bytes"),
                actual: format!("{} bytes", scales.len()),
            });
        }

        Ok(Self {
            values,
            scales,
            rows,
            cols,
            ld: rows,
        })
    }

    /// Creates a matrix from packed E2M1 values and cuBLASLt-layout UE4M3
    /// scales.
    ///
    /// This is the import path for external checkpoints that already store FP4
    /// values. `packed_values` must contain two logical matrix elements per
    /// byte in column-major `rows x cols` order. `scales` must already be in
    /// cuBLASLt's tiled `VEC16_UE4M3` scale layout for `outer_dim=cols` and
    /// `inner_dim=rows`.
    pub fn from_packed_col_major_parts(
        rows: usize,
        cols: usize,
        packed_values: &[u8],
        scales: &[u8],
    ) -> Result<Self> {
        let expected_values = (rows * cols).div_ceil(2);
        if packed_values.len() != expected_values {
            return Err(Error::Shape {
                label: "NVFP4 packed values",
                expected: format!("{expected_values} bytes"),
                actual: format!("{} bytes", packed_values.len()),
            });
        }

        let expected_scales = format::ue4m3_scale_layout_len(cols, rows);
        if scales.len() != expected_scales {
            return Err(Error::Shape {
                label: "NVFP4 scales",
                expected: format!("{expected_scales} bytes"),
                actual: format!("{} bytes", scales.len()),
            });
        }

        Ok(Self {
            values: DeviceBuffer::from_host(packed_values)?,
            scales: DeviceBuffer::from_host(scales)?,
            rows,
            cols,
            ld: rows,
        })
    }

    /// Quantizes a column-major f32 matrix into E2M1 values and UE4M3 scales.
    ///
    /// The scale layout is built as `outer_dim=cols`, `inner_dim=rows`, matching
    /// how the current TN matmul stores A as `K x M` and B as `K x N`.
    pub fn quantize_col_major_f32(rows: usize, cols: usize, values: &[f32]) -> Result<Self> {
        assert_eq!(values.len(), rows * cols);
        let quantized = format::quantize_nvfp4_col_major(rows, cols, values);
        Ok(Self {
            values: DeviceBuffer::from_host(&quantized.packed_values)?,
            scales: DeviceBuffer::from_host(&quantized.scales)?,
            rows,
            cols,
            ld: rows,
        })
    }

    /// Packs column-major values as E2M1 with unit UE4M3 scales.
    ///
    /// This constructor assumes `values` are already representable after
    /// scaling by `1.0`; it is retained for exact smoke fixtures. General inputs
    /// should use [`Nvfp4Matrix::quantize_col_major_f32`].
    pub fn from_e2m1_values_with_unit_scales(
        rows: usize,
        cols: usize,
        values: &[f32],
    ) -> Result<Self> {
        assert_eq!(values.len(), rows * cols);
        let packed = format::pack_e2m1(values);
        let scales = format::ue4m3_ones_scale_layout(cols, rows);
        Ok(Self {
            values: DeviceBuffer::from_host(&packed)?,
            scales: DeviceBuffer::from_host(&scales)?,
            rows,
            cols,
            ld: rows,
        })
    }

    /// Creates a column-major matrix filled with `1.0` for smoke tests and
    /// benchmarks.
    pub fn ones_col_major(rows: usize, cols: usize) -> Result<Self> {
        Self::from_e2m1_values_with_unit_scales(rows, cols, &vec![1.0; rows * cols])
    }

    /// Returns packed E2M1 payload bytes.
    pub fn value_bytes(&self) -> usize {
        (self.rows * self.cols).div_ceil(2)
    }

    /// Returns the packed E2M1 payload pointer.
    pub fn values_ptr(&self) -> *const u8 {
        self.values.as_const_ptr().cast()
    }

    /// Returns the UE4M3 scale metadata pointer.
    pub fn scales_ptr(&self) -> *const u8 {
        self.scales.as_const_ptr().cast()
    }

    /// Synchronizes `stream` and copies the packed E2M1 payload to host memory.
    pub fn copy_values_to_host<'a>(&'a self, stream: &CudaStream) -> Result<HostRead<'a, u8>> {
        self.values.copy_to_host(stream)
    }

    /// Synchronizes `stream` and copies the UE4M3 scale metadata to host memory.
    pub fn copy_scales_to_host<'a>(&'a self, stream: &CudaStream) -> Result<HostRead<'a, u8>> {
        self.scales.copy_to_host(stream)
    }

    /// Returns the packed E2M1 payload output pointer.
    pub fn values_mut_ptr(&mut self) -> *mut u8 {
        self.values.as_mut_ptr().cast()
    }

    /// Returns the UE4M3 scale metadata output pointer.
    pub fn scales_mut_ptr(&mut self) -> *mut u8 {
        self.scales.as_mut_ptr().cast()
    }

    /// Borrows this matrix as a kernel input role.
    pub fn input(&self) -> Nvfp4MatrixInput<'_> {
        Nvfp4MatrixInput {
            values: self.values.input(),
            scales: self.scales.input(),
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Borrows this matrix as a kernel output role.
    pub fn output(&mut self) -> Nvfp4MatrixOutput<'_> {
        Nvfp4MatrixOutput {
            values: self.values.output(),
            scales: self.scales.output(),
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns UE4M3 scale metadata bytes.
    pub fn scale_bytes(&self) -> usize {
        format::ue4m3_scale_layout_len(self.cols, self.rows)
    }

    /// Returns total device bytes owned by this matrix.
    pub fn device_bytes(&self) -> usize {
        self.value_bytes() + self.scale_bytes()
    }
}

/// Device-resident BF16 matrix used as cuBLASLt C/D storage.
pub struct Bf16Matrix {
    pub(crate) data: DeviceBuffer<u16>,
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

/// Input-role view of a BF16 matrix.
pub struct Bf16MatrixInput<'a> {
    data: DeviceInput<'a, u16>,
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

impl Bf16MatrixInput<'_> {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns the BF16 data pointer.
    pub(crate) fn data_ptr(&self) -> *const u16 {
        self.data.as_const_ptr().cast()
    }
}

/// Output-role view of a BF16 matrix.
pub struct Bf16MatrixOutput<'a> {
    data: DeviceOutput<'a, u16>,
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

impl Bf16MatrixOutput<'_> {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns the BF16 data output pointer.
    pub(crate) fn data_mut_ptr(&mut self) -> *mut u16 {
        self.data.as_mut_ptr().cast()
    }
}

impl Bf16Matrix {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Allocates a zero-initialized BF16 matrix.
    pub fn zeroed(rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            data: DeviceBuffer::zeroed(rows * cols)?,
            rows,
            cols,
            ld: rows,
        })
    }

    /// Creates a BF16 matrix from raw host BF16 values.
    pub fn from_bf16_host(rows: usize, cols: usize, values: &[u16]) -> Result<Self> {
        if values.len() != rows * cols {
            return Err(Error::Shape {
                label: "BF16 matrix host values",
                expected: format!("{} values", rows * cols),
                actual: format!("{} values", values.len()),
            });
        }
        Ok(Self {
            data: DeviceBuffer::from_host(values)?,
            rows,
            cols,
            ld: rows,
        })
    }

    /// Returns the underlying device buffer.
    pub fn data(&self) -> &DeviceBuffer<u16> {
        &self.data
    }

    /// Returns the BF16 data pointer.
    pub fn data_ptr(&self) -> *const u16 {
        self.data.as_const_ptr().cast()
    }

    /// Returns the BF16 data output pointer.
    pub fn data_mut_ptr(&mut self) -> *mut u16 {
        self.data.as_mut_ptr().cast()
    }

    /// Borrows this matrix as a kernel input role.
    pub fn input(&self) -> Bf16MatrixInput<'_> {
        Bf16MatrixInput {
            data: self.data.input(),
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Borrows this matrix as a kernel output role.
    pub fn output(&mut self) -> Bf16MatrixOutput<'_> {
        Bf16MatrixOutput {
            data: self.data.output(),
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns total device bytes owned by this matrix.
    pub fn device_bytes(&self) -> usize {
        self.rows * self.cols * std::mem::size_of::<u16>()
    }
}

/// Device-resident F32 matrix used as cuBLASLt C/D storage.
pub struct F32Matrix {
    pub(crate) data: DeviceBuffer<f32>,
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

/// Input-role view of an F32 matrix.
pub struct F32MatrixInput<'a> {
    data: DeviceInput<'a, f32>,
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

impl F32MatrixInput<'_> {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns the F32 data pointer.
    pub(crate) fn data_ptr(&self) -> *const f32 {
        self.data.as_const_ptr().cast()
    }
}

/// Output-role view of an F32 matrix.
pub struct F32MatrixOutput<'a> {
    data: DeviceOutput<'a, f32>,
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Leading dimension in elements.
    pub ld: usize,
}

impl F32MatrixOutput<'_> {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Returns the F32 data output pointer.
    pub(crate) fn data_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr().cast()
    }
}

impl F32Matrix {
    /// Returns the logical matrix shape.
    pub fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Allocates a zero-initialized F32 matrix.
    pub fn zeroed(rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            data: DeviceBuffer::zeroed(rows * cols)?,
            rows,
            cols,
            ld: rows,
        })
    }

    /// Returns the underlying device buffer.
    pub fn data(&self) -> &DeviceBuffer<f32> {
        &self.data
    }

    /// Returns the underlying device buffer mutably.
    pub fn data_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.data
    }

    /// Returns the F32 data pointer.
    pub fn data_ptr(&self) -> *const f32 {
        self.data.as_const_ptr().cast()
    }

    /// Returns the F32 data output pointer.
    pub fn data_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr().cast()
    }

    /// Borrows this matrix as a kernel input role.
    pub fn input(&self) -> F32MatrixInput<'_> {
        F32MatrixInput {
            data: self.data.input(),
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Borrows this matrix as a kernel output role.
    pub fn output(&mut self) -> F32MatrixOutput<'_> {
        F32MatrixOutput {
            data: self.data.output(),
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
        }
    }

    /// Consumes the matrix and returns the underlying device buffer.
    pub fn into_data(self) -> DeviceBuffer<f32> {
        self.data
    }

    /// Returns total device bytes owned by this matrix.
    pub fn device_bytes(&self) -> usize {
        self.rows * self.cols * std::mem::size_of::<f32>()
    }
}
