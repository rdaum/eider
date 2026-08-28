//! Narrow tensor-facing wrappers over the current device matrix owners.
//!
//! These types are intentionally 2D, device-owned, and layout-explicit. They
//! do not try to become a general tensor abstraction; they name the storage
//! boundary the inference API should expose while still mapping directly to the
//! cuBLASLt layouts used by the FP4 path.
//!
//! The first inference-facing convention is column-major 2D storage. For the
//! current FP4 TN operation, weights are represented as `K x M` or `K x N`
//! tensors so cuBLASLt can launch `D[M,N] = A[K,M]^T * B[K,N]` without an
//! intermediate dequantized weight matrix.

use crate::error::Result;
use crate::matrix::{Bf16Matrix, Nvfp4Matrix};

/// Explicit storage layout for a 2D tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tensor2dLayout {
    /// Column-major storage with `ld == rows`.
    ColumnMajor,
}

/// Borrowed metadata for a device-resident 2D tensor.
///
/// Views intentionally expose only shape and layout metadata today. They are a
/// stable way for higher layers to describe tensors and validate operation
/// contracts without gaining direct access to raw device pointers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor2dView {
    rows: usize,
    cols: usize,
    leading_dim: usize,
    layout: Tensor2dLayout,
}

impl Tensor2dView {
    /// Creates a 2D tensor view from explicit metadata.
    pub const fn new(rows: usize, cols: usize, leading_dim: usize, layout: Tensor2dLayout) -> Self {
        Self {
            rows,
            cols,
            leading_dim,
            layout,
        }
    }

    /// Returns the logical row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the logical column count.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the leading dimension in elements.
    pub const fn leading_dim(&self) -> usize {
        self.leading_dim
    }

    /// Returns the explicit tensor layout.
    pub const fn layout(&self) -> Tensor2dLayout {
        self.layout
    }

    /// Returns the logical element count.
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Returns true when the logical tensor has no elements.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Device-owned 2D NVFP4 tensor.
///
/// Values are packed E2M1 nibbles and scales are UE4M3 `VEC16` scale bytes in
/// the cuBLASLt-compatible layout produced by [`crate::format`].
pub struct Nvfp4Tensor2d {
    matrix: Nvfp4Matrix,
    layout: Tensor2dLayout,
}

impl Nvfp4Tensor2d {
    /// Quantizes a column-major f32 tensor into device-resident NVFP4 storage.
    pub fn quantize_col_major_f32(rows: usize, cols: usize, values: &[f32]) -> Result<Self> {
        Ok(Self {
            matrix: Nvfp4Matrix::quantize_col_major_f32(rows, cols, values)?,
            layout: Tensor2dLayout::ColumnMajor,
        })
    }

    /// Creates a column-major tensor filled with `1.0`.
    pub fn ones_col_major(rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            matrix: Nvfp4Matrix::ones_col_major(rows, cols)?,
            layout: Tensor2dLayout::ColumnMajor,
        })
    }

    /// Returns the logical row count.
    pub fn rows(&self) -> usize {
        self.matrix.rows
    }

    /// Returns the logical column count.
    pub fn cols(&self) -> usize {
        self.matrix.cols
    }

    /// Returns the leading dimension in elements.
    pub fn leading_dim(&self) -> usize {
        self.matrix.ld
    }

    /// Returns the explicit tensor layout.
    pub fn layout(&self) -> Tensor2dLayout {
        self.layout
    }

    /// Returns borrowed shape and layout metadata.
    pub fn view(&self) -> Tensor2dView {
        Tensor2dView::new(self.rows(), self.cols(), self.leading_dim(), self.layout())
    }

    /// Returns packed E2M1 payload bytes.
    pub fn value_bytes(&self) -> usize {
        self.matrix.value_bytes()
    }

    /// Returns UE4M3 scale metadata bytes.
    pub fn scale_bytes(&self) -> usize {
        self.matrix.scale_bytes()
    }

    /// Returns total device bytes owned by this tensor.
    pub fn device_bytes(&self) -> usize {
        self.value_bytes() + self.scale_bytes()
    }

    /// Returns the underlying cuBLASLt-compatible matrix owner.
    pub fn as_matrix(&self) -> &Nvfp4Matrix {
        &self.matrix
    }

    /// Consumes the tensor and returns the underlying matrix owner.
    pub fn into_matrix(self) -> Nvfp4Matrix {
        self.matrix
    }
}

/// Device-owned 2D BF16 tensor.
///
/// This is the current output/accumulator storage for the cuBLASLt FP4 path.
pub struct Bf16Tensor2d {
    matrix: Bf16Matrix,
    layout: Tensor2dLayout,
}

impl Bf16Tensor2d {
    /// Allocates a zero-initialized column-major BF16 tensor.
    pub fn zeroed_col_major(rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            matrix: Bf16Matrix::zeroed(rows, cols)?,
            layout: Tensor2dLayout::ColumnMajor,
        })
    }

    /// Returns the logical row count.
    pub fn rows(&self) -> usize {
        self.matrix.rows
    }

    /// Returns the logical column count.
    pub fn cols(&self) -> usize {
        self.matrix.cols
    }

    /// Returns the leading dimension in elements.
    pub fn leading_dim(&self) -> usize {
        self.matrix.ld
    }

    /// Returns the explicit tensor layout.
    pub fn layout(&self) -> Tensor2dLayout {
        self.layout
    }

    /// Returns borrowed shape and layout metadata.
    pub fn view(&self) -> Tensor2dView {
        Tensor2dView::new(self.rows(), self.cols(), self.leading_dim(), self.layout())
    }

    /// Returns total device bytes owned by this tensor.
    pub fn device_bytes(&self) -> usize {
        self.matrix.device_bytes()
    }

    /// Returns the underlying cuBLASLt-compatible matrix owner.
    pub fn as_matrix(&self) -> &Bf16Matrix {
        &self.matrix
    }

    /// Consumes the tensor and returns the underlying matrix owner.
    pub fn into_matrix(self) -> Bf16Matrix {
        self.matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_view_reports_explicit_metadata() {
        let view = Tensor2dView::new(7, 11, 16, Tensor2dLayout::ColumnMajor);
        assert_eq!(view.rows(), 7);
        assert_eq!(view.cols(), 11);
        assert_eq!(view.leading_dim(), 16);
        assert_eq!(view.layout(), Tensor2dLayout::ColumnMajor);
        assert_eq!(view.len(), 77);
        assert!(!view.is_empty());
    }
}
