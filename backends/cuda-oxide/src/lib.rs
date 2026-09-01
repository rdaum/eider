//! cuda-oxide implementation of Eider's SM121 device kernels.
//!
//! The crate produces PTX only. The `eider-cuda` build script assembles the PTX
//! into a CUBIN and embeds it in the stable Rust host crate.

#![deny(unsafe_op_in_unsafe_fn)]

mod common;
mod kernels {
    //! Device kernels grouped by runtime responsibility.

    mod dflash2;
    mod elementwise;
    mod flash_next;
    mod gdn;
    mod gdn_chunk;
    mod kv_cache;
    mod linear;
    mod nvfp4;
    mod position;
}
