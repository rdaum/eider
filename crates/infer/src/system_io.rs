//! Reviewed host system-call adapters used by inference loading paths.

use std::fs::File;
use std::io::IoSliceMut;
use std::os::fd::AsRawFd;

/// Reads every destination from `file` at `offset`, retrying interrupted
/// vectored reads and rejecting incomplete records.
pub(crate) fn read_exact_vectored_at(
    file: &File,
    mut destinations: &mut [IoSliceMut<'_>],
    mut offset: u64,
) -> std::io::Result<()> {
    const MAX_DESTINATIONS: usize = 6;
    while !destinations.is_empty() {
        if destinations.len() > MAX_DESTINATIONS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "too many vectored read destinations",
            ));
        }
        let mut vectors: [libc::iovec; MAX_DESTINATIONS] = std::array::from_fn(|_| libc::iovec {
            iov_base: std::ptr::null_mut(),
            iov_len: 0,
        });
        for (vector, destination) in vectors.iter_mut().zip(destinations.iter_mut()) {
            vector.iov_base = destination.as_mut_ptr().cast();
            vector.iov_len = destination.len();
        }
        // `vectors` contains only live mutable destination slices, and all
        // destinations stay borrowed until `preadv` returns.
        let bytes = unsafe {
            libc::preadv(
                file.as_raw_fd(),
                vectors.as_ptr(),
                destinations.len() as i32,
                offset as libc::off_t,
            )
        };
        if bytes == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        if bytes < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        let bytes = bytes as usize;
        offset = offset.saturating_add(bytes as u64);
        IoSliceMut::advance_slices(&mut destinations, bytes);
    }
    Ok(())
}
