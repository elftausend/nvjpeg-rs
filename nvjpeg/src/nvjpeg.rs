use std::ptr::null_mut;

use nvjpeg_sys::{nvjpegHandle_t, nvjpegCreateSimple};

use crate::error::{NVJpegError, ToNVJpegResult};

#[derive(Debug)]
pub struct JpegHandle {
    inner: nvjpegHandle_t
}

impl JpegHandle {
    #[inline]
    pub fn new_simple() -> Result<Self, NVJpegError> {
        let mut inner = null_mut();
        unsafe {
            nvjpegCreateSimple(&mut inner).to_result()?;
        }

        Ok(JpegHandle { inner })
    }
}

impl Default for JpegHandle {
    #[inline]
    fn default() -> Self {
        Self::new_simple().expect("Could not create default simple JpegHandle")
    }
}

impl Drop for JpegHandle {
    fn drop(&mut self) {
        unsafe {
            nvjpeg_sys::nvjpegDestroy(self.inner);
        }
    }
}