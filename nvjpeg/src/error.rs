use nvjpeg_sys::nvjpegStatus_t;

#[derive(Debug)]
#[repr(C)]
pub enum NVJpegError {
    NotInitialized = 1,
    InvalidParameter = 2,
    BadJpeg = 3,
    JpegNotSupported = 4,
    AllocatorFailure = 5,
    ExecutionFailed = 6,
    ArchMismatch = 7,
    InternalError = 8,
    ImplementationNotSupported = 9,
    IncompleteBitstream = 10,
}

impl NVJpegError {
    pub fn as_str(&self) -> &'static str {
        // uses error text messages from: https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpeg-type-declarations
        match self {
            NVJpegError::NotInitialized => "The library handle was not initialized. A call to nvjpegCreate() is required to initialize the handle.",
            NVJpegError::InvalidParameter => "Wrong parameter was passed. For example, a null pointer as input data, or an image index not in the allowed range.",
            NVJpegError::BadJpeg => "Cannot parse the JPEG stream. Check that the encoded JPEG stream and its size parameters are correct.",
            NVJpegError::JpegNotSupported => "Attempting to decode a JPEG stream that is not supported by the nvJPEG library.",
            NVJpegError::AllocatorFailure => "The user-provided allocator functions, for either memory allocation or for releasing the memory, returned a non-zero code.",
            NVJpegError::ExecutionFailed => "Error during the execution of the device tasks.",
            NVJpegError::ArchMismatch => "The device capabilities are not enough for the set of input parameters provided (input parameters such as backend, encoded stream parameters, output format).",
            NVJpegError::InternalError => "Error during the execution of the device tasks.",
            NVJpegError::ImplementationNotSupported => "Not supported.",
            NVJpegError::IncompleteBitstream => "Bitstream input data incomplete.",
        }
    }
}

impl From<usize> for NVJpegError {
    fn from(value: usize) -> Self {
        match value {
            1 => NVJpegError::NotInitialized,
            2 => NVJpegError::InvalidParameter,
            3 => NVJpegError::BadJpeg,
            4 => NVJpegError::JpegNotSupported,
            5 => NVJpegError::AllocatorFailure,
            6 => NVJpegError::ExecutionFailed,
            7 => NVJpegError::ArchMismatch,
            8 => NVJpegError::InternalError,
            9 => NVJpegError::ImplementationNotSupported,
            10 => NVJpegError::IncompleteBitstream,
            _ => panic!("Unknown NVJpegError value: {}", value),
        }
    }
}

pub trait ToNVJpegResult {
    fn to_result(self) -> Result<(), NVJpegError>;
}

impl ToNVJpegResult for nvjpegStatus_t {
    #[inline]
    fn to_result(self) -> Result<(), NVJpegError> {
        match self {
            0 => Ok(()),
            _ => Err(NVJpegError::from(self as usize)),
        }
    }
}

impl std::fmt::Display for NVJpegError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for NVJpegError {}
