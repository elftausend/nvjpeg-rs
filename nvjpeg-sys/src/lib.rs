//! Raw FFI Rust bindings to nvJPEG.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod bindings;

pub use bindings::*;

#[macro_export]
macro_rules! check {
    ($status:ident, $err:literal) => {
        if $status != 0 {
            Err(format!("{}. Error occured with code: {}", $err, $status))?
        }
    };
}
