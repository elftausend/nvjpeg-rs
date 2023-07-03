use std::ptr::null_mut;

use custos::{CUDA, cuda::api::Stream};
use nvjpeg_sys::{
    nvjpegChromaSubsampling_t, nvjpegCreateSimple, nvjpegDestroy, nvjpegGetImageInfo,
    nvjpegHandle_t, nvjpegJpegStateCreate, nvjpegJpegStateDestroy, nvjpegJpegState_t,
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw_data = std::fs::read("cat_798x532.jpg")?;
    //let raw_data = std::fs::read("cat.jpg")?;

    let device = CUDA::new(0)?;

    let image = unsafe { decode_raw_jpeg(&raw_data, device.stream())? };


    Ok(())
}

#[derive(Debug, Default)]
pub struct Image {
}

unsafe fn decode_raw_jpeg(raw_data: &[u8], stream: &Stream) -> Result<Image, String> {
    let mut handle: nvjpegHandle_t = null_mut();

    let status = nvjpegCreateSimple(&mut handle);
    if status != 0 {
        return Err(format!(
            "Could not create simple hanlde. Error occured with code: {status}"
        ));
    }

    let mut jpeg_state: nvjpegJpegState_t = null_mut();
    let status = nvjpegJpegStateCreate(handle, &mut jpeg_state);

    if status != 0 {
        return Err(format!(
            "Could not create jpeg state. Error occured with code: {status}"
        ));
    }

    if status != 0 {
        return Err(format!("Error occured with code: {status}"));
    }

    let mut n_components = 0;
    let mut subsampling: nvjpegChromaSubsampling_t = 0;
    let mut widths = [0, 0, 0];
    let mut heights = [0, 0, 0];

    let status = nvjpegGetImageInfo(
        handle,
        raw_data.as_ptr(),
        raw_data.len(),
        &mut n_components,
        &mut subsampling,
        widths.as_mut_ptr(),
        heights.as_mut_ptr(),
    );

    if status != 0 {
        return Err(format!(
            "Could not get image info. Error occured with code: {status}"
        ));
    }

    println!("n_components: {n_components}, subsampling: {subsampling}, widths: {widths:?}, heights: {heights:?}");

    // free

    let status = nvjpegJpegStateDestroy(jpeg_state);

    if status != 0 {
        return Err(format!(
            "Could not free jpeg state. Error occured with code: {status}"
        ));
    }

    let status = nvjpegDestroy(handle);

    if status != 0 {
        return Err(format!(
            "Could not free nvjpeg handle. Error occured with code: {status}"
        ));
    }

    Ok(Image::default())
}
