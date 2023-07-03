use std::ptr::null_mut;

use custos::{cuda::api::Stream, Buffer, CUDA};
use nvjpeg_sys::{
    nvjpegChromaSubsampling_t, nvjpegCreateSimple, nvjpegDecode, nvjpegDestroy, nvjpegGetImageInfo,
    nvjpegHandle_t, nvjpegImage_t, nvjpegJpegStateCreate, nvjpegJpegStateDestroy,
    nvjpegJpegState_t, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGBI,
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw_data = std::fs::read("cat_798x532.jpg")?;
    //let raw_data = std::fs::read("cat.jpg")?;

    let device = CUDA::new(0)?;

    let image = unsafe { decode_raw_jpeg(&raw_data, &device)? };

    Ok(())
}

#[derive(Debug, Default)]
pub struct Image {}

macro_rules! check {
    ($status:ident, $err:literal) => {
        if $status != 0 {
            return Err(format!("{}. Error occured with code: {}", $err, $status))
        }
    };
}

unsafe fn decode_raw_jpeg(raw_data: &[u8], device: &CUDA) -> Result<Image, String> {
    let mut handle: nvjpegHandle_t = null_mut();

    let status = nvjpegCreateSimple(&mut handle);
    check!(status, "Could not create simple handle. ");
    

    let mut jpeg_state: nvjpegJpegState_t = null_mut();
    let status = nvjpegJpegStateCreate(handle, &mut jpeg_state);
    check!(status, "Could not create jpeg state. ");

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
    check!(status, "Could not get image info. ");

    heights[0] = heights[1] * 2;

    println!("n_components: {n_components}, subsampling: {subsampling}, widths: {widths:?}, heights: {heights:?}");

    let mut image: nvjpegImage_t = nvjpegImage_t::new();
    
    image.pitch[0] = widths[0] as usize;
    image.pitch[1] = widths[0] as usize;
    image.pitch[2] = widths[0] as usize;

    let channel0 = Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);
    let channel1 = Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);
    let channel2 = Buffer::<u8, _>::new(device, image.pitch[0] * heights[0] as usize);

    image.channel[0] = channel0.cu_ptr() as *mut _;
    image.channel[1] = channel1.cu_ptr() as *mut _;
    image.channel[2] = channel2.cu_ptr() as *mut _;
        
    let status = nvjpegDecode(
        handle,
        jpeg_state,
        raw_data.as_ptr(),
        raw_data.len(),
        nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
        &mut image,
        device.stream().0 as *mut _,
    );
    check!(status, "Could not decode image. ");


    println!("channel: {channel0:?}");

    // free

    let status = nvjpegJpegStateDestroy(jpeg_state);
    check!(status, "Could not free jpeg state. ");

    let status = nvjpegDestroy(handle);
    check!(status, "Could not free nvjpeg handle. ");


    Ok(Image::default())
}
