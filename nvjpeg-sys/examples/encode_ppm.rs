use std::ptr::null_mut;

use custos::CUDA;
use nvjpeg_sys::{nvjpegEncoderParamsCreate, nvjpegEncoderParams_t, check, nvjpegHandle_t, nvjpegCreateSimple, nvjpegEncoderStateCreate, nvjpegEncoderState_t, nvjpegEncodeImage, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB, nvjpegInputFormat_t_NVJPEG_INPUT_RGB, nvjpegImage_t};


fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw_data = std::fs::read("cat_798x532.ppm")?;
    
    let mut data = Vec::with_capacity(16);
    let mut stop = 0;
    for raw_datum in &raw_data {
        if *raw_datum == '\n' as u8 {
            if stop == 1 {
                break;
            }
            stop += 1;
        }
        data.push(*raw_datum);
    }

    let (_, img_desc_str) = std::str::from_utf8(&data).unwrap().split_once('\n').unwrap();
    let (width, height) = img_desc_str.split_once(' ').unwrap();
    let width = width.parse::<i32>().unwrap();
    let height = height.parse::<i32>().unwrap();

    //println!("w: {width:?}");

    //let raw_data = std::fs::read("cat.jpg")?;

    let device = CUDA::new(0)?;

    let image = unsafe { encode_raw_ppm(width, height, &raw_data, &device)? };

    Ok(())    
}

unsafe fn encode_raw_ppm(width: i32, height: i32, raw_data: &[u8], device: &CUDA) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

    let mut handle: nvjpegHandle_t = null_mut();
    let status = nvjpegCreateSimple(&mut handle);
    check!(status, "Could not create simple handle. ");

    let mut encoder_params: nvjpegEncoderParams_t = null_mut();
    let status = nvjpegEncoderParamsCreate(handle, &mut encoder_params, device.stream().0 as _);
    check!(status, "Could not create encoder params.");

    let mut encoder_state: nvjpegEncoderState_t = null_mut();
    let status = nvjpegEncoderStateCreate(handle, &mut encoder_state, device.stream().0 as _);
    check!(status, "Could not create encoder state.");


    let mut source: nvjpegImage_t = nvjpegImage_t::new();

    source.pitch[0] = width as usize;
    source.pitch[1] = width as usize;
    source.pitch[2] = width as usize;

    
        
    nvjpegEncodeImage(handle, encoder_state, encoder_params, &source, nvjpegInputFormat_t_NVJPEG_INPUT_RGB, width, height, device.stream().0 as _);

    // free resources

    Ok(())
}