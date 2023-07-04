use std::{io::Write, ptr::null_mut, ops::Mul};

use custos::{Buffer, CUDA, prelude::Number};
use nvjpeg_sys::{
    check, nvjpegChromaSubsampling_t, nvjpegCreateSimple, nvjpegDecode, nvjpegDestroy,
    nvjpegGetImageInfo, nvjpegHandle_t, nvjpegImage_t, nvjpegJpegStateCreate,
    nvjpegJpegStateDestroy, nvjpegJpegState_t, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGB,
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

unsafe fn decode_raw_jpeg(
    raw_data: &[u8],
    device: &CUDA,
) -> Result<Image, Box<dyn std::error::Error + Send + Sync>> {
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

    //device.stream().sync()?;

    let channel0 = channel0.read();
    let channel1 = channel1.read();
    let channel2 = channel2.read();


    let filter_rows = 17;
    let filter_cols = 17;
    let filter = vec![1./(filter_rows * filter_cols) as f32; filter_rows * filter_cols];

    let mut channel0_out = vec![0.; channel0.len()];
    correlate_fully(&channel0.iter().map(|x| *x as f32).collect::<Vec<_>>(), &filter, &mut channel0_out, heights[0] as usize, widths[0] as usize, filter_rows, filter_cols);

    let mut channel1_out = vec![0.; channel1.len()];
    correlate_fully(&channel1.iter().map(|x| *x as f32).collect::<Vec<_>>(), &filter, &mut channel1_out, heights[0] as usize, widths[0] as usize, filter_rows, filter_cols);

    let mut channel2_out = vec![0.; channel2.len()];
    correlate_fully(&channel2.iter().map(|x| *x as f32).collect::<Vec<_>>(), &filter, &mut channel2_out, heights[0] as usize, widths[0] as usize, filter_rows, filter_cols);
    
    let channel0_out = channel0_out.into_iter().map(|x| x as u8).collect::<Vec<_>>();
    let channel1_out = channel1_out.into_iter().map(|x| x as u8).collect::<Vec<_>>();
    let channel2_out = channel2_out.into_iter().map(|x| x as u8).collect::<Vec<_>>();


    let file = std::fs::File::create("cat_798x532.ppm")?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write(format!("P6\n{} {}\n255\n", widths[0], heights[0]).as_bytes())?;

    for row in 0..heights[0] {
        let row = row as usize;
        for col in 0..widths[0] {
            let col = col as usize;
            writer.write(&[
                channel0_out[row * widths[0] as usize + col],
                channel1_out[row * widths[0] as usize + col],
                channel2_out[row * widths[0] as usize + col],
            ])?;
        }
    }

    writer.flush()?;

    // free

    let status = nvjpegJpegStateDestroy(jpeg_state);
    check!(status, "Could not free jpeg state. ");

    let status = nvjpegDestroy(handle);
    check!(status, "Could not free nvjpeg handle. ");

    Ok(Image::default())
}

pub fn add_padding<T: Number>(inputs: &[T], inp_rows: usize, inp_cols: usize, x_padding: usize, y_padding: usize) -> Vec<T> {
    let mut padded_inputs = vec![T::zero(); (inp_rows + y_padding * 2) * (inp_cols + x_padding * 2)];

    for inp_row in 0..inp_rows {
        for inp_col in 0..inp_cols {
            padded_inputs[y_padding * (inp_cols + 2*x_padding) + x_padding + inp_row * (inp_cols + 2*x_padding) + inp_col] = inputs[inp_row * inp_cols + inp_col];
        }
    }
    padded_inputs
}

pub fn correlate_fully<T: Number + Mul<U, Output = T>, U: Number>(
    inputs: &[T],
    filter: &[U],
    out: &mut [T],
    inp_rows: usize,
    inp_cols: usize,
    filter_rows: usize,
    filter_cols: usize,
) {
    let x_padding = filter_cols - 1;
    let y_padding = filter_rows - 1;

    let padded_inputs = add_padding(inputs, inp_rows, inp_cols, x_padding, y_padding);
    let padded_rows = inp_rows + y_padding * 2;
    let padded_cols = inp_cols + x_padding * 2;

    // attention: leaves the last padded row, col out
    for move_down in 0..=padded_rows - filter_rows - y_padding {
        for move_right in 0..=padded_cols - filter_cols - x_padding {

            let mut sum = T::default();
            for idx in 0..filter_rows {
                
                let filter_idx = idx * filter_cols;
                let filter_row = &filter[filter_idx..filter_idx + filter_cols];

                let input_idx = move_down * padded_cols + move_right + idx * padded_cols;
                let input_row = &padded_inputs[input_idx..input_idx + filter_cols];
    
                for (filter_row, input_row) in input_row.iter().zip(filter_row) {
                    sum += *filter_row * *input_row;
                }
            }
            out[move_down * inp_cols + move_right] = sum;
        }
    }
}

#[test]
fn test_correlate() {
    let data = [
        1., 2., 3., 5., 
        4., 3., 2., 1.,
        8., 7., 4., 2.,
        7., 3., 2., 1.,
        8., 5., 3., 8.
    ];

    let filter = [1./4., 1./4., 1./4., 1./4.];
    let mut out = [0.; 5*4];
    correlate_fully(&data, &filter, &mut out, 5, 4, 2, 2);

    println!("out: {out:?}");

    let filter = [1./4.; 9];
    let mut out = [0.; 5*4];
    correlate_fully(&data, &filter, &mut out, 5, 4, 3, 3);

    println!("out: {out:?}");
}

#[test]
fn test_add_padding() {
    let data = [
        1, 2, 3, 5, 
        4, 3, 2, 1,
        8, 7, 4, 2,
        7, 3, 2, 1,
        8, 5, 3, 8
    ];

    let padded = add_padding(&data, 5, 4, 2, 2);

    for (idx, padded_val) in padded.iter().enumerate() {
        print!("{padded_val}, ");
        if (idx+1) % (2 + 2 + 4) == 0 {
            println!()
        }
    }
}
