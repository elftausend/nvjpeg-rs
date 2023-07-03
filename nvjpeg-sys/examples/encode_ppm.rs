use custos::CUDA;


fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw_data = std::fs::read("cat_798x532.ppm")?;
    //let raw_data = std::fs::read("cat.jpg")?;

    let device = CUDA::new(0)?;

    let image = unsafe { encode_raw_ppm(&raw_data, &device)? };

    Ok(())    
}

unsafe fn encode_raw_ppm(raw_data: &[u8], device: &CUDA) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    
    todo!()
}