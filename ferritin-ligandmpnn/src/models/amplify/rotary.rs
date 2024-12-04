use candle_core::{Device, Result, Tensor, D};

pub fn precompute_freqs_cis(head_dim: usize, seq_len: usize) -> Result<Tensor> {
    let theta: f32 = 10000.0;
    // Create frequencies using powf
    let freqs = (0..head_dim / 2)
        .into_iter()
        .map(|i| 1.0 / (theta.powf((2 * i) as f32 / (head_dim / 2) as f32)));
    let freqs = Tensor::from_iter(freqs, &Device::Cpu)?;

    // Create time steps
    let t = (0..seq_len).map(|x| x as f32);
    let t = Tensor::from_iter(t, &Device::Cpu)?;

    // Compute outer product
    let freqs = t.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;

    // Convert to complex representation
    let freqs_cos = freqs.cos()?;
    let freqs_sin = freqs.sin()?;

    Ok(Tensor::stack(&[freqs_cos, freqs_sin], D::Minus1)?)
}

pub fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b_sz, seq_len, h, headdim) = xq.dims4()?;

    // Calculate dimensions for reshape
    let complex_dim = 2;
    let half_headdim = headdim / complex_dim;

    // Reshape inputs for complex multiplication
    let xq = xq.reshape((b_sz, seq_len, h, half_headdim, complex_dim))?;
    let xk = xk.reshape((b_sz, seq_len, h, half_headdim, complex_dim))?;

    // Reshape freqs_cis to match and broadcast across attention heads
    let freqs_cis = freqs_cis.narrow(0, 0, seq_len)?;
    let freqs_cis = freqs_cis
        .reshape((seq_len, half_headdim, complex_dim))?
        .unsqueeze(0)? // Add batch dim
        .unsqueeze(2)? // Add head dim
        .expand((b_sz, seq_len, h, half_headdim, complex_dim))?; // Expand to match input dimensions

    // Define complex multiplication operation
    let complex_mul = |x: &Tensor| -> Result<Tensor> {
        let real = x.narrow(4, 0, 1)?.squeeze(4)?;
        let imag = x.narrow(4, 1, 1)?.squeeze(4)?;
        let freqs_cos = freqs_cis.narrow(4, 0, 1)?.squeeze(4)?;
        let freqs_sin = freqs_cis.narrow(4, 1, 1)?.squeeze(4)?;

        let real = real.mul(&freqs_cos)?.sub(&imag.mul(&freqs_sin)?)?;
        let imag = real.mul(&freqs_sin)?.add(&imag.mul(&freqs_cos)?)?;

        Tensor::stack(&[real, imag], 4)
    };

    // Apply rotation to query and key
    let xq_out = complex_mul(&xq)?;
    let xk_out = complex_mul(&xk)?;

    // Reshape back to original dimensions
    let xq_out = xq_out.reshape((b_sz, seq_len, h, headdim))?;
    let xk_out = xk_out.reshape((b_sz, seq_len, h, headdim))?;

    Ok((xq_out, xk_out))
}
