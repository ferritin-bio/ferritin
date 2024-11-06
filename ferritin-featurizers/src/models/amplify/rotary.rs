use candle_core::{DType, Device, Result, Tensor};
use std::f64::consts::PI;

pub fn precompute_freqs_cis(dim: usize, end: usize, theta: f64) -> Result<Tensor> {
    let device = Device::Cpu; // Or pass device as parameter

    // Calculate frequencies
    let mut freqs: Vec<f64> = (0..dim / 2)
        .map(|i| 1.0 / (theta.powf((2 * i) as f64 / dim as f64)))
        .collect();
    let freqs = Tensor::from_vec(freqs, (dim / 2,), &device)?;

    // Create time steps
    let t: Vec<f64> = (0..end).map(|x| x as f64).collect();
    let t = Tensor::from_vec(t, (end,), &device)?;

    // Compute outer product
    let freqs = t.outer(&freqs)?;

    // Create complex numbers using cos and sin
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // Stack cos and sin to represent complex numbers
    Tensor::stack(&[cos, sin], -1)
}

pub fn reshape_for_broadcast(freqs_cis: &Tensor, x: &Tensor) -> Result<Tensor> {
    let x_dims = x.dims();
    if x_dims.len() < 2 {
        return Err(candle_core::Error::Msg(
            "Input tensor must have at least 2 dimensions".to_string(),
        ));
    }

    let freqs_shape = freqs_cis.dims();
    if freqs_shape != [x_dims[1], x_dims[x_dims.len() - 1]] {
        return Err(candle_core::Error::Msg(
            "Frequency tensor shape mismatch".to_string(),
        ));
    }

    // Create new shape for broadcasting
    let mut new_shape: Vec<usize> = vec![1; x_dims.len()];
    new_shape[1] = x_dims[1];
    new_shape[x_dims.len() - 1] = x_dims[x_dims.len() - 1];

    freqs_cis.reshape(new_shape)
}

pub fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    let xq_shape = xq.dims();
    let last_dim = *xq_shape.last().unwrap();

    // Reshape inputs to separate real and imaginary parts
    let xq_reshaped = xq.reshape((-1, last_dim / 2, 2))?;
    let xk_reshaped = xk.reshape((-1, last_dim / 2, 2))?;

    // Split freqs_cis into cos and sin
    let (cos, sin) = freqs_cis.split(2, -1)?;
    let cos = reshape_for_broadcast(&cos, &xq_reshaped)?;
    let sin = reshape_for_broadcast(&sin, &xq_reshaped)?;

    // Apply rotation
    // For complex multiplication (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let xq_real = &xq_reshaped.select(-1, 0)?;
    let xq_imag = &xq_reshaped.select(-1, 1)?;
    let xk_real = &xk_reshaped.select(-1, 0)?;
    let xk_imag = &xk_reshaped.select(-1, 1)?;

    // Compute rotated values
    let xq_out_real = xq_real.mul(&cos)?.sub(&xq_imag.mul(&sin)?)?;
    let xq_out_imag = xq_real.mul(&sin)?.add(&xq_imag.mul(&cos)?)?;
    let xk_out_real = xk_real.mul(&cos)?.sub(&xk_imag.mul(&sin)?)?;
    let xk_out_imag = xk_real.mul(&sin)?.add(&xk_imag.mul(&cos)?)?;

    // Stack real and imaginary parts and reshape back
    let xq_out = Tensor::stack(&[xq_out_real, xq_out_imag], -1)?.reshape(xq_shape)?;
    let xk_out = Tensor::stack(&[xk_out_real, xk_out_imag], -1)?.reshape(xk_shape)?;

    Ok((xq_out, xk_out))
}
