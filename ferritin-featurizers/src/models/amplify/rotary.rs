use candle_core::{Device, Error, Result, Tensor, D};

// Example1: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/starcoder2.rs#L22
// Example 2: phi3: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/phi3.rs#L32
pub fn precompute_freqs_cis(dim: usize, end: usize) -> Result<Tensor> {
    let theta: f64 = 10000.;
    let device = Device::Cpu;
    let freqs = (0..dim / 2)
        .into_iter()
        .map(|i| 1.0 / (theta.powf((2 * i) as f64 / dim as f64)));
    let freqs = Tensor::from_iter(freqs, &device)?;
    // Create time steps
    let t = Tensor::from_iter((0..end).map(|x| x as f64), &device)?;

    // Compute outer product
    // let freqs = t.outer(&freqs)?;
    let freqs = t.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;

    // Create complex numbers using cos and sin
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // Stack cos and sin to represent complex numbers
    Tensor::stack(&[cos, sin], D::Minus1)
}

// pub fn reshape_for_broadcast(freqs_cis: &Tensor, x: &Tensor) -> Result<Tensor> {
//     let x_dims = x.dims();
//     println!(
//         "Reshape Tensor Shapes. FREQ and X: {:?} and {:?}",
//         freqs_cis.dims(),
//         x.dims()
//     );
//     // Reshape Tensor Shapes. FREQ and X: [1, 10, 1] and [80, 32, 2]
//     if x_dims.len() < 2 {
//         return Err(candle_core::Error::Msg(
//             "Input tensor must have at least 2 dimensions".to_string(),
//         ));
//     }

//     let freqs_shape = freqs_cis.dims();
//     if freqs_shape != [x_dims[1], x_dims[x_dims.len() - 1]] {
//         return Err(candle_core::Error::Msg(
//             "Frequency tensor shape mismatch".to_string(),
//         ));
//     }

//     // Create new shape for broadcasting
//     let mut new_shape: Vec<usize> = vec![1; x_dims.len()];
//     new_shape[1] = x_dims[1];
//     new_shape[x_dims.len() - 1] = x_dims[x_dims.len() - 1];

//     freqs_cis.reshape(new_shape)
// }

// fn reshape_for_broadcast(freqs_cis: &Tensor, x: &Tensor) -> Result<Tensor> {
//     // Get number of dimensions
//     let ndim = x.dims().len();

//     // Verify dimensions
//     if !(0 <= 1 && 1 < ndim) {
//         return Err(candle_core::Error::Msg(
//             "Invalid dimension index".to_string(),
//         ));
//     }

//     // Get shapes
//     let x_shape = x.dims();
//     let freqs_shape = freqs_cis.dims();

//     // Verify shapes match
//     if freqs_shape != &[x_shape[1], x_shape[ndim - 1]] {
//         return Err(candle_core::Error::Msg(format!(
//             "freqs_cis shape doesn't match expected dimensions\n Freqs_Shape1: {:?}\nShape2: {:?},\nxshape1: {:?}, xshape[n-1]: {:?}",
//             freqs_shape,
//             x_shape,
//             x_shape[1],
//             x_shape[ndim - 1]
//         )));
//     }

//     // Create new shape array
//     let mut new_shape: Vec<usize> = x_shape
//         .iter()
//         .enumerate()
//         .map(|(i, &d)| if i == 1 || i == ndim - 1 { d } else { 1 })
//         .collect();

//     // Reshape tensor
//     freqs_cis.reshape(new_shape)
// }

fn reshape_for_broadcast(freqs_cis: &Tensor, x: &Tensor) -> Result<Tensor> {
    let ndim = x.dims().len();

    // Verify dimensions
    if !(0 <= 1 && 1 < ndim) {
        return Err(candle_core::Error::Msg(
            "Invalid dimension index".to_string(),
        ));
    }

    let x_shape = x.dims();
    let freqs_shape = freqs_cis.dims();

    // Match PyTorch's assertion exactly:
    // assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    if freqs_shape != &[x_shape[1], *x_shape.last().unwrap()] {
        return Err(candle_core::Error::Msg(format!(
            "freqs_cis shape doesn't match expected dimensions\n Expected: [{}, {}]\n Got: {:?}",
            x_shape[1],
            x_shape.last().unwrap(),
            freqs_shape,
        )));
    }

    // Create new shape matching PyTorch's list comprehension:
    // [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    let new_shape: Vec<usize> = x_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| if i == 1 || i == ndim - 1 { d } else { 1 })
        .collect();

    freqs_cis.reshape(new_shape)
}

pub fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    println!("Rotary Embeddings....");
    let xq_shape = xq.dims();
    let last_dim = *xq_shape.last().unwrap();

    // Reshape inputs to separate real and imaginary parts
    let total_elements = xq.elem_count();
    let inferred_dim = total_elements / (last_dim / 2) / 2;
    let xq_reshaped = xq.reshape((inferred_dim, last_dim / 2, 2))?;
    let xk_reshaped = xk.reshape((inferred_dim, last_dim / 2, 2))?;

    println!(
        "Element sizes of inferred_dim, xq_reshaped, xk_reshaped, freqs: {:?}, {:?}, {:?}, {:?}",
        inferred_dim,
        xq_reshaped.dims(),
        xk_reshaped.dims(),
        freqs_cis.dims()
    );

    // Split freqs_cis into cos and sin
    let chunks = freqs_cis.chunk(2, D::Minus1)?;
    let cos = chunks[0].squeeze(D::Minus1)?; // Remove the last dimension of size 1
    let sin = chunks[1].squeeze(D::Minus1)?;

    println!(
        "After squeeze - cos shape: {:?}, sin shape: {:?}",
        cos.dims(),
        sin.dims()
    );

    println!("About to reshape cos ....");
    println!(
        "Element sizes of cos, sin, xq_reshaped: {:?}, {:?}, {:?}",
        sin.dims(),
        cos.dims(),
        xq_reshaped.dims()
    );

    let cos = reshape_for_broadcast(&cos, &xq_reshaped)?;
    println!("About to reshape sin ....");
    let sin = reshape_for_broadcast(&sin, &xq_reshaped)?;

    // Apply rotation
    // For complex multiplication (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    let last_dim = xq_reshaped.dims().len() - 1;
    let xq_real = &xq_reshaped.narrow(last_dim, 0, 1)?; // or .get(last_dim, 0)?
    let xq_imag = &xq_reshaped.narrow(last_dim, 1, 1)?; // or .get(last_dim, 1)?
    let xk_real = &xk_reshaped.narrow(last_dim, 0, 1)?; // or .get(last_dim, 0)?
    let xk_imag = &xk_reshaped.narrow(last_dim, 1, 1)?; // or .get(last_dim, 1)?

    // Compute rotated values
    let xq_out_real = xq_real.mul(&cos)?.sub(&xq_imag.mul(&sin)?)?;
    let xq_out_imag = xq_real.mul(&sin)?.add(&xq_imag.mul(&cos)?)?;
    let xk_out_real = xk_real.mul(&cos)?.sub(&xk_imag.mul(&sin)?)?;
    let xk_out_imag = xk_real.mul(&sin)?.add(&xk_imag.mul(&cos)?)?;

    // Stack real and imaginary parts and reshape back
    let xq_out = Tensor::stack(&[xq_out_real, xq_out_imag], D::Minus1)?.reshape(xq_shape)?;
    let xk_out = Tensor::stack(&[xk_out_real, xk_out_imag], D::Minus1)?.reshape(xk.dims())?;

    Ok((xq_out, xk_out))
}
