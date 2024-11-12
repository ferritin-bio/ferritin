use candle_core::{DType, Device, Error, Result, Tensor, D};

// pub fn precompute_freqs_cis(head_dim: usize, seq_len: usize) -> Result<Tensor> {
//     println!("in precompute freqs fn!");
//     let theta = 10000f32; // AMPLIFY's theta value
//                           // let theta = Tensor::new(theta, &Device::Cpu)?;

//     let head_dim = head_dim / 2;
//     let theta = Tensor::new(theta, &Device::Cpu)?.expand((head_dim,))?;

//     // Create position and dimension tensors
//     let positions = Tensor::arange(0f32, seq_len as f32, &Device::Cpu)?;
//     let dims = Tensor::arange(0f32, head_dim as f32, &Device::Cpu)?;
//     // let dims = dims.to_dtype(DType::F32)?;
//     println!("in precompute freqs fn! 2");
//     let dims = ((dims * 2.0 as f64)? / head_dim as f64)?;
//     let dims = theta.powf(&dims)?;
//     println!("in precompute freqs fn! 4");
//     let ones = Tensor::ones((seq_len, 1), DType::F32, &Device::Cpu)?;
//     println!("in precompute freqs fn! 5");
//     let dims = dims.reshape((1, head_dim))?;
//     println!("in precompute freqs fn! 6");
//     let dims = ones.matmul(&dims)?;

//     // Calculate frequencies
//     let positions = positions.unsqueeze(1)?;
//     let freqs = positions.matmul(&dims.recip()?.unsqueeze(0)?)?;

//     // Convert to complex representation
//     let freqs_cos = freqs.cos()?;
//     let freqs_sin = freqs.sin()?;

//     println!(
//         "Precomputed freqs shape - cos: {:?}, sin: {:?}",
//         freqs_cos.dims(),
//         freqs_sin.dims()
//     );

//     Tensor::stack(&[freqs_cos, freqs_sin], D::Minus1)
// }

pub fn precompute_freqs_cis(head_dim: usize, seq_len: usize) -> Result<Tensor> {
    println!("in precompute freqs fn!");
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

    println!(
        "Precomputed freqs shape - cos: {:?}, sin: {:?}",
        freqs_cos.dims(),
        freqs_sin.dims()
    );

    Tensor::stack(&[freqs_cos, freqs_sin], D::Minus1)
}
pub fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b_sz, h, seq_len, headdim) = xq.dims4()?;
    println!(
        "Rotary inputs - xq: {:?}, freqs_cis: {:?}",
        xq.dims(),
        freqs_cis.dims()
    );

    // Calculate dimensions for reshape
    let complex_dim = 2;
    let half_headdim = headdim / complex_dim;

    // Reshape inputs for complex multiplication
    let xq = xq.reshape((b_sz, h, seq_len, half_headdim, complex_dim))?;
    let xk = xk.reshape((b_sz, h, seq_len, half_headdim, complex_dim))?;

    // Get appropriate length of freqs_cis and reshape
    let freqs_cis = freqs_cis.narrow(0, 0, seq_len)?;
    let freqs_cis = freqs_cis.reshape((1, 1, seq_len, half_headdim, complex_dim))?;

    println!(
        "Reshaped tensors - xq: {:?}, freqs_cis: {:?}",
        xq.dims(),
        freqs_cis.dims()
    );

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
    let xq_out = xq_out.reshape((b_sz, h, seq_len, headdim))?;
    let xk_out = xk_out.reshape((b_sz, h, seq_len, headdim))?;

    println!(
        "Output shapes - xq: {:?}, xk: {:?}",
        xq_out.dims(),
        xk_out.dims()
    );

    Ok((xq_out, xk_out))
}

// copy from phi3
//
// fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
//     let (b_sz, h, seq_len, headdim) = xq.dims4()?;

//     // Reshape for complex multiplication
//     let xq = xq.reshape((b_sz, h, seq_len, -1, 2))?;
//     let xk = xk.reshape((b_sz, h, seq_len, -1, 2))?;

//     let freqs_cis = freqs_cis.narrow(0, 0, seq_len)?;
//     let freqs_cis = freqs_cis.reshape((1, 1, seq_len, -1, 2))?;

//     let complex_mul = |x: &Tensor| {
//         let real = x.narrow(-1, 0, 1)?.squeeze(-1)?;
//         let imag = x.narrow(-1, 1, 1)?.squeeze(-1)?;
//         let freqs_cos = freqs_cis.narrow(-1, 0, 1)?.squeeze(-1)?;
//         let freqs_sin = freqs_cis.narrow(-1, 1, 1)?.squeeze(-1)?;

//         let real = real.mul(&freqs_cos)?.sub(&imag.mul(&freqs_sin)?)?;
//         let imag = real.mul(&freqs_sin)?.add(&imag.mul(&freqs_cos)?)?;

//         Tensor::stack(&[real, imag], -1)
//     };

//     let xq_out = complex_mul(&xq)?;
//     let xk_out = complex_mul(&xk)?;

//     let xq_out = xq_out.reshape((b_sz, h, seq_len, headdim))?;
//     let xk_out = xk_out.reshape((b_sz, h, seq_len, headdim))?;

//     Ok((xq_out, xk_out))
// }

// fn precompute_freqs_cis(head_dim: usize, seq_len: usize, theta: f32) -> Result<Tensor> {
//     let theta = Tensor::new(theta, &Device::Cpu)?;
//     let head_dim = head_dim / 2;
//     let positions = Tensor::arange(0u32, seq_len as u32, &Device::Cpu)?;
//     let dims = Tensor::arange(0u32, head_dim as u32, &Device::Cpu)?;

//     let dims = dims.mul(2f32)?.div(head_dim as f32)?;
//     let dims = theta.pow(&dims)?;
//     let dims = Tensor::ones(seq_len, &Device::Cpu)?.outer(&dims)?;

//     let positions = positions.unsqueeze(1)?;
//     let freqs = positions.matmul(&dims.reciprocal()?.unsqueeze(0)?)?;

//     let freqs_cos = freqs.cos()?;
//     let freqs_sin = freqs.sin()?;

//     Tensor::stack(&[freqs_cos, freqs_sin], -1)
// }

// Example1: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/starcoder2.rs#L22
// Example 2: phi3: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/phi3.rs#L32
// pub fn precompute_freqs_cis(dim: usize, end: usize) -> Result<Tensor> {
//     let theta: f64 = 10000.;
//     let device = Device::Cpu;
//     let freqs = (0..dim / 2)
//         .into_iter()
//         .map(|i| 1.0 / (theta.powf((2 * i) as f64 / dim as f64)));
//     let freqs = Tensor::from_iter(freqs, &device)?;
//     // Create time steps
//     let t = Tensor::from_iter((0..end).map(|x| x as f64), &device)?;

//     // Compute outer product
//     // let freqs = t.outer(&freqs)?;
//     let freqs = t.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;

//     // Create complex numbers using cos and sin
//     let cos = freqs.cos()?;
//     let sin = freqs.sin()?;

//     // Stack cos and sin to represent complex numbers
//     Tensor::stack(&[cos, sin], D::Minus1)
// }

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

// fn reshape_for_broadcast(freqs_cis: &Tensor, x: &Tensor) -> Result<Tensor> {
//     let ndim = x.dims().len();

//     // Verify dimensions
//     if !(0 <= 1 && 1 < ndim) {
//         return Err(candle_core::Error::Msg(
//             "Invalid dimension index".to_string(),
//         ));
//     }

//     let x_shape = x.dims();
//     let freqs_shape = freqs_cis.dims();

//     // Match PyTorch's assertion exactly:
//     // assert freqs_cis.shape == (x.shape[1], x.shape[-1])
//     if freqs_shape != &[x_shape[1], *x_shape.last().unwrap()] {
//         return Err(candle_core::Error::Msg(format!(
//             "freqs_cis shape doesn't match expected dimensions\n Expected: [{}, {}]\n Got: {:?}",
//             x_shape[1],
//             x_shape.last().unwrap(),
//             freqs_shape,
//         )));
//     }

//     // Create new shape matching PyTorch's list comprehension:
//     // [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
//     let new_shape: Vec<usize> = x_shape
//         .iter()
//         .enumerate()
//         .map(|(i, &d)| if i == 1 || i == ndim - 1 { d } else { 1 })
//         .collect();

//     freqs_cis.reshape(new_shape)
// }

// pub fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
//     println!("Rotary Embeddings....");
//     let xq_shape = xq.dims4()?;
//     let (batch_size, seq_len, n_heads, head_dim) = xq_shape;
//     // let last_dim = *xq_shape.last().unwrap();
//     let last_dim = head_dim;

//     // Reshape inputs to separate real and imaginary parts
//     let total_elements = xq.elem_count();
//     let inferred_dim = total_elements / (last_dim / 2) / 2;
//     let xq_reshaped = xq.reshape((inferred_dim, last_dim / 2, 2))?;
//     let xk_reshaped = xk.reshape((inferred_dim, last_dim / 2, 2))?;

//     println!(
//         "Element sizes of inferred_dim, xq,xq_reshaped, xk_reshaped, freqs: {:?}, {:?}, {:?}, {:?}, {:?}",
//         inferred_dim,
//         xq.dims(),
//         xq_reshaped.dims(),
//         xk_reshaped.dims(),
//         freqs_cis.dims()
//     );

//     // Split freqs_cis into cos and sin
//     // let chunks = freqs_cis.chunk(2, D::Minus1)?;
//     // let cos = chunks[0].squeeze(D::Minus1)?; // Remove the last dimension of size 1
//     // let sin = chunks[1].squeeze(D::Minus1)?;

//     // let chunks = freqs_cis.chunk(2, D::Minus1)?;
//     // let cos = chunks[0].squeeze(D::Minus1)?; // Keep as [8, 32]
//     // let sin = chunks[1].squeeze(D::Minus1)?; // Keep as [8, 32]
//     // println!(
//     //     "After squeeze - cos shape: {:?}, sin shape: {:?}",
//     //     cos.dims(),
//     //     sin.dims()
//     // );

//     // println!("About to reshape cos ....");
//     // println!(
//     //     "Element sizes of cos, sin, xq_reshaped: {:?}, {:?}, {:?}",
//     //     sin.dims(),
//     //     cos.dims(),
//     //     xq_reshaped.dims()
//     // );

//     // let cos = reshape_for_broadcast(&cos, &xq_reshaped)?;
//     // println!("About to reshape sin ....");
//     // let sin = reshape_for_broadcast(&sin, &xq_reshaped)?;

//     // // Apply rotation
//     // // For complex multiplication (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//     // let last_dim = xq_reshaped.dims().len() - 1;
//     // let xq_real = &xq_reshaped.narrow(last_dim, 0, 1)?; // or .get(last_dim, 0)?
//     // let xq_imag = &xq_reshaped.narrow(last_dim, 1, 1)?; // or .get(last_dim, 1)?
//     // let xk_real = &xk_reshaped.narrow(last_dim, 0, 1)?; // or .get(last_dim, 0)?
//     // let xk_imag = &xk_reshaped.narrow(last_dim, 1, 1)?; // or .get(last_dim, 1)?

//     // // Compute rotated values
//     // let xq_out_real = xq_real.mul(&cos)?.sub(&xq_imag.mul(&sin)?)?;
//     // let xq_out_imag = xq_real.mul(&sin)?.add(&xq_imag.mul(&cos)?)?;
//     // let xk_out_real = xk_real.mul(&cos)?.sub(&xk_imag.mul(&sin)?)?;
//     // let xk_out_imag = xk_real.mul(&sin)?.add(&xk_imag.mul(&cos)?)?;

//     // // Stack real and imaginary parts and reshape back
//     // let xq_out = Tensor::stack(&[xq_out_real, xq_out_imag], D::Minus1)?.reshape(xq_shape)?;
//     // let xk_out = Tensor::stack(&[xk_out_real, xk_out_imag], D::Minus1)?.reshape(xk.dims())?;

//     // Ok((xq_out, xk_out))
//     //
//     //

//     // Instead of flattening to 80, keep the batch and head dimensions separate
//     let xq = xq.reshape((batch_size * n_heads, seq_len, head_dim))?;
//     let xk = xk.reshape((batch_size * n_heads, seq_len, head_dim))?;

//     // Split the head_dim in half to match the freq_cis complex dimension
//     let xq = xq.reshape((batch_size * n_heads, seq_len, head_dim / 2, 2))?;
//     let xk = xk.reshape((batch_size * n_heads, seq_len, head_dim / 2, 2))?;

//     // Get cos and sin components
//     let cos = freqs_cis.narrow(2, 0, 1)?.squeeze(2)?;
//     let sin = freqs_cis.narrow(2, 1, 1)?.squeeze(2)?;

//     // Reshape cos and sin for broadcasting
//     let cos = cos.reshape((1, seq_len, head_dim / 2))?;
//     let sin = sin.reshape((1, seq_len, head_dim / 2))?;

//     // Apply rotation
//     let xq_out = apply_rot(xq, &cos, &sin)?;
//     let xk_out = apply_rot(xk, &cos, &sin)?;

//     // Reshape back to original dimensions
//     let xq_out = xq_out.reshape((batch_size, seq_len, n_heads, head_dim))?;
//     let xk_out = xk_out.reshape((batch_size, seq_len, n_heads, head_dim))?;

//     Ok((xq_out, xk_out))
// }

// fn apply_rot(x: Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
//     let x_real = x.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
//     let x_imag = x.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

//     let real = x_real.mul(cos)?.sub(&x_imag.mul(sin)?)?;
//     let imag = x_real.mul(sin)?.add(&x_imag.mul(cos)?)?;

//     Tensor::stack(&[real, imag], D::Minus1)
// }
