use candle_core::{Module, Result, Tensor};

// fn rotate_half(x: &Tensor) -> Result<Tensor> {
//     let (x1, x2) = x.chunk(2, -1)?;
//     let neg_x2 = x2.neg()?;
//     Tensor::cat(&[&neg_x2, &x1], -1)
// }

// fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
//     let cos = cos.narrow(1, 0, x.dim(-2)?)?;
//     let sin = sin.narrow(1, 0, x.dim(-2)?)?;

//     let x_cos = x.mul(&cos)?;
//     let x_rot = rotate_half(x)?;
//     let x_sin = x_rot.mul(&sin)?;
//     x_cos.add(&x_sin)
// }

#[derive(Debug)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
    seq_len_cached: Option<usize>,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
}

impl RotaryEmbedding {
    // pub fn new(dim: usize, device: &Device) -> Result<Self> {
    //     let inv_freq = (0..dim)
    //         .step_by(2)
    //         .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
    //         .collect::<Vec<_>>();
    //     let inv_freq = Tensor::new(inv_freq, device)?;

    //     Ok(Self {
    //         inv_freq,
    //         seq_len_cached: None,
    //         cos_cached: None,
    //         sin_cached: None,
    //     })
    // }

    // fn update_cos_sin_tables(
    //     &mut self,
    //     x: &Tensor,
    //     seq_dimension: i64,
    // ) -> Result<(&Tensor, &Tensor)> {
    //     let seq_len = x.dim(seq_dimension)?;

    //     if self.seq_len_cached != Some(seq_len)
    //         || self
    //             .cos_cached
    //             .as_ref()
    //             .map_or(true, |t| t.device() != x.device())
    //     {
    //         self.seq_len_cached = Some(seq_len);
    //         let t = Tensor::arange(0u32, seq_len as u32, x.device())?
    //             .to_dtype(self.inv_freq.dtype())?;
    //         let freqs = t.unsqueeze(1)?.matmul(&self.inv_freq.unsqueeze(0)?)?;
    //         let emb = Tensor::cat(&[&freqs, &freqs], -1)?;

    //         self.cos_cached = Some(emb.cos()?.unsqueeze(0)?);
    //         self.sin_cached = Some(emb.sin()?.unsqueeze(0)?);
    //     }

    //     Ok((
    //         self.cos_cached.as_ref().unwrap(),
    //         self.sin_cached.as_ref().unwrap(),
    //     ))
    // }

    // pub fn forward(&mut self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
    //     let (cos_cached, sin_cached) = self.update_cos_sin_tables(k, -2)?;

    //     Ok((
    //         apply_rotary_pos_emb(q, cos_cached, sin_cached)?,
    //         apply_rotary_pos_emb(k, cos_cached, sin_cached)?,
    //     ))
    // }
}
