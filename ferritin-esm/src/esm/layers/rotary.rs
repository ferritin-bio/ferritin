use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::esm::models::esmc::ESMCConfig;

// NOTE: This implementation is based on LLaMA 2's rotary embeddings
fn rotate_half(x: &Tensor, interleaved: bool) -> Result<Tensor> {
    if !interleaved {
        let (x1, x2) = x.chunk(2, -1)?;
        let neg_x2 = x2.neg();
        Tensor::cat(&[&neg_x2, &x1], -1)
    } else {
        let x1 = x.index_select_along_dim(x.ndim() - 1, 0, 2)?;
        let x2 = x.index_select_along_dim(x.ndim() - 1, 1, 2)?;
        let neg_x2 = x2.neg();
        let stacked = Tensor::stack(&[&neg_x2, &x1], -1)?;
        stacked.flatten_from(-2)
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Result<Tensor> {
    let ro_dim = cos.dim(1)? * 2;
    let (d1, d2, d3, d4) = x.dims4()?;
    assert!(ro_dim <= d4);

    let seqlen = d2;
    let cos = cos.narrow(0, 0, seqlen)?;
    let sin = sin.narrow(0, 0, seqlen)?;

    let cos = cos.unsqueeze(1)?.repeat((1, 1, 2))?;
    let sin = sin.unsqueeze(1)?.repeat((1, 1, 2))?;

    let x_rot = x.narrow(-1, 0, ro_dim)?;
    let x_pass = x.narrow(-1, ro_dim, d4 - ro_dim)?;

    let x_rotated = rotate_half(&x_rot, interleaved)?;
    let x_rot_out = (x_rot * &cos)? + (x_rotated * &sin)?;

    Tensor::cat(&[&x_rot_out, &x_pass], -1)
}

pub struct RotaryEmbedding {
    dim: usize,
    base: f64,
    interleaved: bool,
    // scale_base: Option<f64>,
    scaling_factor: f64,
    seq_len_cached: usize,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
    cos_k_cached: Option<Tensor>,
    sin_k_cached: Option<Tensor>,
    inv_freq: Tensor,
    scale: Option<Tensor>,
}

impl RotaryEmbedding {
    // pub fn new(
    //     dim: usize,
    //     device: &Device,
    //     base: f64,
    //     interleaved: bool,
    //     scale_base: Option<f64>,
    //     scaling_factor: f64,
    // ) -> Result<Self> {
    //     // self,
    //     // dim: int,
    //     // base=10000.0,
    //     // interleaved=False,
    //     // scale_base=None,
    //     // scaling_factor=1.0,
    //     // pos_idx_in_fp32=True,
    //     // device=None,

    //     let inv_freq = Self::compute_inv_freq(dim, base, device)?;

    //     let scale = if let Some(scale_base) = scale_base {
    //         let arange = Tensor::arange(0., dim as f64, 2., device)?;
    //         let scale = (arange + 0.4 * dim as f64) / (1.4 * dim as f64);
    //         Some(scale)
    //     } else {
    //         None
    //     };

    //     Ok(Self {
    //         dim,
    //         base,
    //         interleaved,
    //         scale_base,
    //         scaling_factor,
    //         seq_len_cached: 0,
    //         cos_cached: None,
    //         sin_cached: None,
    //         cos_k_cached: None,
    //         sin_k_cached: None,
    //         inv_freq,
    //         scale,
    //     })
    // }
    pub fn load(vb: VarBuilder, config: &ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model, n_heads, ..
        } = config;

        let rotary_dims = d_model / n_heads;
        let base = 10000.0;
        let device = vb.device();
        let interleaved = false;
        let scaling_factor = 1.0;
        // scale_base=None,
        // scaling_factor=1.0,
        // pos_idx_in_fp32=True,

        let inv_freq = Self::compute_inv_freq(rotary_dims, base, device)?;
        let arange = Tensor::arange(0., rotary_dims as f64, device)?;
        let scale = {
            let numerator = (&arange + (0.4 * rotary_dims as f64))?;
            let denominator = 1.4 * rotary_dims as f64;
            numerator / denominator
        };

        Ok(Self {
            dim: rotary_dims,
            base,
            interleaved,
            // scale_base,
            scaling_factor,
            seq_len_cached: 0,
            cos_cached: None,
            sin_cached: None,
            cos_k_cached: None,
            sin_k_cached: None,
            inv_freq,
            scale: Some(scale?),
        })
    }
    fn compute_inv_freq(dim: usize, base: f64, device: &Device) -> Result<Tensor> {
        let arange = Tensor::arange(0., dim as f64, 2., device)?;
        let inv_freq = (base.powf(arange / dim as f64)).recip();
        Ok(inv_freq)
    }

    fn update_cos_sin_cache(&mut self, seqlen: usize) -> Result<()> {
        if seqlen > self.seq_len_cached || self.cos_cached.is_none() {
            self.seq_len_cached = seqlen;

            let t = (Tensor::arange(0., seqlen as f64, 1., self.inv_freq.device())?)
                / self.scaling_factor;
            let freqs = t.outer(&self.inv_freq)?;

            if self.scale.is_none() {
                self.cos_cached = Some(freqs.cos()?);
                self.sin_cached = Some(freqs.sin()?);
            } else {
                let scale = self.scale.as_ref().unwrap();
                let power = ((Tensor::arange(0., seqlen as f64, 1., scale.device())?
                    - (seqlen / 2) as f64)
                    / self.scale_base.unwrap())?;
                let scale = scale.pow(&power.unsqueeze(-1)?)?;

                let cos = freqs.cos()?;
                let sin = freqs.sin()?;

                self.cos_cached = Some((&cos * &scale)?);
                self.sin_cached = Some((&sin * &scale)?);
                self.cos_k_cached = Some((&cos / &scale)?);
                self.sin_k_cached = Some((&sin / &scale)?);
            }
        }
        Ok(())
    }

    // pub fn forward(
    //     &mut self,
    //     q: &Tensor,
    //     k: &Tensor,
    //     seqlen_offset: usize,
    // ) -> Result<(Tensor, Tensor)> {
    //     let seqlen = q.dim(1)? + seqlen_offset;
    //     self.update_cos_sin_cache(seqlen)?;

    //     if self.scale.is_none() {
    //         let cos = self
    //             .cos_cached
    //             .as_ref()
    //             .unwrap()
    //             .narrow(0, seqlen_offset, q.dim(1)?)?;
    //         let sin = self
    //             .sin_cached
    //             .as_ref()
    //             .unwrap()
    //             .narrow(0, seqlen_offset, q.dim(1)?)?;

    //         let q_out = apply_rotary_emb(q, &cos, &sin, self.interleaved)?;
    //         let k_out = apply_rotary_emb(k, &cos, &sin, self.interleaved)?;

    //         Ok((q_out, k_out))
    //     } else {
    //         panic!("Scaled rotary embeddings not implemented");
    //     }
    // }
}
