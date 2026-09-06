//! AMPLIFY is an optimized transformer model focused on optimizing the context of sequence models
//! while maintaining computational efficiency.
//!
//! Key features:
//! - Rotary positional embeddings
//! - RMSNorm for improved training stability
//! - SwiGLU activation function
//! - Specialized architecture optimizations
//! - Memory efficient inference
//!
//!
use super::config::AMPLIFYConfig;
use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::{
    Dropout, Embedding, Linear, RmsNorm, VarBuilder, embedding, linear, linear_no_bias,
    ops::softmax_last_dim, rms_norm,
};
use tokenizers::Tokenizer;

/// The AMPLIFY model
///
/// - [GH PythonModel](https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/src/amplify/model/amplify.py)
/// - [paper](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
/// - [HF](https://huggingface.co/chandar-lab/AMPLIFY_120M)
///
#[derive(Debug)]
pub struct AMPLIFY {
    encoder: Embedding,
    transformer_encoder: Vec<EncoderBlock>,
    layer_norm_2: RmsNorm,
    decoder: Linear,
    freqs_cis: Tensor,
    config: AMPLIFYConfig,
}
impl AMPLIFY {
    fn process_attention_mask(&self, pad_mask: Option<&Tensor>) -> Result<Option<Tensor>> {
        match pad_mask {
            None => Ok(None),
            Some(mask) => {
                let batch_size = mask.dim(0)?;
                let seq_length = mask.dim(D::Minus1)?;
                let num_heads = self.config.num_attention_heads;
                // (batch, seq) -> (batch, 1, 1, seq) -> (batch, heads, seq, seq)
                mask.unsqueeze(1)?
                    .unsqueeze(1)?
                    .expand((batch_size, num_heads, seq_length, seq_length))
                    .map(Some)
            }
        }
    }
    pub fn forward(
        &self,
        src: &Tensor,
        pad_mask: Option<&Tensor>,
        output_hidden_states: bool,
        output_attentions: bool,
    ) -> Result<AmplifyOutput> {
        let mut hidden_states = vec![];
        let mut attentions = vec![];
        let attention_mask = self.process_attention_mask(pad_mask)?;
        let freqs_cis = self.freqs_cis.narrow(0, 0, src.dim(1)?)?;
        let mut x = self.encoder.forward(src)?.contiguous()?;
        for layer in self.transformer_encoder.iter() {
            let (new_x, attn) =
                layer.forward(&x, attention_mask.as_ref(), &freqs_cis, output_attentions)?;
            x = new_x;
            if output_hidden_states {
                hidden_states.push(x.clone());
            }
            if output_attentions && let Some(attn) = attn {
                attentions.push(attn);
            }
        }
        // Final layer norm and decoder
        let logits = if self.config.layer_norm_before_last_layer {
            self.decoder.forward(&self.layer_norm_2.forward(&x)?)?
        } else {
            self.decoder.forward(&x)?
        };
        Ok(AmplifyOutput {
            logits,
            hidden_states: if output_hidden_states {
                Some(hidden_states)
            } else {
                None
            },
            attentions: if output_attentions {
                Some(attentions)
            } else {
                None
            },
        })
    }
    pub fn load(vb: VarBuilder, cfg: &AMPLIFYConfig) -> Result<Self> {
        let transformer_encoder = (0..cfg.num_hidden_layers)
            .map(|i| EncoderBlock::load(vb.pp("transformer_encoder"), cfg, i as i32))
            .collect::<Result<Vec<_>>>()?;
        let encoder = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("encoder"))?;
        let layer_norm_2 = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("layer_norm_2"))?;
        let decoder = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("decoder"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        // The rotary table is computed in F32 for precision, then cast to the
        // model's dtype — without the cast, an F16/BF16 model fails in
        // apply_rotary_emb with "dtype mismatch in mul" (ferritin-100.9).
        let freqs_cis = precompute_freqs_cis(head_dim, cfg.max_length)?
            .to_device(vb.device())?
            .to_dtype(vb.dtype())?;
        Ok(Self {
            encoder,
            transformer_encoder,
            layer_norm_2,
            decoder,
            freqs_cis,
            config: cfg.clone(),
        })
    }
    /// The configuration this model was loaded with.
    pub fn config(&self) -> &AMPLIFYConfig {
        &self.config
    }
    pub fn get_device(&self) -> &Device {
        self.freqs_cis.device()
    }
    pub fn load_tokenizer() -> Result<Tokenizer> {
        let tokenizer_bytes = include_bytes!("tokenizer.json");
        Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))
    }
}

/// Precomputes frequency-based complex rotation matrices for rotary embeddings.
///
/// # Arguments
/// * `head_dim` - Dimension of each attention head
/// * `seq_len` - Maximum sequence length to precompute
///
/// # Returns
/// A tensor of shape [seq_len, head_dim/2, 2] containing cos and sin values
///
pub fn precompute_freqs_cis(head_dim: usize, seq_len: usize) -> Result<Tensor> {
    let half_dim = head_dim / 2;
    let theta = 10000.0_f32;
    // AMPLIFY rotary.py: freqs = 1 / theta ** (arange(0, head_dim, 2) / head_dim),
    // i.e. exponent 2*i / head_dim for i in 0..head_dim/2. The denominator is the
    // full head_dim, NOT half_dim — dividing by half_dim makes the frequencies
    // decay twice as fast and skews rotary phase with position.
    let freqs = Tensor::from_iter(
        (0..half_dim).map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32)),
        &Device::Cpu,
    )?;
    let t = (0..seq_len).map(|x| x as f32);
    let t = Tensor::from_iter(t, &Device::Cpu)?;
    let freqs = t.unsqueeze(1)?.matmul(&freqs.unsqueeze(0)?)?;
    let freqs_cos = freqs.cos()?;
    let freqs_sin = freqs.sin()?;
    Tensor::stack(&[freqs_cos, freqs_sin], D::Minus1)
}

pub fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, freqs_cis: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b_sz, seq_len, h, headdim) = xq.dims4()?;
    let half_headdim = headdim / 2;
    let xq = xq.reshape((b_sz, seq_len, h, half_headdim, 2))?;
    let xk = xk.reshape((b_sz, seq_len, h, half_headdim, 2))?;
    let freqs_cis = freqs_cis.narrow(0, 0, seq_len)?;
    let freqs_cis = freqs_cis
        .reshape((seq_len, half_headdim, 2))?
        .unsqueeze(0)?
        .unsqueeze(2)?
        .expand((b_sz, seq_len, h, half_headdim, 2))?;
    let complex_mul = |x: &Tensor| -> Result<Tensor> {
        let real = x.narrow(4, 0, 1)?.squeeze(4)?;
        let imag = x.narrow(4, 1, 1)?.squeeze(4)?;
        let freqs_cos = freqs_cis.narrow(4, 0, 1)?.squeeze(4)?;
        let freqs_sin = freqs_cis.narrow(4, 1, 1)?.squeeze(4)?;
        // Complex rotation: (real + i*imag) * (cos + i*sin)
        // new_real = real*cos - imag*sin
        // new_imag = real*sin + imag*cos  (must use original `real`, not new_real)
        let new_real = real.mul(&freqs_cos)?.sub(&imag.mul(&freqs_sin)?)?;
        let new_imag = real.mul(&freqs_sin)?.add(&imag.mul(&freqs_cos)?)?;
        Tensor::stack(&[new_real, new_imag], 4)
    };
    let xq_out = complex_mul(&xq)?.reshape((b_sz, seq_len, h, headdim))?;
    let xk_out = complex_mul(&xk)?.reshape((b_sz, seq_len, h, headdim))?;
    Ok((xq_out, xk_out))
}

/// An encoder block in the AMPLIFY transformer architecture.
///
/// This implements a standard transformer encoder block with:
/// - Multi-head self-attention with rotary positional embeddings
/// - Feed-forward network with SwiGLU activation
/// - RMSNorm for layer normalization
///
/// # Arguments
/// * `config` - Configuration parameters for the model
/// * `vb` - Variable builder for loading weights
/// * `layer` - Layer index in the transformer stack
///
/// - [T5](https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/t5.rs#L331)
/// - [distilbert](https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/distilbert.rs#L198)
/// - [glm4](https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/glm4.rs#L340)
/// - [SwiGLu Implementation](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/swiglu_op.py#L462)
#[derive(Debug)]
pub struct EncoderBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    wo: Linear,
    resid_dropout: Dropout,
    w12: Linear,
    w3: Linear,
    ffn_norm: RmsNorm,
    attention_norm: RmsNorm,
    ffn_dropout: Dropout,
    d_head: usize,
    num_heads: usize,
    dropout_prob: f64,
}
impl EncoderBlock {
    pub fn forward(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        freqs_cis: &Tensor,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed = self.attention_norm.forward(x)?;
        let (attn, contacts) =
            self.attention_block(&normed, pad_mask, freqs_cis, output_attentions)?;
        let x = x.add(&attn)?;
        let normed = self.ffn_norm.forward(&x)?;
        let ffn_output = self.ffn_forward(&normed)?;
        let ff = self.ffn_dropout.forward(&ffn_output, false)?; // Todo: pass in the Inference/Training bit
        let x = x.add(&ff)?;
        Ok((x, contacts))
    }
    /// Process the FFN block using SwiGLU activation.
    /// w12 projects to 2×intermediate_size; we split and gate with silu.
    fn ffn_forward(&self, x: &Tensor) -> Result<Tensor> {
        let chunks = self.w12.forward(x)?.chunk(2, D::Minus1)?;
        let hidden = chunks[0].silu()?.mul(&chunks[1])?;
        self.w3.forward(&hidden)
    }
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: f64,
        _is_causal: bool,
    ) -> Result<Tensor> {
        // Calculate scaled attention scores (B, H, L, S)
        // (B, H, L, S) = (batch, heads, query_length, key_length)
        let scale = 1.0 / (key.dim(D::Minus1)? as f64).sqrt();
        let scores = (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let masked_scores = attn_mask.map_or(Ok(scores.clone()), |mask| scores.add(mask))?;
        let attn_weights = softmax_last_dim(&masked_scores)?;
        let attn_probs = if dropout_p > 0.0 {
            candle_nn::ops::dropout(&attn_weights, dropout_p as f32)
        } else {
            Ok(attn_weights)
        }?;
        attn_probs.matmul(value)
    }
    fn attention_block(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        freqs_cis: &Tensor,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len, _hidden_size) = x.dims3()?;
        let xq = self.q.forward(x)?.contiguous()?;
        let xk = self.k.forward(x)?.contiguous()?;
        let xv = self.v.forward(x)?.contiguous()?;
        // Reshape for rotary embeddings
        let shape = (batch_size, seq_len, self.num_heads, self.d_head);
        let xq = xq.reshape(shape)?;
        let xk = xk.reshape(shape)?;
        let xv = xv.reshape(shape)?;
        let (xq, xk) = apply_rotary_emb(&xq, &xk, freqs_cis)?;
        // pad_mask arrives pre-expanded to (batch, heads, seq, seq) from process_attention_mask
        let attn = self.scaled_dot_product_attention(
            &xq.permute((0, 2, 1, 3))?.contiguous()?,
            &xk.permute((0, 2, 1, 3))?.contiguous()?,
            &xv.permute((0, 2, 1, 3))?.contiguous()?,
            pad_mask,
            self.dropout_prob,
            false,
        )?;
        // `[batch, num_heads, seq_len, head_dim]` → `[batch, seq_len, num_heads, head_dim]`
        let attn = attn.permute((0, 2, 1, 3))?;
        let _attn = if output_attentions {
            let xq_t = xq.permute((0, 2, 1, 3))?.contiguous()?;
            let xk_t = xk.permute((0, 2, 3, 1))?.contiguous()?;
            let mut attn_weights = xq_t.matmul(&xk_t)?;
            let scale = (xq.dim(D::Minus1)? as f64).sqrt();
            attn_weights = (attn_weights / scale)?;
            // attn_weights = attn_weights.add(pad_mask)?;  <- Todo. Revisit
            Some(softmax_last_dim(&attn_weights)?)
        } else {
            None
        };
        // Final projection and dropout
        let output = attn.reshape((batch_size, seq_len, self.num_heads * self.d_head))?;
        let output01 = self.wo.forward(&output)?;
        let output02 = self.resid_dropout.forward(&output01, false)?;
        Ok((output02, _attn))
    }
    /// Load Weights from a Model
    pub fn load(vb: VarBuilder, config: &AMPLIFYConfig, layer: i32) -> Result<Self> {
        // To keep the number of parameters and the amount of computation constant, we reduce the number of
        // hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
        // avoid RuntimeError due to misaligned operand
        let multiple_of = 8;
        let intermediate_size = (config.intermediate_size * 2) / 3;
        let intermediate_size = multiple_of * intermediate_size.div_ceil(multiple_of);
        let vb = vb.pp(layer);
        let attn_linear = |d_in, d_out, vb| {
            if config.att_bias {
                linear(d_in, d_out, vb)
            } else {
                linear_no_bias(d_in, d_out, vb)
            }
        };
        let ffn_linear = |d_in, d_out, vb| {
            if config.ffn_bias {
                linear(d_in, d_out, vb)
            } else {
                linear_no_bias(d_in, d_out, vb)
            }
        };
        let q = attn_linear(config.hidden_size, config.hidden_size, vb.pp("q"))?;
        let k = attn_linear(config.hidden_size, config.hidden_size, vb.pp("k"))?;
        let v = attn_linear(config.hidden_size, config.hidden_size, vb.pp("v"))?;
        let wo = attn_linear(config.hidden_size, config.hidden_size, vb.pp("wo"))?;
        let w12 = ffn_linear(config.hidden_size, intermediate_size * 2, vb.pp("ffn.w12"))?;
        let w3 = ffn_linear(intermediate_size, config.hidden_size, vb.pp("ffn.w3"))?;
        let ffn_norm = rms_norm(config.hidden_size, config.norm_eps, vb.pp("ffn_norm"))?;
        let attention_norm =
            rms_norm(config.hidden_size, config.norm_eps, vb.pp("attention_norm"))?;
        Ok(Self {
            q,
            k,
            v,
            wo,
            resid_dropout: Dropout::new(config.dropout_prob as f32),
            w12,
            w3,
            attention_norm,
            ffn_norm,
            ffn_dropout: Dropout::new(config.dropout_prob as f32),
            d_head: config.hidden_size / config.num_attention_heads,
            num_heads: config.num_attention_heads,
            dropout_prob: config.dropout_prob,
        })
    }
}

// Helper structs and enums
#[derive(Debug)]
/// Amplify Model Output
///
/// logits, hidden states, and attentions.
///
///  logits -> distribution of the sequences.
///  attentions -> contact map
pub struct AmplifyOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

impl AmplifyOutput {
    /// "Perform average product correct, used for contact prediction."
    /// https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/examples/utils.py#L83
    /// "Perform average product correct, used for contact prediction."
    fn apc(&self, x: &Tensor) -> Result<Tensor> {
        let a1 = x.sum_keepdim(D::Minus1)?;
        let a2 = x.sum_keepdim(D::Minus2)?;
        let a12 = x.sum_keepdim((D::Minus1, D::Minus2))?;
        let avg = a1.matmul(&a2)?;
        let avg = avg.div(&a12.broadcast_as(avg.shape())?)?;
        x.sub(&avg)
    }
    // From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
    // https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/examples/utils.py#L77
    // "Make layer symmetric in final two dimensions, used for contact prediction."
    fn symmetrize(&self, x: &Tensor) -> Result<Tensor> {
        let x_transpose = x.transpose(D::Minus1, D::Minus2)?;
        x.add(&x_transpose)
    }
    /// Contact maps can be obtained from the self-attentions
    pub fn get_contact_map(&self) -> Result<Option<Tensor>> {
        let Some(attentions) = &self.attentions else {
            return Ok(None);
        };
        // we need the dimensions to reshape below.
        // the attention blocks have the following shape
        let (_batch, _n_head, _seq_length, seq_length) = attentions.first().unwrap().dims4()?;
        let attn_stacked = Tensor::stack(attentions, 0)?;
        let total_elements = attn_stacked.dims().iter().product::<usize>();
        let first_dim = total_elements / (seq_length * seq_length);
        let attn_map = attn_stacked.reshape(&[first_dim, seq_length, seq_length])?;
        // In PyTorch: attn_map = attn_map[:, 1:-1, 1:-1]
        let attn_map = attn_map
            .narrow(1, 1, attn_map.dim(1)? - 2)? // second dim
            .narrow(2, 1, attn_map.dim(2)? - 2)?; // third dim
        let symmetric = self.symmetrize(&attn_map)?;
        let normalized = self.apc(&symmetric)?;
        let proximity_map = normalized.permute((1, 2, 0))?; //  # (residues, residues, map)

        Ok(Some(proximity_map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    // -----------------------------------------------------------------------
    // apply_rotary_emb correctness
    // -----------------------------------------------------------------------

    /// A zero-angle rotation (cos=1, sin=0) must be the identity.
    /// freqs_cis shape: (seq, half_dim, 2) with cos=1, sin=0 everywhere.
    #[test]
    fn test_rotary_emb_identity() -> Result<()> {
        let device = Device::Cpu;
        let (b, seq, h, d) = (1usize, 4, 2, 8);
        let half = d / 2;

        // xq: all ones
        let xq = Tensor::ones(&[b, seq, h, d], candle_core::DType::F32, &device)?;
        let xk = Tensor::ones(&[b, seq, h, d], candle_core::DType::F32, &device)?;

        // freqs_cis: cos=1, sin=0  →  identity rotation
        let mut cos_sin = vec![0f32; seq * half * 2];
        for i in 0..seq * half {
            cos_sin[i * 2] = 1.0; // cos
            cos_sin[i * 2 + 1] = 0.0; // sin
        }
        let freqs_cis = Tensor::from_vec(cos_sin, &[seq, half, 2], &device)?;

        let (xq_out, xk_out) = apply_rotary_emb(&xq, &xk, &freqs_cis)?;

        let xq_vals: Vec<f32> = xq_out.flatten_all()?.to_vec1()?;
        let xk_vals: Vec<f32> = xk_out.flatten_all()?.to_vec1()?;
        for v in xq_vals.iter().chain(xk_vals.iter()) {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "identity rotation should preserve all-ones input, got {v}"
            );
        }
        Ok(())
    }

    /// A 90-degree rotation (cos=0, sin=1): real→-imag, imag→real.
    /// Input: real=1, imag=0  →  expected output: new_real=0, new_imag=1.
    #[test]
    fn test_rotary_emb_90deg() -> Result<()> {
        let device = Device::Cpu;
        let (b, seq, h, d) = (1usize, 1, 1, 2); // d=2 so half=1
        let half = d / 2; // 1

        // xq: [real=1.0, imag=0.0]
        let xq = Tensor::from_vec(vec![1.0f32, 0.0], &[b, seq, h, d], &device)?;
        let xk = xq.clone();

        // freqs_cis: cos=0, sin=1  →  90-degree rotation
        let freqs_cis = Tensor::from_vec(vec![0.0f32, 1.0], &[seq, half, 2], &device)?;

        let (xq_out, _) = apply_rotary_emb(&xq, &xk, &freqs_cis)?;
        let vals: Vec<f32> = xq_out.flatten_all()?.to_vec1()?;

        // new_real = 1*0 - 0*1 = 0
        // new_imag = 1*1 + 0*0 = 1
        assert!(
            vals[0].abs() < 1e-5,
            "new_real should be 0 after 90-deg rotation, got {}",
            vals[0]
        );
        assert!(
            (vals[1] - 1.0).abs() < 1e-5,
            "new_imag should be 1 after 90-deg rotation, got {}",
            vals[1]
        );
        Ok(())
    }

    /// Verify the old shadowing bug would have given a wrong answer.
    /// With the bug: new_imag = new_real*sin + imag*cos (uses wrong real).
    /// This test passes only because we fixed it.
    #[test]
    fn test_rotary_emb_non_trivial() -> Result<()> {
        let device = Device::Cpu;
        let (b, seq, h, d) = (1usize, 1, 1, 2);
        let half = d / 2;

        // real=3, imag=4; angle=pi/6: cos=√3/2, sin=1/2
        let cos = (std::f32::consts::PI / 6.0).cos();
        let sin = (std::f32::consts::PI / 6.0).sin();
        let (r, im) = (3.0f32, 4.0f32);

        let xq = Tensor::from_vec(vec![r, im], &[b, seq, h, d], &device)?;
        let freqs_cis = Tensor::from_vec(vec![cos, sin], &[seq, half, 2], &device)?;

        let (xq_out, _) = apply_rotary_emb(&xq, &xq.clone(), &freqs_cis)?;
        let vals: Vec<f32> = xq_out.flatten_all()?.to_vec1()?;

        let expected_real = r * cos - im * sin;
        let expected_imag = r * sin + im * cos; // uses original r, not expected_real
        assert!(
            (vals[0] - expected_real).abs() < 1e-5,
            "new_real mismatch: got {}, expected {expected_real}",
            vals[0]
        );
        assert!(
            (vals[1] - expected_imag).abs() < 1e-5,
            "new_imag mismatch: got {}, expected {expected_imag}",
            vals[1]
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ffn_forward shape
    // -----------------------------------------------------------------------

    /// After simplification, ffn_forward must preserve (batch, seq, hidden) shape.
    #[test]
    fn test_ffn_forward_shape() -> Result<()> {
        use candle_nn::{VarBuilder, VarMap};
        let device = Device::Cpu;
        let cfg = AMPLIFYConfig::amp_120m();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        // Load a single encoder block (layer 0)
        let block = EncoderBlock::load(vb.pp("transformer_encoder"), &cfg, 0)?;

        let (batch, seq) = (2usize, 7usize);
        let x = Tensor::randn(0f32, 1f32, &[batch, seq, cfg.hidden_size], &device)?;
        let out = block.ffn_forward(&x)?;
        assert_eq!(
            out.shape().dims(),
            &[batch, seq, cfg.hidden_size],
            "ffn_forward shape mismatch"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // process_attention_mask: correct head count
    // -----------------------------------------------------------------------

    /// Verify that process_attention_mask expands to (batch, num_attention_heads, seq, seq),
    /// NOT (batch, num_hidden_layers, seq, seq). Uses a 1-layer config so load is fast.
    #[test]
    fn test_process_attention_mask_shape() -> Result<()> {
        use candle_nn::{VarBuilder, VarMap};
        let device = Device::Cpu;
        // Minimal 1-layer config so AMPLIFY::load is fast
        let cfg = AMPLIFYConfig {
            num_hidden_layers: 1,
            hidden_size: 64,
            num_attention_heads: 4,
            intermediate_size: 256,
            vocab_size: 27,
            max_length: 32,
            ..AMPLIFYConfig::amp_120m()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let model = AMPLIFY::load(vb, &cfg)?;

        let (batch, seq) = (2usize, 10usize);
        let mask = Tensor::ones(&[batch, seq], candle_core::DType::F32, &device)?;
        let expanded = model
            .process_attention_mask(Some(&mask))?
            .expect("expected Some mask");

        // Must be heads (4), not layers (1) — they differ, so this catches the old bug
        assert_eq!(
            expanded.shape().dims(),
            &[batch, cfg.num_attention_heads, seq, seq],
            "mask must expand to (batch, num_attention_heads, seq, seq)"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // precompute_freqs_cis shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_precompute_freqs_cis_shape() -> Result<()> {
        let head_dim = 64;
        let seq_len = 128;
        let freqs = precompute_freqs_cis(head_dim, seq_len)?;
        assert_eq!(
            freqs.shape().dims(),
            &[seq_len, head_dim / 2, 2],
            "freqs_cis shape should be (seq_len, head_dim/2, 2)"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // AmplifyOutput contact map
    // -----------------------------------------------------------------------

    /// Builds a minimal AmplifyOutput with synthetic attention tensors and
    /// checks that get_contact_map returns the right shape.
    #[test]
    fn test_contact_map_output_shape() -> Result<()> {
        let device = Device::Cpu;
        // Simulate 2 layers, 4 heads, seq_len=9 (including BOS/EOS)
        let (layers, heads, seq_len) = (2usize, 4usize, 9usize);
        let attn_shape = &[1usize, heads, seq_len, seq_len];
        let single = Tensor::rand(0f32, 1f32, attn_shape, &device)?;
        let attentions: Vec<Tensor> = (0..layers).map(|_| single.clone()).collect();

        let output = AmplifyOutput {
            logits: Tensor::zeros(
                &[1usize, seq_len, 27usize],
                candle_core::DType::F32,
                &device,
            )?,
            hidden_states: None,
            attentions: Some(attentions),
        };

        let contact_map = output.get_contact_map()?.expect("expected contact map");
        // BOS and EOS stripped: seq_len - 2
        let expected = seq_len - 2;
        assert_eq!(
            contact_map.shape().dims()[0],
            expected,
            "contact map dim 0 should be seq_len-2"
        );
        assert_eq!(
            contact_map.shape().dims()[1],
            expected,
            "contact map dim 1 should be seq_len-2"
        );
        Ok(())
    }
}
