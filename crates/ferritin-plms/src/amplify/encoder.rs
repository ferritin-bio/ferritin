use super::config::AMPLIFYConfig;
use super::rotary::apply_rotary_emb;
use candle_core::{D, Module, Result, Tensor};
use candle_nn::{
    Dropout, Linear, RmsNorm, VarBuilder, linear_no_bias, ops::softmax_last_dim, rms_norm,
};

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
    /// process the FFN Block using swiglu
    fn ffn_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _hidden_dim) = x.dims3()?;
        let mut batch_shape = vec![batch, seq_len];
        let x_flat = x.flatten_to(1)?;
        let w12_out = self.w12.forward(&x_flat)?;
        let chunks = w12_out.chunk(2, 1)?;
        let x1 = &chunks[0];
        let x2 = &chunks[1];
        let hidden = x1.silu()?.mul(x2)?;
        let output = self.w3.forward(&hidden)?;
        batch_shape.push(output.dim(1)?);
        output.reshape(batch_shape) // todo fix the shape calculation
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
        let scale = 1.0 / (key.dim(D::Minus1)? as f64).sqrt();
        // (B, H, L, S) = (batch, heads, query_length, key_length)
        let scores = (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let masked_scores = if let Some(mask) = attn_mask {
            scores.add(mask)?
        } else {
            scores
        };
        let attn_weights = softmax_last_dim(&masked_scores)?;
        let attn_probs = if dropout_p > 0.0 {
            candle_nn::ops::dropout(&attn_weights, dropout_p as f32)?
        } else {
            attn_weights
        };
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

        // need to handle pad_mask better ....
        let pad_mask = if let Some(mask) = pad_mask {
            let (batch_size, seq_len) = (x.dim(0)?, x.dim(1)?);
            // Following PyTorch's implementation:
            // 1. unsqueeze twice to add head dimensions
            // 2. repeat to match attention matrix size
            let mask = mask.unsqueeze(1)?.unsqueeze(1)?.expand((
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
            ))?;
            Some(mask)
        } else {
            None
        };
        let attn = self.scaled_dot_product_attention(
            &xq.permute((0, 2, 1, 3))?.contiguous()?,
            &xk.permute((0, 2, 1, 3))?.contiguous()?,
            &xv.permute((0, 2, 1, 3))?.contiguous()?,
            pad_mask.as_ref(),
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
        let intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);
        let vb = vb.pp(layer);
        let q = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("q"))?;
        let k = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("k"))?;
        let v = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("v"))?;
        let wo = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("wo"))?;
        let w12 = linear_no_bias(config.hidden_size, intermediate_size * 2, vb.pp("ffn.w12"))?;
        let w3 = linear_no_bias(intermediate_size, config.hidden_size, vb.pp("ffn.w3"))?;
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
