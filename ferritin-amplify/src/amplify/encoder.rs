use super::config::AMPLIFYConfig;
use super::rotary::apply_rotary_emb;
use candle_core::{Module, Result, Tensor, D};
use candle_nn::{
    linear, linear_no_bias, ops::softmax_last_dim, rms_norm, Dropout, Linear, RmsNorm, VarBuilder,
};

/// Amplify EncoderBlock implementation
///
/// References for coding the block from similar models.
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
    config: AMPLIFYConfig,
}

impl EncoderBlock {
    pub fn new(config: &AMPLIFYConfig, vb: VarBuilder, layer: i32) -> Result<Self> {
        let multiple_of = 8;
        let intermediate_size = (config.intermediate_size * 2) / 3;
        let intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);
        let vb = vb.pp(layer);
        let q = linear(config.hidden_size, config.hidden_size, vb.pp("q"))?;
        let k = linear(config.hidden_size, config.hidden_size, vb.pp("k"))?;
        let v = linear(config.hidden_size, config.hidden_size, vb.pp("v"))?;
        let wo = linear(config.hidden_size, config.hidden_size, vb.pp("wo"))?;
        let w12 = linear_no_bias(intermediate_size * 2, config.hidden_size, vb.pp("ffn.w12"))?;
        let w3 = linear_no_bias(config.hidden_size, intermediate_size, vb.pp("ffn.w3"))?;
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
            config: config.clone(), // Todo: remove this clone
        })
    }
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
    // process the FFN Block using swiglu
    fn ffn_forward(&self, x: &Tensor) -> Result<Tensor> {
        // Swiglu
        //
        // Todo: see if the apply or add can be done di
        // Store original batch dimensions
        let dims = x.dims();
        let batch_shape = &dims[..dims.len() - 1];
        // Reshape input to 2D: (batch_size, input_dim)
        let x_flat = self.flatten_last_dim(&x)?;
        // Apply packed W1W2 linear transformation
        let w12_out = self.w12.forward(&x_flat)?;
        // Split the output into two halves (for SwiGLU activation)
        let chunks = w12_out.chunk(2, 1)?;
        let x1 = &chunks[0];
        let x2 = &chunks[1];

        // Apply SwiGLU: silu(x1) * x2
        let hidden = x1.silu()?.mul(x2)?;
        // Final linear transformation
        let output = self.w3.forward(&hidden)?;
        // Reshape back to original batch dimensions
        let mut new_shape = batch_shape.to_vec();
        new_shape.push(output.dim(1)?);
        output.reshape(new_shape)
    }
    fn flatten_last_dim(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let last_dim = dims[dims.len() - 1];
        let total_elements = dims.iter().product::<usize>();
        let first_dim = total_elements / last_dim;
        x.reshape((first_dim, last_dim))
    }
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: f64,
        is_causal: bool,
    ) -> Result<Tensor> {
        // Calculate attention scores
        let d_k = key.dim(key.dims().len() - 1)? as f64;
        let scaling = 1.0 / d_k.sqrt();
        // (B, H, L, S) = (batch, heads, query_length, key_length)
        let scores = (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scaling)?;

        // Apply mask if provided
        if let Some(mask) = attn_mask {
            let scores = scores.add(mask)?;
        }
        // Apply softmax
        let attn = softmax_last_dim(&scores)?;

        // Apply dropout if needed
        let attn = if dropout_p > 0.0 {
            candle_nn::ops::dropout(&attn, dropout_p as f32)?
        } else {
            attn
        };
        // Final matrix multiplication with values
        attn.matmul(value)
    }
    fn attention_block(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        freqs_cis: &Tensor,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Query, Key, Value projections
        let (batch_size, seq_len, _) = x.dims3()?;
        // [batch_size, seq_len, hidden_size]
        let xq = self.q.forward(x)?.contiguous()?;
        let xk = self.k.forward(x)?.contiguous()?;
        let xv = self.v.forward(x)?.contiguous()?;
        // Reshape for rotary embeddings
        let xq = xq.reshape((
            batch_size,
            seq_len,
            self.config.num_attention_heads,
            self.d_head,
        ))?;
        let xk = xk.reshape((
            batch_size,
            seq_len,
            self.config.num_attention_heads,
            self.d_head,
        ))?;
        let xv = xv.reshape((
            batch_size,
            seq_len,
            self.config.num_attention_heads,
            self.d_head,
        ))?;
        let (xq, xk) = apply_rotary_emb(&xq, &xk, &freqs_cis)?;
        let dropout_prob = self.config.dropout_prob;

        // need to handle pad_mask better ....
        let pad_mask = if let Some(mask) = pad_mask {
            let (batch_size, seq_len) = (x.dim(0)?, x.dim(1)?);
            let num_heads = self.config.num_attention_heads;

            // Following PyTorch's implementation:
            // 1. unsqueeze twice to add head dimensions
            // 2. repeat to match attention matrix size
            let mask = mask
                .unsqueeze(1)?
                .unsqueeze(1)?
                .expand((batch_size, num_heads, seq_len, seq_len))?; // Expand to full attention size
            Some(mask)
        } else {
            None
        };

        let attn = self.scaled_dot_product_attention(
            &xq.permute((0, 2, 1, 3))?.contiguous()?,
            &xk.permute((0, 2, 1, 3))?.contiguous()?,
            &xv.permute((0, 2, 1, 3))?.contiguous()?,
            pad_mask.as_ref(),
            dropout_prob,
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
        let output = attn.reshape((
            batch_size,
            seq_len,
            self.config.num_attention_heads * self.d_head,
        ))?;
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
        let vb = vb.pp(layer); // handle the layer nubmer here.
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
            config: config.clone(),
        })
    }
}
