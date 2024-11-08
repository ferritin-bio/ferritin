//! AMPLIFY is a optimized transformer model focused on optimizing the context of sequence models
//! while maintaining computational efficiency.
//!
//! Key features:
//! - Rotary positional embeddings
//! - RMSNorm for improved training stability
//! - SwiGLU activation function
//! - Specialized architecture optimizations
//! - Memory efficient inference

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, rms_norm, Activation, Dropout, Embedding, Linear, RmsNorm,
    VarBuilder,
};

// Config struct
#[derive(Debug, Clone)]
pub struct AMPLIFYConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub dropout_prob: f64,
    pub embedding_init_range: f64,
    pub decoder_init_range: f64,
    pub rms_norm: bool,
    pub norm_eps: f64,
    pub hidden_act: Activation,
    pub layer_norm_after_embedding: bool,
    pub layer_norm_before_last_layer: bool,
    pub vocab_size: usize,
    pub ffn_bias: bool,
    pub att_bias: bool,
    pub pad_token_id: usize,
    pub max_length: usize,
}

impl Default for AMPLIFYConfig {
    fn default() -> Self {
        Self {
            hidden_size: 960,
            num_hidden_layers: 32,
            num_attention_heads: 15,
            intermediate_size: 3840,
            dropout_prob: 0.0,
            embedding_init_range: 0.02,
            decoder_init_range: 0.02,
            rms_norm: true,
            norm_eps: 1e-5,
            hidden_act: Activation::Swiglu,
            layer_norm_after_embedding: false,
            layer_norm_before_last_layer: true,
            vocab_size: 27,
            ffn_bias: false,
            att_bias: false,
            pad_token_id: 0,
            max_length: 2048,
        }
    }
}

impl AMPLIFYConfig {
    pub fn amp_120m(&self) -> Self {
        Self {
            hidden_size: 640,
            num_hidden_layers: 24,
            num_attention_heads: 10,
            intermediate_size: 2560,
            dropout_prob: 0.0,
            embedding_init_range: 0.02,
            decoder_init_range: 0.02,
            rms_norm: true,
            norm_eps: 1e-5,
            hidden_act: Activation::Swiglu,
            layer_norm_after_embedding: false,
            layer_norm_before_last_layer: true,
            vocab_size: 27,
            ffn_bias: false,
            att_bias: false,
            pad_token_id: 0,
            max_length: 2048,
        }
    }
    pub fn amp_350m(self) -> Self {
        AMPLIFYConfig::default()
    }
}

/// Amplify EncoderBlock implementation
///
/// example 01: T5: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/t5.rs#L331
//
/// Example 01: FFN: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/distilbert.rs#L198
/// Example: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/glm4.rs#L340
/// SwiGLu Implementation:  https://github.com/facebookresearch/xformers/blob/main/xformers/ops/swiglu_op.py#L462
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
}

impl EncoderBlock {
    pub fn new(config: &AMPLIFYConfig, vb: VarBuilder, layer: i32) -> Result<Self> {
        let d_head = config.hidden_size / config.num_attention_heads;
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
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        freqs_cis: &Tensor,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed = self.attention_norm.forward(x)?;

        // Todo: confirm the attention block
        let (attn, contacts) =
            self.attention_block(&normed, pad_mask, freqs_cis, output_attentions)?;

        // add encoded bits and the self-attention...
        let x = x.add(&attn)?;

        // FFN add ...
        let normed = self.ffn_norm.forward(&x)?;

        let ffn_output = self.ffn.forward(&normed)?;
        let ff = self.ffn_dropout.forward(&ffn_output, false); // Todo: pass in the Inference/Training bit

        // Todo: see if the apply or add can be done differently in idiomatic Candle
        let x = x.add(&ff)?;
        Ok((x, contacts))
    }

    fn att_block(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        freqs_cis: &Tensor,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Get dimensions
        let batch_size = x.dim(0)?;
        let seq_len = x.dim(1)?;
        // Query, Key, Value projections
        let xq = self.q.forward(x)?;
        let xk = self.k.forward(x)?;
        let xv = self.v.forward(x)?;
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
        // Apply rotary embeddings
        let (xq, xk) = apply_rotary_emb(&xq, &xk, freqs_cis)?;
        // Attention computation
        let dropout_prob = if self.training {
            self.config.dropout_prob
        } else {
            0.0
        };
        let attn = memory_efficient_attention(&xq, &xk, &xv, pad_mask, dropout_prob)?;
        // Optional attention matrix computation for output
        let _attn = if output_attentions {
            let xq_t = xq.permute((0, 2, 1, 3))?;
            let xk_t = xk.permute((0, 2, 3, 1))?;
            let mut attn_weights = xq_t.matmul(&xk_t)?;
            let scale = (xq.dim(D::Minus1)? as f64).sqrt();
            attn_weights = attn_weights.div_scalar(scale)?;
            if let Some(mask) = pad_mask {
                attn_weights = attn_weights.add(mask)?;
            }
            Some(attn_weights.softmax(D::Minus1)?)
        } else {
            None
        };

        // Final projection and dropout
        let output = attn.reshape((
            batch_size,
            seq_len,
            self.config.num_attention_heads * self.d_head,
        ))?;
        let output = self.wo.forward(&output)?;
        let output = self.resid_dropout.forward(&output, false);
        Ok((output, _attn))
    }

    pub fn load(vb: VarBuilder, cfg: &AMPLIFYConfig, layer: i32) -> Result<Self> {
        // To keep the number of parameters and the amount of computation constant, we reduce the number of
        // hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
        // avoid RuntimeError due to misaligned operand
        let multiple_of = 8;
        let intermediate_size = (cfg.intermediate_size * 2) / 3;
        let intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);
        let vb = vb.pp(layer); // handle the layer nubmer here.
        let q = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q"))?;
        let k = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("k"))?;
        let v = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("v"))?;
        let wo = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("wo"))?;
        let w12 = linear_no_bias(cfg.hidden_size, intermediate_size * 2, vb.pp("ffn.w12"))?;
        let w3 = linear_no_bias(intermediate_size, cfg.hidden_size, vb.pp("ffn.w3"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        let attention_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("attention_norm"))?;

        Ok(Self {
            q,
            k,
            v,
            wo,
            resid_dropout: Dropout::new(cfg.dropout_prob as f32),
            w12,
            w3,
            attention_norm,
            ffn_norm,
            ffn_dropout: Dropout::new(cfg.dropout_prob as f32),
        })
    }
}

/// The AMPLIFY model
///
/// - [GH PythonModel](https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/src/amplify/model/amplify.py)
/// - [paper](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
/// - [HF](https://huggingface.co/chandar-lab/AMPLIFY_120M)
///
pub struct AMPLIFY {
    encoder: Embedding,
    transformer_encoder: Vec<EncoderBlock>,
    layer_norm_2: RmsNorm,
    decoder: Linear,
    freqs_cis: Tensor,
}

impl AMPLIFY {
    pub fn new(config: &AMPLIFYConfig, vb: VarBuilder) -> Result<Self> {
        unimplemented!()
    }

    fn process_attention_mask(
        &self,
        pad_mask: Option<&Tensor>,
        num_attention_heads: i64,
    ) -> Result<Option<Tensor>> {
        let Some(mask) = pad_mask else {
            return Ok(None);
        };
        if mask.sum_all()?.to_scalar::<f32>()? == 0.0 {
            return Ok(None);
        }
        let batch_size = mask.dim(0)? as usize;
        let seq_length = mask.dim(D::Minus1)? as usize;
        let num_heads = num_attention_heads as usize;
        let expanded_mask = mask
            .unsqueeze(1)? // Add head dimension
            .unsqueeze(1)? // Add query dimension
            .expand((batch_size, num_heads, seq_length, seq_length))?;
        Ok(Some(expanded_mask))
    }

    pub fn forward(
        &self,
        src: &Tensor,
        pad_mask: Option<&Tensor>,
        output_hidden_states: bool,
        output_attentions: bool,
    ) -> Result<ModelOutput> {
        let mut hidden_states = vec![];
        let mut attentions = vec![];

        // Process attention mask if provided
        let attention_mask =
            self.process_attention_mask(pad_mask, self.transformer_encoder.len() as i64);

        // Get appropriate length of freqs_cis
        let freqs_cis = self.freqs_cis.narrow(0, 0, src.dim(1)?)?;

        // Embedding layer
        let mut x = self.encoder.forward(src)?;

        // Transform through encoder blocks
        for layer in self.transformer_encoder.iter() {
            let (new_x, attn) =
                layer.forward(&x, attention_mask.as_ref(), &freqs_cis, output_attentions)?;
            x = new_x;

            if output_hidden_states {
                hidden_states.push(x.clone());
            }
            if output_attentions {
                if let Some(attn) = attn {
                    attentions.push(attn);
                }
            }
        }

        // Final layer norm and decoder
        let logits = if self.config.layer_norm_before_last_layer {
            self.decoder.forward(&self.layer_norm_2.forward(&x)?)?
        } else {
            self.decoder.forward(&x)?
        };

        Ok(ModelOutput {
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
        // process the transformer section
        let mut transformer_encoder = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            transformer_encoder.push(EncoderBlock::load(
                vb.pp("transformer_encoder"),
                cfg,
                i as i32,
            )?);
        }
        let encoder = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("encoder"))?;
        let layer_norm_2 = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("layer_norm_2"))?;
        let decoder = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("decoder"))?;

        // Todo: Double check this....
        let freqs_cis = Tensor::zeros(
            (cfg.max_length, cfg.num_attention_heads, 2),
            DType::F32,
            &Device::Cpu,
        )?;

        Ok(Self {
            encoder,
            transformer_encoder,
            layer_norm_2,
            decoder,
            freqs_cis,
        })
    }
}

// Helper structs and enums
#[derive(Debug)]
pub struct ModelOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}
