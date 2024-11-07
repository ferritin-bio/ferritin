//! AMPLIFY is a optimized transformer model focused on optimizing the context of sequence models
//! while maintaining computational efficiency.
//!
//! Key features:
//! - Rotary positional embeddings
//! - RMSNorm for improved training stability
//! - SwiGLU activation function
//! - Specialized architecture optimizations
//! - Memory efficient inference

use candle_core::{DType, Device, Result, Tensor};
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
// example 01: T5: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/t5.rs#L331
//
// Example 01: FFN: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/distilbert.rs#L198
// ffn: FeedForward,
// // Clean Implementation
// Example: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/glm4.rs#L340
// SwiGLu Implementation:  https://github.com/facebookresearch/xformers/blob/main/xformers/ops/swiglu_op.py#L462
//
pub struct EncoderBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    wo: Linear,
    w12: Linear,
    w3: Linear,
    attention_norm: RmsNorm, // <----- Check These
    ffn_norm: RmsNorm,
    // resid_dropout: Dropout,
    // ffn_dropout: Dropout,
}

impl EncoderBlock {
    pub fn new(config: &AMPLIFYConfig, vb: VarBuilder) -> Result<Self> {
        let d_head = config.hidden_size / config.num_attention_heads;

        let multiple_of = 8;
        let multiple_of = 8;
        let intermediate_size = (cfg.intermediate_size * 2) / 3;
        let intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);
        let q = linear(config.hidden_size, config.hidden_size, vb.pp("q"))?;
        let k = linear(config.hidden_size, config.hidden_size, vb.pp("k"))?;
        let v = linear(config.hidden_size, config.hidden_size, vb.pp("v"))?;
        let wo = linear(config.hidden_size, config.hidden_size, vb.pp("wo"))?;
        let w12 = linear_no_bias(intermediate_size * 2, config.hidden_size, vb.pp("ffn.w12"))?;
        let w3 = linear_no_bias(config.hidden_size, intermediate_size, vb.pp("ffn.w3"))?;
        let [ffn_norm, attention_norm] = [
            rms_norm(config.hidden_size, config.norm_eps, vb.pp("ffn_norm"))?,
            rms_norm(config.hidden_size, config.norm_eps, vb.pp("attention_norm"))?,
        ];

        Ok(Self {
            q,
            k,
            v,
            wo,
            w12,
            w3,
            attention_norm,
            ffn_norm,
            // ffn_dropout: Dropout::new(config.dropout_prob),
            // resid_dropout: Dropout::new(config.dropout_prob),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        freqs_cis: &Tensor,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // let normed = self.attention_norm.forward(x)?;
        // let (attn, contacts) =
        //     self.attention_block(&normed, pad_mask, freqs_cis, output_attentions)?;
        // let x = x.add(&attn)?;
        // let normed = self.ffn_norm.forward(&x)?;
        // let ff = self.ff_block(&normed)?;
        // let x = x.add(&ff)?;
        // Ok((x, contacts))
        unimplemented!()
    }

    fn attention_block(
        &self,
        x: &Tensor,
        pad_mask: Option<&Tensor>,
        freqs_cis: &Tensor,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        unimplemented!()
    }

    pub fn load(vb: VarBuilder, cfg: &AMPLIFYConfig, layer: i32) -> Result<Self> {
        // To keep the number of parameters and the amount of computation constant, we reduce the number of
        // hidden units by a factor of 2/3 (https://arxiv.org/pdf/2002.05202.pdf) and make it a multiple of 8 to
        // avoid RuntimeError due to misaligned operand
        let multiple_of = 8;
        let intermediate_size = (cfg.intermediate_size * 2) / 3;
        let intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);

        #[rustfmt::skip]
        let names = ["q", "k", "v", "wo", "ffn.w12", "ffn.w3", "ffn_norm", "attention_norm"]
            .map(|suffix| format!("{}.{}", layer, suffix));

        // Linear layers
        let [q, k, v, wo, w12, w3] = [
            linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(&names[0]))?,
            linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(&names[1]))?,
            linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(&names[2]))?,
            linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(&names[3]))?,
            linear_no_bias(intermediate_size * 2, cfg.hidden_size, vb.pp(&names[4]))?,
            linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp(&names[5]))?,
        ];

        // Norm layers
        let [ffn_norm, attention_norm] = [
            rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp(&names[6]))?,
            rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp(&names[7]))?,
        ];

        Ok(Self {
            q,
            k,
            v,
            wo,
            w12,
            w3,
            attention_norm,
            ffn_norm,
            // resid_dropout,
            // ffn_dropout,
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
        let attention_mask = if let Some(mask) = pad_mask {
            if !mask.all_close(&mask.zeros_like()?, 1e-6, 1e-6)? {
                Some(mask.unsqueeze(1)?.unsqueeze(1)?.expand((
                    mask.dim(0)?,
                    self.config.num_attention_heads,
                    mask.dim(-1)?,
                    mask.dim(-1)?,
                ))?)
            } else {
                None
            }
        } else {
            None
        };

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
                hidden_states.push(x.clone()?);
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
