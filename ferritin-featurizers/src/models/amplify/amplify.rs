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

// EncoderBlock implementation
//
// example 01: T5: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/t5.rs#L331
//
// Example 01: FFN: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/distilbert.rs#L198
// ffn: FeedForward,
//
// // Clean Implementation
//
// Example: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/glm4.rs#L340
// SwiGLu Implementation:  https://github.com/facebookresearch/xformers/blob/main/xformers/ops/swiglu_op.py#L462
//
pub struct EncoderBlock {
    q: Linear,
    k: Linear,
    v: Linear,
    wo: Linear,
    // resid_dropout: Dropout,
    w12: Linear,
    w3: Linear,
    attention_norm: candle_nn::RmsNorm, // <----- Check These
    ffn_norm: candle_nn::RmsNorm,
    // ffn_dropout: Dropout,
}

impl EncoderBlock {
    pub fn new(config: &AMPLIFYConfig, vb: VarBuilder) -> Result<Self> {
        // let d_head = config.hidden_size / config.num_attention_heads;
        // // Attention layers
        // let q = Linear::new(
        //     vb.pp("q").get_with_hints(
        //         (config.hidden_size, config.hidden_size),
        //         "weight",
        //         candle_nn::init::ZERO,
        //     )?,
        //     if config.att_bias {
        //         Some(vb.pp("q").get_with_hints(
        //             config.hidden_size,
        //             "bias",
        //             candle_nn::init::ZERO,
        //         )?)
        //     } else {
        //         None
        //     },
        // );
        // // Similar initialization for k, v, and wo...
        // // FFN initialization based on activation type
        // let multiple_of = 8;
        // let intermediate_size =-
        //     (2 * config.intermediate_size / 3).div_ceil(multiple_of) * multiple_of;
        // let ffn = FeedForward::load(config, vb);
        // FFN::SwiGLU(SwiGLUFFN::new(
        //     config.hidden_size,
        //     intermediate_size,
        //     config.hidden_size,
        //     config.ffn_bias,
        //     vb.pp("ffn"),
        // )?)

        // let attention_norm: Box<dyn Module> = if config.rms_norm {
        //     Box::new(RMSNorm::new(
        //         config.hidden_size,
        //         config.norm_eps,
        //         vb.pp("attention_norm"),
        //     )?)
        // } else {
        //     Box::new(LayerNorm::new(
        //         config.hidden_size,
        //         config.norm_eps,
        //         vb.pp("attention_norm"),
        //     )?)
        // };

        // // Similar for ffn_norm...

        // Ok(Self {
        //     config: config.clone(),
        //     q,
        //     k,
        //     v,
        //     wo,
        //     resid_dropout: Dropout::new(config.dropout_prob),
        //     ffn,
        //     attention_norm,
        //     ffn_norm,
        //     ffn_dropout: Dropout::new(config.dropout_prob),
        // })
        unimplemented!();
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

        // names
        let q_name = format!("{}.q", &layer);
        let k_name = format!("{}.k", &layer);
        let v_name = format!("{}.v", &layer);
        let wo_name = format!("{}.wo", &layer);
        let ffn_w12_name = format!("{}.ffn.w12", &layer);
        let ffn_w3_name = format!("{}.ffn.w3", &layer);
        let ffn_norm_name = format!("{}.ffn_norm", &layer);
        let attention_norm_name = format!("{}.attention_norm", &layer);

        // layers
        let q = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(q_name))?;
        let k = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(k_name))?;
        let v = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(v_name))?;
        let wo = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp(wo_name))?;
        let w12 = linear_no_bias(intermediate_size * 2, cfg.hidden_size, vb.pp(ffn_w12_name))?;
        let w3 = linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp(ffn_w3_name))?;
        let ffn_norm = rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp(ffn_norm_name.as_str()))?;
        let attention_norm = rms_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            vb.pp(attention_norm_name.as_str()),
        )?;

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
        unimplemented!()
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
