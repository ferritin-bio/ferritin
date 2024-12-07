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
use super::encoder::EncoderBlock;
use super::rotary::precompute_freqs_cis;
use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
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
        let batch_size = mask.dim(0)?;
        let seq_length = mask.dim(D::Minus1)?;
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
        let attention_mask =
            self.process_attention_mask(pad_mask, self.transformer_encoder.len() as i64)?;
        let freqs_cis = self.freqs_cis.narrow(0, 0, src.dim(1)?)?;
        let mut x = self.encoder.forward(src)?.contiguous()?;
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
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let freqs_cis = precompute_freqs_cis(head_dim, cfg.max_length)?.to_device(vb.device())?;

        Ok(Self {
            encoder,
            transformer_encoder,
            layer_norm_2,
            decoder,
            freqs_cis,
            config: cfg.clone(),
        })
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

// Helper structs and enums
#[derive(Debug)]
/// Amplify Model Output
///
/// logits, hidden states, and attentions.
///
///  logits -> distribution of the sequences.
///  attentions -> contact map
pub struct ModelOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

impl ModelOutput {
    /// "Perform average product correct, used for contact prediction."
    /// https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/examples/utils.py#L83
    /// "Perform average product correct, used for contact prediction."
    fn apc(&self, x: &Tensor) -> Result<Tensor> {
        let a1 = x.sum_keepdim(D::Minus1)?;
        let a2 = x.sum_keepdim(D::Minus2)?;
        let a12 = x.sum_keepdim((D::Minus1, D::Minus2))?;
        let avg = a1.matmul(&a2)?;
        // Divide by a12 (equivalent to pytorch's div_)
        // println!("IN the APC: avg, a12 {:?}, {:?}", avg, a12);
        // let avg = avg.div(&a12)?;
        let a12_broadcast = a12.broadcast_as(avg.shape())?;
        let avg = avg.div(&a12_broadcast)?;
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
        let (_1, _n_head, _seq_length, seq_length) = attentions.first().unwrap().dims4()?;
        let last_dim = seq_length;
        let attn_stacked = Tensor::stack(attentions, 0)?;
        let total_elements = attn_stacked.dims().iter().product::<usize>();
        let first_dim = total_elements / (last_dim * last_dim);
        let attn_map_combined2 = attn_stacked.reshape(&[first_dim, last_dim, last_dim])?;

        // In PyTorch: attn_map = attn_map[:, 1:-1, 1:-1]
        let attn_map_combined2 = attn_map_combined2
            .narrow(1, 1, attn_map_combined2.dim(1)? - 2)? // second dim
            .narrow(2, 1, attn_map_combined2.dim(2)? - 2)?; // third dim
        let symmetric = self.symmetrize(&attn_map_combined2)?;
        let normalized = self.apc(&symmetric)?;
        let proximity_map = normalized.permute((1, 2, 0))?; //  # (residues, residues, map)

        Ok(Some(proximity_map))
    }
}
