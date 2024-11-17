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
use super::rotary::{apply_rotary_emb, precompute_freqs_cis};
use super::tokenizer::ProteinTokenizer;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::{
    embedding, linear, linear_no_bias, ops::softmax_last_dim, rms_norm, Activation, Dropout,
    Embedding, Linear, RmsNorm, VarBuilder,
};

#[derive(Debug, Clone)]
/// Configuration Struct for AMPLIFY
///
/// Currently only holds the weight params for
/// those modeld found on GH: the 120M and 350M models.
///
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
        AMPLIFYConfig::amp_120m()
    }
}
impl AMPLIFYConfig {
    pub fn amp_120m() -> Self {
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
    pub fn amp_350m() -> Self {
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

/// Amplify EncoderBlock implementation
///
/// References for coding the block from similar models.
///
/// - [T5](https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/t5.rs#L331)
/// - [distilbert](https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/distilbert.rs#L198)
/// - [glm4](https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/glm4.rs#L340)
/// - [SwiGLu Imple](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/swiglu_op.py#L462)
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
        let xq = self.q.forward(x)?; // [batch_size, seq_len, hidden_size]
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

        let (xq, xk) = apply_rotary_emb(&xq, &xk, &freqs_cis)?;
        let dropout_prob = self.config.dropout_prob;

        // need to handle pad_mask better ....
        //
        let pad_mask = if let Some(mask) = pad_mask {
            let (batch_size, seq_len) = (x.dim(0)?, x.dim(1)?);
            let num_heads = self.config.num_attention_heads;

            // Following PyTorch's implementation:
            // 1. unsqueeze twice to add head dimensions
            // 2. repeat to match attention matrix size
            let mask = mask
                .unsqueeze(1)? // Add first head dimension
                .unsqueeze(1)? // Add second head dimension
                .expand((batch_size, num_heads, seq_len, seq_len))?; // Expand to full attention size
            Some(mask)
        } else {
            None
        };

        let attn = self.scaled_dot_product_attention(
            &xq.permute((0, 2, 1, 3))?,
            &xk.permute((0, 2, 1, 3))?,
            &xv.permute((0, 2, 1, 3))?,
            pad_mask.as_ref(),
            dropout_prob,
            false,
        )?;
        // `[batch, num_heads, seq_len, head_dim]` → `[batch, seq_len, num_heads, head_dim]`
        let attn = attn.permute((0, 2, 1, 3))?;
        let _attn = if output_attentions {
            let xq_t = xq.permute((0, 2, 1, 3))?;
            let xk_t = xk.permute((0, 2, 3, 1))?;
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
        // Process attention mask if provided
        let attention_mask =
            self.process_attention_mask(pad_mask, self.transformer_encoder.len() as i64)?;
        let freqs_cis = self.freqs_cis.narrow(0, 0, src.dim(1)?)?;
        // Embedding layer
        let mut x = self.encoder.forward(src)?;
        // Transform through encoder blocks
        // println!("AMPLIFY.forward():  running through the transformer");
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
        // println!("AMPLIFY.forward():  calculating logits");
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
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let freqs_cis = precompute_freqs_cis(head_dim, cfg.max_length)?;

        Ok(Self {
            encoder,
            transformer_encoder,
            layer_norm_2,
            decoder,
            freqs_cis,
            config: cfg.clone(),
        })
    }
    /// Retreive the model and make it available for usage.
    /// hardcode the 120M for the moment...
    pub fn load_from_huggingface() -> Result<(ProteinTokenizer, Self)> {
        let ampconfig = AMPLIFYConfig::amp_120m();
        let model_id = "chandar-lab/AMPLIFY_120M";
        let revision = "main";
        let api = Api::new().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));
        // Load and analyze the safetensors file
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &Device::Cpu)?
        };
        let config = AMPLIFYConfig::amp_120m();
        let model = AMPLIFY::load(vb, &config)?;
        let tokenizer = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let protein_tokenizer =
            ProteinTokenizer::new(tokenizer).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Ok((protein_tokenizer, model))
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
///
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
        // Sum along last dimension (keeping dims)
        let a1 = x.sum_keepdim(D::Minus1)?;
        // Sum along second-to-last dimension (keeping dims)
        let a2 = x.sum_keepdim(D::Minus2)?;
        // Sum along both last dimensions (keeping dims)
        let a12 = x.sum_keepdim((D::Minus1, D::Minus2))?;
        // Multiply a1 and a2
        let avg = a1.matmul(&a2)?;
        // Divide by a12 (equivalent to pytorch's div_)
        // println!("IN the APC: avg, a12 {:?}, {:?}", avg, a12);
        // let avg = avg.div(&a12)?;
        let a12_broadcast = a12.broadcast_as(avg.shape())?;
        // Divide by a12 (with proper broadcasting)
        let avg = avg.div(&a12_broadcast)?;
        // Subtract avg from x
        x.sub(&avg)
    }

    //From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
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
        // we need the dimentions to reshape below.
        // the attention blocks have the following shaep
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
