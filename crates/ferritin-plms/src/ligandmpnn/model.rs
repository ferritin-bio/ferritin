//! A message passing protein design neural network
//! that samples sequences diffusing conditional probabilities.
//!
//! - See the [LigandMPNN Repo](https://github.com/dauparas/LigandMPNN)
//!
use std::collections::HashMap;

use super::configs::{ModelTypes, ProteinMPNNConfig};
use super::proteinfeatures::ProteinFeatures;
use super::proteinfeaturesmodel::ProteinFeaturesModel;
use super::utilities::{cat_neighbors_nodes, gather_nodes, int_to_aa1};
use candle_core::safetensors;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::encoding::one_hot;
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::{embedding, layer_norm, linear, Dropout, Embedding, LayerNorm, Linear, VarBuilder};
use candle_transformers::generation::LogitsProcessor;

// refactoring common fn
fn concat_node_tensors(h_v: &Tensor, h_e: &Tensor, e_idx: &Tensor) -> Result<Tensor> {
    let h_ev = cat_neighbors_nodes(&h_v, h_e, e_idx)?;
    let h_v_expand = h_v.unsqueeze(D::Minus2)?;
    let expand_shape = [
        h_ev.dims()[0],
        h_ev.dims()[1],
        h_ev.dims()[2],
        h_v_expand.dims()[3],
    ];
    let h_v_expand = h_v_expand.expand(&expand_shape)?.to_dtype(h_ev.dtype())?;
    Tensor::cat(&[&h_v_expand, &h_ev], D::Minus1)?.contiguous()
}
// refactoring common fn
fn apply_dropout_and_norm(
    input: &Tensor,
    delta: &Tensor,
    dropout: &Dropout,
    norm: &LayerNorm,
    training: bool,
) -> Result<Tensor> {
    let delta_dropout = dropout.forward(delta, training)?;
    norm.forward(&(input + delta_dropout)?)
}

pub fn multinomial_sample(probs: &Tensor, temperature: f64, seed: u64) -> Result<Tensor> {
    let mut logits_processor = LogitsProcessor::new(
        seed,              // seed for reproducibility
        Some(temperature), // temperature scaling
        // None,              // top_p (nucleus sampling), we don't need this
        Some(0.95), // top_p (nucleus sampling), we don't need this
    );
    let idx = logits_processor.sample(probs)?;
    // println!("Selected index: {}", idx);
    if idx >= 21 {
        println!("WARNING: Invalid index {} selected", idx);
    }
    Tensor::new(&[idx], probs.device())
}

// Primary Return Object from the ProtMPNN Model
#[derive(Clone, Debug)]
pub struct ScoreOutput {
    // Sequence
    pub(crate) s: Tensor,
    pub(crate) log_probs: Tensor,
    pub(crate) logits: Tensor,
    pub(crate) decoding_order: Tensor,
}
impl ScoreOutput {
    // S dims are [Batch, seqlength]
    pub fn get_sequences(&self) -> Result<Vec<String>> {
        let (b, l) = self.s.dims2()?;
        let mut sequences = Vec::with_capacity(b);
        for batch_idx in 0..b {
            let mut sequence = String::with_capacity(l);
            for pos in 0..l {
                let aa_idx = self.s.get(batch_idx)?.get(pos)?.to_vec0::<u32>()?;
                // println!("Position {}, Raw index: {}", pos, aa_idx);
                let aa = int_to_aa1(aa_idx);
                // println!("Converted to: {}", aa);
                sequence.push(aa);
            }
            sequences.push(sequence);
        }
        Ok(sequences)
    }
    pub fn get_decoding_order(&self) -> Result<Vec<u32>> {
        let values = self.decoding_order.flatten_all()?.to_vec1::<u32>()?;
        Ok(values)
    }
    pub fn get_log_probs(&self) -> &Tensor {
        &self.log_probs
    }
    pub fn save_as_safetensors(&self, filename: String) -> Result<()> {
        let mut tensors = HashMap::new();
        tensors.insert("S".to_string(), self.s.clone());
        tensors.insert("log_probs".to_string(), self.log_probs.clone());
        tensors.insert("logits".to_string(), self.logits.clone());
        tensors.insert("decoding_order".to_string(), self.decoding_order.clone());

        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&filename).parent() {
            std::fs::create_dir_all(parent)?;
        }

        safetensors::save(&tensors, &filename);
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct PositionWiseFeedForward {
    w1: Linear,
    w2: Linear,
}

impl PositionWiseFeedForward {
    fn new(vb: VarBuilder, dim_input: usize, dim_feedforward: usize) -> Result<Self> {
        let w1 = linear::linear(dim_input, dim_feedforward, vb.pp("W_in"))?;
        let w2 = linear::linear(dim_feedforward, dim_input, vb.pp("W_out"))?;
        Ok(Self { w1, w2 })
    }
}

impl Module for PositionWiseFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w1.forward(x)?;
        let x = x.gelu()?;
        self.w2.forward(&x)
    }
}

#[derive(Clone, Debug)]
pub struct EncLayer {
    num_hidden: usize,
    num_in: usize,
    scale: f64,
    dropout1: Dropout,
    dropout2: Dropout,
    dropout3: Dropout,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    w11: Linear,
    w12: Linear,
    w13: Linear,
    dense: PositionWiseFeedForward,
}

impl EncLayer {
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig, layer: i32) -> Result<Self> {
        let vb = vb.pp(layer); // handle the layer number here.
        let num_hidden = config.hidden_dim as usize;
        let augment_eps = config.augment_eps as f64;
        let num_in = (config.hidden_dim * 2) as usize;
        let dropout_ratio = config.dropout_ratio;
        let norm1 = layer_norm::layer_norm(num_hidden, augment_eps, vb.pp("norm1"))?;
        let norm2 = layer_norm::layer_norm(num_hidden, augment_eps, vb.pp("norm2"))?;
        let norm3 = layer_norm::layer_norm(num_hidden, augment_eps, vb.pp("norm3"))?;

        let w1 = linear::linear(num_hidden + num_in, num_hidden, vb.pp("W1"))?;
        let w2 = linear::linear(num_hidden, num_hidden, vb.pp("W2"))?;
        let w3 = linear::linear(num_hidden, num_hidden, vb.pp("W3"))?;
        let w11 = linear::linear(num_hidden + num_in, num_hidden, vb.pp("W11"))?;
        let w12 = linear::linear(num_hidden, num_hidden, vb.pp("W12"))?;
        let w13 = linear::linear(num_hidden, num_hidden, vb.pp("W13"))?;

        let dropout1 = Dropout::new(dropout_ratio);
        let dropout2 = Dropout::new(dropout_ratio);
        let dropout3 = Dropout::new(dropout_ratio);

        // note in the pytorch code they add the GELU activation here.
        let dense = PositionWiseFeedForward::new(vb.pp("dense"), num_hidden, num_hidden * 4)?;

        Ok(Self {
            num_hidden,
            num_in,
            scale: config.scale_factor,
            dropout1,
            dropout2,
            dropout3,
            norm1,
            norm2,
            norm3,
            w1,
            w2,
            w3,
            w11,
            w12,
            w13,
            dense,
        })
    }
    fn forward(
        &self,
        h_v: &Tensor,
        h_e: &Tensor,
        e_idx: &Tensor,
        mask_v: Option<&Tensor>,
        mask_attend: Option<&Tensor>,
        training: Option<bool>,
    ) -> Result<(Tensor, Tensor)> {
        let training = training.unwrap_or(false);
        let h_v = h_v.to_dtype(DType::F32)?;
        let h_ev = concat_node_tensors(&h_v, h_e, e_idx)?;
        let h_message = self
            .w1
            .forward(&h_ev)?
            .gelu()?
            .apply(&self.w2)?
            .gelu()?
            .apply(&self.w3)?;

        let h_message = if let Some(mask) = mask_attend {
            mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_message)?
        } else {
            h_message
        };
        // Safe division with scale
        let sum = h_message.sum(D::Minus2)?;
        let scale = if self.scale == 0.0 { 1.0 } else { self.scale };
        let dh = (sum / scale)?;
        let h_v = apply_dropout_and_norm(&h_v, &dh, &self.dropout1, &self.norm1, training)?;
        let dense_output = self.dense.forward(&h_v)?;
        let h_v =
            apply_dropout_and_norm(&h_v, &dense_output, &self.dropout2, &self.norm2, training)?;
        let h_v = if let Some(mask) = mask_v {
            mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_v)?
        } else {
            h_v
        };

        let h_ev = concat_node_tensors(&h_v, h_e, e_idx)?;
        let h_message = self
            .w11
            .forward(&h_ev)?
            .gelu()?
            .apply(&self.w12)?
            .gelu()?
            .apply(&self.w13)?;
        let h_e = apply_dropout_and_norm(h_e, &h_message, &self.dropout3, &self.norm3, training)?;
        Ok((h_v, h_e))
    }
}

#[derive(Clone, Debug)]
pub struct DecLayer {
    num_hidden: usize,
    num_in: usize,
    scale: f64,
    dropout1: Dropout,
    dropout2: Dropout,
    norm1: LayerNorm,
    norm2: LayerNorm,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    dense: PositionWiseFeedForward,
}

impl DecLayer {
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig, layer: i32) -> Result<Self> {
        let vb = vb.pp(layer); // handle the layer number here.
        let num_hidden = config.hidden_dim as usize;
        let augment_eps = config.augment_eps as f64;
        let num_in = (config.hidden_dim * 3) as usize;
        let dropout_ratio = config.dropout_ratio;

        let norm1 = layer_norm::layer_norm(num_hidden, augment_eps, vb.pp("norm1"))?;
        let norm2 = layer_norm::layer_norm(num_hidden, augment_eps, vb.pp("norm2"))?;

        let w1 = linear::linear(num_hidden + num_in, num_hidden, vb.pp("W1"))?;
        let w2 = linear::linear(num_hidden, num_hidden, vb.pp("W2"))?;
        let w3 = linear::linear(num_hidden, num_hidden, vb.pp("W3"))?;
        let dropout1 = Dropout::new(dropout_ratio);
        let dropout2 = Dropout::new(dropout_ratio);

        let dense = PositionWiseFeedForward::new(vb.pp("dense"), num_hidden, num_hidden * 4)?;

        Ok(Self {
            num_hidden,
            num_in,
            scale: config.scale_factor,
            dropout1,
            dropout2,
            norm1,
            norm2,
            w1,
            w2,
            w3,
            dense,
        })
    }
    pub fn forward(
        &self,
        h_v: &Tensor,
        h_e: &Tensor,
        mask_v: Option<&Tensor>,
        mask_attend: Option<&Tensor>,
        training: Option<bool>,
    ) -> Result<Tensor> {
        let training_bool = training.unwrap_or(false);

        let expand_shape = [
            h_e.dims()[0], // batch (1)
            h_e.dims()[1], // sequence length (93)
            h_e.dims()[2], // number of neighbors (24)
            h_v.dims()[2], // keep original hidden dim (128)
        ];
        let h_v_expand = h_v.unsqueeze(D::Minus2)?.expand(&expand_shape)?;
        let h_ev = Tensor::cat(&[&h_v_expand, h_e], D::Minus1)?.contiguous()?;

        let h_message = self
            .w1
            .forward(&h_ev)?
            .gelu()?
            .apply(&self.w2)?
            .gelu()?
            .apply(&self.w3)?;

        let h_message = self.dropout1.forward(&h_message, training_bool)?;
        let h_message = if let Some(mask) = mask_attend {
            mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_message)?
        } else {
            h_message
        };
        let dh = (h_message.sum(D::Minus2)? / self.scale)?;
        let h_v = self.norm1.forward(&(h_v + dh)?)?;
        let dh = self.dense.forward(&h_v)?;
        let dh_dropout = self.dropout2.forward(&dh, training_bool)?;
        let h_v = self.norm2.forward(&(h_v + dh_dropout)?)?;
        let h_v = if let Some(mask) = mask_v {
            mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_v)?
        } else {
            h_v
        };
        Ok(h_v)
    }
}

/// ProteinMPNN Model
/// - [link](https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L10C7-L10C18)
pub struct ProteinMPNN {
    pub(crate) config: ProteinMPNNConfig,
    pub(crate) decoder_layers: Vec<DecLayer>,
    pub(crate) device: Device,
    pub(crate) encoder_layers: Vec<EncLayer>,
    pub(crate) features: ProteinFeaturesModel,
    pub(crate) w_e: Linear,
    pub(crate) w_out: Linear,
    pub(crate) w_s: Embedding,
}

impl ProteinMPNN {
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig) -> Result<Self> {
        let hidden_dim = config.hidden_dim as usize;
        let edge_features = config.edge_features as usize;
        let num_letters = config.num_letters as usize;
        let vocab_size = config.vocab as usize;

        // Create encoder and decoder layers using iterators
        let encoder_layers = (0..config.num_encoder_layers)
            .map(|i| EncLayer::load(vb.pp("encoder_layers"), config, i as i32))
            .collect::<Result<Vec<_>>>()?;

        let decoder_layers = (0..config.num_decoder_layers)
            .map(|i| DecLayer::load(vb.pp("decoder_layers"), config, i as i32))
            .collect::<Result<Vec<_>>>()?;

        // Initialize weights
        let w_e = linear::linear(edge_features, hidden_dim, vb.pp("W_e"))?;
        let w_out = linear::linear(hidden_dim, num_letters, vb.pp("W_out"))?;
        let w_s = embedding(vocab_size, hidden_dim, vb.pp("W_s"))?;
        // Features
        let features = ProteinFeaturesModel::load(vb.pp("features"), config.clone())?;

        Ok(Self {
            config: config.clone(), // todo: check the\is clone later...
            decoder_layers,
            device: vb.device().clone(),
            encoder_layers,
            features,
            w_e,
            w_out,
            w_s,
        })
    }
    fn encode(&self, features: &ProteinFeatures) -> Result<(Tensor, Tensor, Tensor)> {
        let s_true = &features.get_sequence();
        let base_dtype = DType::F32;
        let mask = match features.get_sequence_mask() {
            Some(m) => m,
            None => &Tensor::ones_like(&s_true)?,
        };
        match self.config.model_type {
            ModelTypes::ProteinMPNN => {
                let (e, e_idx) = self.features.forward(features, &self.device)?;
                let mut h_v = Tensor::zeros(
                    (e.dim(0)?, e.dim(1)?, e.dim(D::Minus1)?),
                    base_dtype,
                    &self.device,
                )?;
                let mut h_e = self.w_e.forward(&e)?;
                let mask_attend = if let Some(mask) = features.get_sequence_mask() {
                    let mask_expanded = mask.unsqueeze(D::Minus1)?; // [B, L, 1]
                    let mask_gathered = gather_nodes(&mask_expanded, &e_idx)?;
                    let mask_gathered = mask_gathered.squeeze(D::Minus1)?;
                    let mask_attend = {
                        let mask_unsqueezed = mask.unsqueeze(D::Minus1)?; // [B, L, 1]
                        let mask_expanded = mask_unsqueezed.expand((
                            mask_gathered.dim(0)?, // batch
                            mask_gathered.dim(1)?, // sequence length
                            mask_gathered.dim(2)?, // number of neighbors
                        ))?;
                        mask_expanded.mul(&mask_gathered)?
                    };
                    mask_attend
                } else {
                    let (b, l) = mask.dims2()?;
                    let ones = Tensor::ones((b, l, e_idx.dim(2)?), DType::F32, &self.device)?;
                    ones
                };
                println!("Beginning the Encoding...");
                for (_, layer) in self.encoder_layers.iter().enumerate() {
                    let (new_h_v, new_h_e) = layer.forward(
                        &h_v,
                        &h_e,
                        &e_idx,
                        Some(&mask),
                        Some(&mask_attend),
                        Some(false),
                    )?;
                    h_v = new_h_v;
                    h_e = new_h_e;
                }
                Ok((h_v, h_e, e_idx))
            }
            ModelTypes::LigandMPNN => {
                todo!()
                //     let (v, e, e_idx, y_nodes, y_edges, y_m) = self.features.forward(feature_dict)?;
                //     let mut h_v = Tensor::zeros((e.dim(0)?, e.dim(1)?, e.dim(-1)?), device)?;
                //     let mut h_e = self.w_e.forward(&e)?;
                //     let h_e_context = self.w_v.forward(&v)?;
                //     let mask_attend = gather_nodes(&mask.unsqueeze(-1)?, &e_idx)?.squeeze(-1)?;
                //     let mask_attend = mask.unsqueeze(-1)? * &mask_attend;
                //
                //     for layer in &self.encoder_layers {
                //         let (new_h_v, new_h_e) =
                //             layer.forward(&h_v, &h_e, &e_idx, &mask, &mask_attend)?;
                //         h_v = new_h_v;
                //         h_e = new_h_e;
                //     }
                //
                //     let mut h_v_c = self.w_c.forward(&h_v)?;
                //     let y_m_edges = &y_m.unsqueeze(-1)? * &y_m.unsqueeze(-2)?;
                //     let mut y_nodes = self.w_nodes_y.forward(&y_nodes)?;
                //     let y_edges = self.w_edges_y.forward(&y_edges)?;
                //
                //     for (y_layer, c_layer) in self
                //         .y_context_encoder_layers
                //         .iter()
                //         .zip(&self.context_encoder_layers)
                //     {
                //         y_nodes = y_layer.forward(&y_nodes, &y_edges, &y_m, &y_m_edges)?;
                //         let h_e_context_cat = Tensor::cat(&[&h_e_context, &y_nodes], -1)?;
                //         h_v_c = c_layer.forward(&h_v_c, &h_e_context_cat, &mask, &y_m)?;
                //     }
                //     h_v_c = self.v_c.forward(&h_v_c)?;
                //     h_v = &h_v + &self.v_c_norm.forward(&self.dropout.forward(&h_v_c)?)?;
                //     Ok((h_v, h_e, e_idx))
            }
        }
    }
    pub fn sample(
        &self,
        features: &ProteinFeatures,
        temperature: f64,
        seed: u64,
    ) -> Result<ScoreOutput> {
        let sample_dtype = DType::F32;
        let ProteinFeatures {
            s,
            x_mask,
            // symmetry_residues,
            // symmetry_weights,
            ..
        } = features;
        let s_true = s.to_dtype(sample_dtype)?;
        let device = s.device();
        let (b, l) = s.dims2()?;
        // Todo: This is a hack. we should be passing in encoded chains.
        // let chain_mask = Tensor::ones_like(&x_mask.as_ref().unwrap())?.to_dtype(sample_dtype)?;
        // let chain_mask = x_mask.as_ref().unwrap().mul(&chain_mask)?;
        let chain_mask = x_mask.as_ref().unwrap().to_dtype(sample_dtype)?;
        let (h_v, h_e, e_idx) = self.encode(features)?;
        let rand_tensor = Tensor::randn(0f32, 0.25f32, (b, l), device)?.to_dtype(sample_dtype)?;
        let decoding_order = (&chain_mask + 0.0001)?
            .mul(&rand_tensor.abs()?)?
            .arg_sort_last_dim(false)?;
        // Todo add  bias
        // # [B,L,21] - amino acid bias per position
        //  bias = feature_dict["bias"]
        let bias = Tensor::ones((b, l, 21), sample_dtype, device)?;
        println!("todo: We need to add the bias!");
        let symmetry_residues: Option<Vec<i32>> = None;
        match symmetry_residues {
            None => {
                let e_idx = e_idx.repeat(&[b, 1, 1])?;
                let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1f32, 0f32)?
                    .to_dtype(sample_dtype)?
                    .contiguous()?;
                let tril = Tensor::tril2(l, sample_dtype, device)?;
                let tril = tril.unsqueeze(0)?;
                let temp = tril
                    .matmul(&permutation_matrix_reverse.transpose(1, 2)?)?
                    .contiguous()?; //tensor of shape (b, i, q)
                let order_mask_backward = temp
                    .matmul(&permutation_matrix_reverse.transpose(1, 2)?)?
                    .contiguous()?; // This will give us a tensor of shape (b, q, p)
                let mask_attend = order_mask_backward
                    .gather(&e_idx, 2)?
                    .unsqueeze(D::Minus1)?;
                let mask_1d = x_mask.as_ref().unwrap().reshape((b, l, 1, 1))?;
                // Broadcast mask_1d to match mask_attend's shape
                let mask_1d = mask_1d
                    .broadcast_as(mask_attend.shape())?
                    .to_dtype(sample_dtype)?;
                let mask_bw = mask_1d.mul(&mask_attend)?;
                let mask_fw = mask_1d.mul(&(Tensor::ones_like(&mask_attend)? - mask_attend)?)?;
                // Note: `sample` begins to diverge from the `score` here.
                // repeat for decoding
                let s_true = s_true.repeat((b, 1))?;
                let h_v = h_v.repeat((b, 1, 1))?;
                let h_e = h_e.repeat((b, 1, 1, 1))?;
                let mask = x_mask.as_ref().unwrap().repeat((b, 1))?.contiguous()?;
                let chain_mask = &chain_mask.repeat((b, 1))?;
                let bias = bias.repeat((b, 1, 1))?;
                let mut all_probs = Tensor::zeros((b, l, 20), sample_dtype, device)?;
                // why is this one 21 and the others are 20?
                let mut all_log_probs = Tensor::zeros((b, l, 21), sample_dtype, device)?;
                let mut h_s = Tensor::zeros_like(&h_v)?;
                // note: we this value of 20 is `X`. We will need to replace the values below, not add them
                let mut s = Tensor::full(20u32, (b, l), device)?;
                let mut h_v_stack = vec![h_v.clone()];

                for _ in 0..self.decoder_layers.len() {
                    let zeros = Tensor::zeros_like(&h_v)?;
                    h_v_stack.push(zeros);
                }
                let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
                let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;
                let mask_fw = mask_fw
                    .broadcast_as(h_exv_encoder.shape())?
                    .to_dtype(h_exv_encoder.dtype())?;
                let h_exv_encoder_fw = mask_fw.mul(&h_exv_encoder)?;
                for t_ in 0..l {
                    let t = decoding_order.i((.., t_))?;
                    let t_gather = t.unsqueeze(1)?; // Shape [B, 1]
                                                    // Gather masks and bias
                    let chain_mask_t = chain_mask.gather(&t_gather, 1)?.squeeze(1)?;
                    let mask_t = mask.gather(&t_gather, 1)?.squeeze(1)?.contiguous()?;
                    let bias_t = bias
                        .gather(&t_gather.unsqueeze(2)?.expand((b, 1, 21))?.contiguous()?, 1)?
                        .squeeze(1)?;
                    // Gather edge and node indices/features
                    let e_idx_t = e_idx
                        .gather(
                            &t_gather
                                .unsqueeze(2)?
                                .expand((b, 1, e_idx.dim(2)?))?
                                .contiguous()?,
                            1,
                        )?
                        .contiguous()?;
                    let h_e_t = h_e.gather(
                        &t_gather
                            .unsqueeze(2)?
                            .unsqueeze(3)?
                            .expand((b, 1, h_e.dim(2)?, h_e.dim(3)?))?
                            .contiguous()?,
                        1,
                    )?;
                    let n = e_idx_t.dim(2)?; // number of neighbors
                    let c = h_s.dim(2)?; // channels/features
                    let h_e_t = h_e_t
                        .squeeze(1)? // [B, N, C]
                        .unsqueeze(1)? // [B, 1, N, C]
                        .expand((b, l, n, c))? // [B, L, N, C]
                        .contiguous()?;
                    let e_idx_t = e_idx_t
                        .expand((b, l, n))? // [B, L, N]
                        .contiguous()?;
                    let h_es_t = cat_neighbors_nodes(&h_s, &h_e_t, &e_idx_t)?;
                    let h_exv_encoder_t = h_exv_encoder_fw.gather(
                        &t_gather
                            .unsqueeze(2)?
                            .unsqueeze(3)?
                            .expand((b, 1, h_exv_encoder_fw.dim(2)?, h_exv_encoder_fw.dim(3)?))?
                            .contiguous()?,
                        1,
                    )?;
                    let mask_bw_t = mask_bw.gather(
                        &t_gather
                            .unsqueeze(2)?
                            .unsqueeze(3)?
                            .expand((b, 1, mask_bw.dim(2)?, mask_bw.dim(3)?))?
                            .contiguous()?,
                        1,
                    )?;

                    // Decoder layers loop
                    for l in 0..self.decoder_layers.len() {
                        let h_v_stack_l = &h_v_stack[l];
                        let h_esv_decoder_t = cat_neighbors_nodes(h_v_stack_l, &h_es_t, &e_idx_t)?;
                        let h_v_t = h_v_stack_l.gather(
                            &t_gather
                                .unsqueeze(2)?
                                .expand((b, 1, h_v_stack_l.dim(2)?))?
                                .contiguous()?,
                            1,
                        )?;
                        let mask_bw_t = mask_bw_t.expand(h_esv_decoder_t.dims())?.contiguous()?;
                        let h_exv_encoder_t = h_exv_encoder_t
                            .expand(h_esv_decoder_t.dims())?
                            .contiguous()?
                            .to_dtype(sample_dtype)?;
                        let h_esv_t = mask_bw_t
                            .mul(&h_esv_decoder_t.to_dtype(sample_dtype)?)?
                            .add(&h_exv_encoder_t)?
                            .to_dtype(sample_dtype)?
                            .contiguous()?;
                        let h_v_t = h_v_t
                            .expand((
                                h_esv_t.dim(0)?, // batch size
                                h_esv_t.dim(1)?, // sequence length (93)
                                h_v_t.dim(2)?,   // features (128)
                            ))?
                            .contiguous()?;
                        let decoder_output = self.decoder_layers[l].forward(
                            &h_v_t,
                            &h_esv_t,
                            Some(&mask_t),
                            None,
                            None,
                        )?;
                        let t_expanded = t_gather.reshape(&[b])?; // This will give us a 1D tensor of shape [b]
                        let decoder_output = decoder_output
                            .narrow(1, 0, 1)?
                            .squeeze(1)? // Now [1, 128]
                            .unsqueeze(1)?; // Now [1, 1, 128] - same rank as target
                        h_v_stack[l + 1] =
                            h_v_stack[l + 1].index_add(&t_expanded, &decoder_output, 1)?;
                        // h_v_stack[l + 1] =
                        //     h_v_stack[l + 1].index_add(&t_expanded, &decoder_output, 1)?;
                    }
                    let h_v_t = h_v_stack
                        .last()
                        .unwrap()
                        .gather(
                            &t_gather
                                .unsqueeze(2)?
                                .expand((b, 1, h_v_stack.last().unwrap().dim(2)?))?
                                .contiguous()?,
                            1,
                        )?
                        .squeeze(1)?;
                    // Generate logits and probabilities
                    let logits = self.w_out.forward(&h_v_t)?;
                    let log_probs = log_softmax(&logits, D::Minus1)?;

                    // explicit for OoO
                    let probs = {
                        let biased_logits = logits.add(&bias_t)?; // (logits + bias_t)
                        let scaled_logits = (biased_logits / temperature)?; // (logits + bias_t) / temperature
                        softmax(&scaled_logits, D::Minus1)? // softmax((logits + bias_t) / temperature)
                    };

                    let probs_sample = probs
                        .narrow(1, 0, 20)?
                        .div(&probs.narrow(1, 0, 20)?.sum_keepdim(1)?.expand((b, 20))?)?;
                    // Sample new token
                    let sum = probs_sample.sum(1)?;
                    let probs_sample_1d = probs_sample
                        .squeeze(0)? // Remove batch dimension -> [20]
                        .clamp(1e-10, 1.0)?
                        .broadcast_div(&sum)?
                        .contiguous()?;

                    let s_t = multinomial_sample(&probs_sample_1d, temperature, seed)?;
                    let s_t = s_t.to_dtype(sample_dtype)?;
                    let s_true = s_true.to_dtype(sample_dtype)?;
                    let s_true_t = s_true.gather(&t_gather, 1)?.squeeze(1)?;
                    let s_t = s_t
                        .mul(&chain_mask_t)?
                        .add(&s_true_t.mul(&(&chain_mask_t.neg()? + 1.0)?)?)?
                        .to_dtype(DType::U32)?;

                    let s_t_idx = s_t.to_dtype(DType::U32)?;
                    let s_t_idx = s_t_idx.reshape(&[s_t_idx.dim(0)?])?;
                    let h_s_update = self.w_s.forward(&s_t_idx)?.unsqueeze(1)?;
                    let t_gather_expanded = t_gather.reshape(&[b])?;
                    let h_s_update = h_s_update.squeeze(0)?.unsqueeze(1)?;
                    h_s =
                        h_s.index_add(&t_gather_expanded, &Tensor::zeros_like(&h_s_update)?, 1)?;
                    h_s = h_s.index_add(&t_gather_expanded, &h_s_update, 1)?;

                    s = {
                        let dim = 1;
                        let start = t_gather.squeeze(0)?.squeeze(0)?.to_scalar::<u32>()? as usize;
                        let s_t_expanded = s_t.unsqueeze(1)?;
                        s.slice_scatter(&s_t_expanded, dim, start)?
                    };

                    let probs_update = chain_mask_t
                        .unsqueeze(1)?
                        .unsqueeze(2)?
                        .expand((b, 1, 20))?
                        .mul(&probs_sample.unsqueeze(1)?)?;
                    let t_expanded = t_gather.reshape(&[b])?;
                    let probs_update = probs_update
                        .squeeze(1)? // Remove extra dimension
                        .unsqueeze(1)?;
                    all_probs =
                        all_probs.index_add(&t_expanded, &Tensor::zeros_like(&probs_update)?, 1)?;
                    all_probs = all_probs.index_add(&t_expanded, &probs_update, 1)?;
                    let log_probs_update = chain_mask_t
                        .unsqueeze(1)?
                        .unsqueeze(2)?
                        .expand((b, 1, 21))?
                        .mul(&log_probs.unsqueeze(1)?)?
                        .squeeze(1)?
                        .unsqueeze(1)?;

                    all_log_probs = all_log_probs.index_add(
                        &t_expanded,
                        &Tensor::zeros_like(&log_probs_update)?,
                        1,
                    )?;
                    all_log_probs = all_log_probs.index_add(&t_expanded, &log_probs_update, 1)?;
                }

                println!("Before Score");
                Ok(ScoreOutput {
                    s,
                    log_probs: all_probs,
                    logits: all_log_probs,
                    decoding_order,
                })
            }
            Some(symmetry_residues) => {
                todo!()
                // // note this is a literal translation of the code... Howver I think this could lead to
                // // possible unintentional overwriting of value - e.g. if there are multiple identical
                // // values in the index. (I guess you might expect that if they are symetrical. Howver the
                // // weights do not have to be the same)
                // let symmetry_weights = symmetry_weights.as_ref().unwrap();
                // // let symmetry_weights_tensor = Tensro::ones(l, candle_core::DType::F32, device)?;
                // let mut symmetry_weights_vec = vec![1.0_f64; l];
                // for (i1, item_list) in symmetry_residues.iter().enumerate() {
                //     for (i2, &item) in item_list.iter().enumerate() {
                //         let value = symmetry_weights[i1][i2];
                //         symmetry_weights_vec[item as usize] = value;
                //     }
                // }

                // let symmetry_weights_tensor = Tensor::from_vec(symmetry_weights_vec, l, device)?;

                // // let flattened: Vec<i64> = new_decoding_order.into_iter().flatten().collect();
                // let mut new_decoding_order: Vec<Vec<i64>> = Vec::new();
                // let decoding_order_vec: Vec<i64> = decoding_order.get(0)?.to_vec1()?;
                // for &t_dec in &decoding_order_vec {
                //     if !new_decoding_order.iter().flatten().any(|&x| x == t_dec) {
                //         let list_a: Vec<&Vec<i64>> = symmetry_residues
                //             .iter()
                //             .filter(|item| item.contains(&t_dec))
                //             .collect();
                //         if !list_a.is_empty() {
                //             new_decoding_order.push(list_a[0].clone());
                //         } else {
                //             new_decoding_order.push(vec![t_dec]);
                //         }
                //     }
                // }
                // let flattened_order: Vec<i64> =
                //     new_decoding_order.clone().into_iter().flatten().collect();
                // let decoding_order = Tensor::from_vec(flattened_order, l, device)?
                //     .unsqueeze(0)?
                //     .repeat((b, 1))?;

                // // shuffle the decoding. Note: This is now non-deterministic
                // // let mut rng = thread_rng();
                // // let mut new_decoding_order: Vec<i64> = decoding_order_vec.clone();
                // // new_decoding_order.shuffle(&mut rng);
                // // let decoding_order =
                // //     Tensor::from_vec(new_decoding_order, l, device)?.repeat((b, 1))?;

                // let permutation_matrix_reverse = one_hot(decoding_order, l, 1., 0.)?;

                // let tril = Tensor::tril2(l, DType::F64, device)?;
                // let temp = tril.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // (b, i, q)
                // let order_mask_backward =
                //     temp.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // shape (b, q, p)
                // let mask_attend = order_mask_backward
                //     .gather(&e_idx, 2)?
                //     .unsqueeze(D::Minus1)?;

                // let mask_1d = x_mask.unwrap().reshape((b, l, 1, 1))?;
                // let mask_bw = mask_1d.mul(&mask_attend)?;
                // let mask_fw = mask_1d.mul(&(Tensor::ones_like(&mask_attend)? - mask_attend)?)?;

                // // Repeat for decoding
                // let s_true = s_true.repeat((b, 1))?;
                // let h_v = h_v.repeat((b, 1, 1))?;
                // let h_e = h_e.repeat((b, 1, 1, 1))?;
                // let e_idx = e_idx.repeat((b, 1, 1))?;
                // let mask_fw = mask_fw.repeat((b, 1, 1, 1))?;
                // let mask_bw = mask_bw.repeat((b, 1, 1, 1))?;
                // let chain_mask = chain_mask.repeat((b, 1))?;
                // let mask = x_mask.unwrap().repeat((b, 1))?;
                // // Todo: fix bias
                // let bias = Tensor::zeros((b, l, 20), DType::F32, device)?;
                // let bias = bias.repeat((b, 1, 1))?;
                // let all_probs = Tensor::zeros((b, l, 20), candle_core::DType::F32, device)?;
                // let all_log_probs = Tensor::zeros((b, l, 21), candle_core::DType::F32, device)?;
                // let h_s = Tensor::zeros_like(&h_v)?;
                // let s = (Tensor::ones((b, l), candle_core::DType::I64, device)? * 20.)?;

                // let mut h_v_stack = vec![h_v.clone()];
                // h_v_stack.extend(
                //     (0..self.decoder_layers.len()).map(|_| Tensor::zeros_like(&h_v).unwrap()),
                // );
                // let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
                // let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;
                // let h_exv_encoder_fw = mask_fw.mul(&h_exv_encoder)?;

                // for t_list in new_decoding_order {
                //     let mut total_logits = Tensor::zeros((b, 21), candle_core::DType::F32, device)?;

                //     for &t in &t_list {
                //         // Select the t-th column from chain_mask
                //         let chain_mask_t = chain_mask.i((.., t as usize))?;
                //         let mask_t = mask.i((.., t as usize))?;
                //         let bias_t = bias.i((.., t as usize))?;

                //         let e_idx_t = e_idx.narrow(1, t as usize, 1)?;
                //         let h_e_t = h_e.narrow(1, t as usize, 1)?;
                //         let h_es_t = cat_neighbors_nodes(&h_s, &h_e_t, &e_idx_t)?;
                //         let h_exv_encoder_t = h_exv_encoder_fw.narrow(1, t as usize, 1)?;

                //         for (l, layer) in self.decoder_layers.iter().enumerate() {
                //             let h_esv_decoder_t =
                //                 cat_neighbors_nodes(&h_v_stack[l], &h_es_t, &e_idx_t)?;
                //             let h_v_t = h_v_stack[l].narrow(1, t as usize, 1)?;
                //             let h_esv_t = mask_bw
                //                 .narrow(1, t as usize, 1)?
                //                 .mul(&h_esv_decoder_t)?
                //                 .add(&h_exv_encoder_t)?;
                //             let new_h_v = layer.forward(
                //                 &h_v_t,
                //                 &h_esv_t,
                //                 Some(&mask_t.unsqueeze(1)?),
                //                 None,
                //                 None,
                //             )?;
                //             h_v_stack[l + 1].slice_set(&new_h_v, 1, t as usize)?;
                //         }

                //         let h_v_t = h_v_stack.last().unwrap().i((.., t as usize))?;
                //         let logits = self.w_out.forward(&h_v_t)?;
                //         let log_probs = log_softmax(&logits, D::Minus1)?;
                //         let updated_probs = chain_mask_t.unsqueeze(1)?.mul(&log_probs)?;
                //         all_log_probs.slice_set(&updated_probs, 1, t as usize)?;
                //         let symvec = &symmetry_weights[t as usize];
                //         let symten = Tensor::new(symvec.as_slice(), device)?;
                //         total_logits = total_logits.add(&logits.mul(&symten)?)?;
                //     }

                //     // todo: bias t not defined here!
                //     let bias_t = Tensor::zeros_like(&total_logits)?;
                //     let temperature = 20.;
                //     let probs = softmax(&(total_logits.add(&bias_t)? / temperature)?, D::Minus1)?;
                //     let probs_sample = probs
                //         .narrow(1, 0, 20)?
                //         .div(&probs.narrow(1, 0, 20)?.sum_keepdim(1)?)?;

                //     // replce this with sampleing using built in Logit Processing
                //     // let s_t = probs_sample.multinomial(1, true)?.squeeze(1)?;
                //     let seed = 32;
                //     let mut logproc = LogitsProcessor::new(seed, Some(temperature), Some(0.25));
                //     let logits: Vec<u32> = vec![(); l]
                //         .iter()
                //         .map(|_| logproc.sample(&probs_sample))
                //         .filter_map(Result::ok)
                //         .collect();
                //     let s_t = Tensor::from_vec(logits, l, device)?;

                //     for &t in &t_list {
                //         let chain_mask_t = chain_mask.i((.., t as usize))?;
                //         let result = chain_mask_t.unsqueeze(1)?.mul(&probs_sample)?;

                //         all_probs.slice_set(&result, 1, t as usize)?;

                //         let s_true_t = s_true.i((.., t as usize))?;
                //         let s_t = s_t
                //             .mul(&chain_mask_t)?
                //             .add(
                //                 &s_true_t
                //                     .mul(&(Tensor::ones_like(&chain_mask_t)? - chain_mask_t)?)?,
                //             )?
                //             .to_dtype(candle_core::DType::I64)?;

                //         let h_s_t = self.w_s.forward(&s_t)?;
                //         h_s.slice_set(&h_s_t.unsqueeze(1)?, 1, t as usize)?;
                //         s.slice_set(&s_t.unsqueeze(1)?, 1, t as usize)?;
                //     }
                // }
                // Ok(ScoreOutput {
                //     s,
                //     // sampling_probs: all_probs,
                //     log_probs: all_log_probs,
                //     logits: all_probs,
                //     decoding_order,
                // })
            }
        }
    }

    fn single_aa_score() {
        todo!()
    }
    pub fn score(&self, features: &ProteinFeatures, use_sequence: bool) -> Result<ScoreOutput> {
        let sample_dtype = DType::F32;
        let ProteinFeatures { s, x_mask, .. } = &features;

        let s_true = &s.clone();
        let device = s_true.device();
        let (b, l) = s_true.dims2()?;
        let mask = &x_mask.as_ref().clone();
        let b_decoder: usize = b;

        // Todo: This is a hack. we should be passing in encoded chains.
        // Update chain_mask to include missing regions
        let chain_mask = Tensor::zeros_like(mask.unwrap())?.to_dtype(sample_dtype)?;
        let chain_mask = mask.unwrap().mul(&chain_mask)?;

        // encode ...
        let (h_v, h_e, e_idx) = self.encode(features)?;
        let rand_tensor = Tensor::randn(0f32, 1f32, (b, l), device)?.to_dtype(sample_dtype)?;
        // Compute decoding order
        let decoding_order = (chain_mask + 0.001)?
            .mul(&rand_tensor.abs()?)?
            .arg_sort_last_dim(false)?;

        let symmetry_residues: Option<Vec<i32>> = None;

        let (mask_fw, mask_bw, e_idx, decoding_order) = match symmetry_residues {
            Some(symmetry_residues) => {
                todo!();
            }
            None => {
                let e_idx = e_idx.repeat(&[b_decoder, 1, 1])?;
                let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1f32, 0f32)?
                    .to_dtype(sample_dtype)?
                    .contiguous()?;
                let tril = Tensor::tril2(l, sample_dtype, device)?;
                let tril = tril.unsqueeze(0)?;
                let temp = tril
                    .matmul(&permutation_matrix_reverse.transpose(1, 2)?)?
                    .contiguous()?; // shape (b, i, q)
                let order_mask_backward = temp
                    .matmul(&permutation_matrix_reverse.transpose(1, 2)?)?
                    .contiguous()?; // shape (b, q, p)
                let mask_attend = order_mask_backward
                    .gather(&e_idx, 2)?
                    .unsqueeze(D::Minus1)?;
                let mask_1d = mask.unwrap().reshape((b, l, 1, 1))?;
                // Broadcast mask_1d to match mask_attend's shape
                let mask_1d = mask_1d
                    .broadcast_as(mask_attend.shape())?
                    .to_dtype(sample_dtype)?;
                let mask_bw = mask_1d.mul(&mask_attend)?;
                let mask_fw = mask_1d.mul(&(mask_attend - 1.0)?.neg()?)?;
                (mask_fw, mask_bw, e_idx, decoding_order)
            }
        };
        let b_decoder = b_decoder;
        let s_true = s_true.repeat(&[b_decoder, 1])?;
        let h_v = h_v.repeat(&[b_decoder, 1, 1])?;
        let h_e = h_e.repeat(&[b_decoder, 1, 1, 1])?;
        let mask = x_mask.as_ref().unwrap().repeat(&[b_decoder, 1])?;
        let h_s = self.w_s.forward(&s_true)?; // embedding layer
        let h_es = cat_neighbors_nodes(&h_s, &h_e, &e_idx)?;
        // Build encoder embeddings
        let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
        let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;
        let h_exv_encoder_fw = mask_fw
            .broadcast_as(h_exv_encoder.shape())?
            .to_dtype(h_exv_encoder.dtype())?
            .mul(&h_exv_encoder)?;
        let mut h_v = h_v;
        if !use_sequence {
            for layer in &self.decoder_layers {
                h_v = layer.forward(&h_v, &h_exv_encoder_fw, Some(&mask), None, None)?;
            }
        } else {
            for layer in &self.decoder_layers {
                let h_esv = cat_neighbors_nodes(&h_v, &h_es, &e_idx)?;
                let h_esv = mask_bw.mul(&h_esv)?.add(&h_exv_encoder_fw)?;
                h_v = layer.forward(&h_v, &h_esv, Some(&mask), None, None)?;
            }
        }
        let logits = self.w_out.forward(&h_v)?;
        let log_probs = log_softmax(&logits, D::Minus1)?;

        Ok(ScoreOutput {
            s: s_true,
            log_probs,
            logits,
            decoding_order,
        })
    }
}
