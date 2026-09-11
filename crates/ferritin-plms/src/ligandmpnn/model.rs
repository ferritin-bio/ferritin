//! A message passing protein design neural network
//! that samples sequences diffusing conditional probabilities.
//!
//! - See the [LigandMPNN Repo](https://github.com/dauparas/LigandMPNN)
//!

// The layer stacks fold a `Result` accumulator over the encoder/decoder layers;
// the explicit `fold(Ok(..), ..)` reads more clearly than `try_fold` here.
#![allow(clippy::manual_try_fold)]

use super::configs::{ModelTypes, ProteinMPNNConfig};
use super::ligandfeaturesmodel::ProteinFeaturesLigand;
use super::proteinfeatures::ProteinFeatures;
use super::proteinfeaturesmodel::ProteinFeaturesModel;
use crate::featurize::utilities::{cat_neighbors_nodes, gather_nodes, int_to_aa1, linear_last_dim};
use crate::types::PseudoProbability;
use candle_core::safetensors;
use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::encoding::one_hot;
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::{Dropout, Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};
use candle_transformers::generation::LogitsProcessor;
use std::collections::HashMap;

// refactoring common fn
fn concat_node_tensors(h_v: &Tensor, h_e: &Tensor, e_idx: &Tensor) -> Result<Tensor> {
    let h_ev = cat_neighbors_nodes(h_v, h_e, e_idx)?;
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
///  Score dims are [Batch, seqlength]
impl ScoreOutput {
    pub fn get_sequences(&self) -> Result<Vec<String>> {
        let (b, l) = self.s.dims2()?;
        let mut sequences = Vec::with_capacity(b);
        for batch_idx in 0..b {
            let batch = self.s.get(batch_idx)?;
            let mut sequence = String::with_capacity(l);
            for pos in 0..l {
                let aa_idx = batch.get(pos)?.to_vec0::<u32>()?;
                sequence.push(int_to_aa1(aa_idx));
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
    pub fn get_pseudo_probabilities(&self) -> Result<Vec<PseudoProbability>> {
        let (batch_size, seq_len, _vocab_size) = self.logits.dims3()?;
        let mut all_probabilities = Vec::with_capacity(batch_size * seq_len);

        for batch_idx in 0..batch_size {
            let batch_logits = self.logits.get(batch_idx)?;

            for pos in 0..seq_len {
                let pos_logits = batch_logits.get(pos)?;
                let probs = softmax(&pos_logits, 0)?;
                let probs = probs.to_vec1::<f32>()?;

                // Get the decoding order to determine the actual position
                let actual_pos = if let Ok(order) = self
                    .decoding_order
                    .get(batch_idx)?
                    .get(pos)?
                    .to_scalar::<u32>()
                {
                    order as usize
                } else {
                    pos
                };

                #[allow(clippy::needless_range_loop)] // aa_idx also names the output amino acid
                for aa_idx in 0..probs.len() {
                    if probs[aa_idx] > 0.01 {
                        // Only include probabilities above threshold
                        all_probabilities.push(PseudoProbability {
                            position: actual_pos,
                            pseudo_prob: probs[aa_idx],
                            amino_acid: int_to_aa1(aa_idx as u32),
                        });
                    }
                }
            }
        }
        // Sort by position and then by probability (descending)
        all_probabilities.sort_by(|a, b| {
            a.position.cmp(&b.position).then(
                b.pseudo_prob
                    .partial_cmp(&a.pseudo_prob)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        Ok(all_probabilities)
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
        let _ = safetensors::save(&tensors, &filename);
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

impl PositionWiseFeedForward {
    /// [`Module::forward`] for inputs above rank 3.
    ///
    /// `DecLayerJ` feeds this `(B, L, M, C)`, which `Linear::forward` cannot
    /// matmul directly (ferritin-100.11).
    fn forward_any_rank(&self, x: &Tensor) -> Result<Tensor> {
        let h = linear_last_dim(&self.w1, x)?.gelu_erf()?;
        linear_last_dim(&self.w2, &h)
    }
}

impl Module for PositionWiseFeedForward {
    /// `gelu_erf`, not `gelu`.
    ///
    /// candle's `Tensor::gelu` is the tanh approximation; `torch.nn.GELU()`
    /// defaults to `approximate="none"`, the exact erf form. Every activation
    /// in this module used the approximation, which left ~1e-3 of drift per
    /// layer against the reference (ferritin-100.11).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w1
            .forward(x)?
            .gelu_erf()
            .and_then(|x| self.w2.forward(&x))
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
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
        let augment_eps = 1e-5f64;
        let num_in = (config.hidden_dim * 2) as usize;
        let dropout_ratio = config.dropout_ratio;

        // Create layer norms
        let norm1 = layer_norm(num_hidden, augment_eps, vb.pp("norm1"))?;
        let norm2 = layer_norm(num_hidden, augment_eps, vb.pp("norm2"))?;
        let norm3 = layer_norm(num_hidden, augment_eps, vb.pp("norm3"))?;

        // Create linear layers
        let w1 = linear(num_hidden + num_in, num_hidden, vb.pp("W1"))?;
        let w2 = linear(num_hidden, num_hidden, vb.pp("W2"))?;
        let w3 = linear(num_hidden, num_hidden, vb.pp("W3"))?;
        let w11 = linear(num_hidden + num_in, num_hidden, vb.pp("W11"))?;
        let w12 = linear(num_hidden, num_hidden, vb.pp("W12"))?;
        let w13 = linear(num_hidden, num_hidden, vb.pp("W13"))?;

        // Create dropouts with same ratio
        let dropout1 = Dropout::new(dropout_ratio);
        let dropout2 = Dropout::new(dropout_ratio);
        let dropout3 = Dropout::new(dropout_ratio);

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
            .gelu_erf()?
            .apply(&self.w2)?
            .gelu_erf()?
            .apply(&self.w3)?;

        let h_message = mask_attend
            .map(|mask| mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_message))
            .transpose()?
            .unwrap_or(h_message);

        // Safe division with scale
        let scale = if self.scale == 0.0 { 1.0 } else { self.scale };
        let dh = (h_message.sum(D::Minus2)? / scale)?;

        let h_v = apply_dropout_and_norm(&h_v, &dh, &self.dropout1, &self.norm1, training)?;
        let dense_output = self.dense.forward(&h_v)?;
        let h_v =
            apply_dropout_and_norm(&h_v, &dense_output, &self.dropout2, &self.norm2, training)?;

        // Apply mask if provided
        let h_v = mask_v
            .map(|mask| mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_v))
            .transpose()?
            .unwrap_or(h_v);

        let h_ev = concat_node_tensors(&h_v, h_e, e_idx)?;
        let h_message = self
            .w11
            .forward(&h_ev)?
            .gelu_erf()?
            .apply(&self.w12)?
            .gelu_erf()?
            .apply(&self.w13)?;

        let h_e = apply_dropout_and_norm(h_e, &h_message, &self.dropout3, &self.norm3, training)?;
        Ok((h_v, h_e))
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
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
    /// The main decoder stack: `num_in = 3 * hidden_dim`.
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig, layer: i32) -> Result<Self> {
        Self::load_with_num_in(vb, config, layer, (config.hidden_dim * 3) as usize)
    }

    /// `num_in` explicitly, for LigandMPNN's `context_encoder_layers`.
    ///
    /// Those are `DecLayer(hidden_dim, hidden_dim * 2)` in the reference — the
    /// message is `[h_V_C, h_E_context, Y_nodes]` rather than the decoder's
    /// three hidden-width blocks — so their `W1` is `(128, 384)` where the
    /// decoder's is `(128, 512)` (ferritin-100.11).
    pub fn load_with_num_in(
        vb: VarBuilder,
        config: &ProteinMPNNConfig,
        layer: i32,
        num_in: usize,
    ) -> Result<Self> {
        let vb = vb.pp(layer); // handle the layer number here.
        let num_hidden = config.hidden_dim as usize;
        let augment_eps = 1e-5f64;
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

        // Expand node features to match edge dimensions
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
            .gelu_erf()?
            .apply(&self.w2)?
            .gelu_erf()?
            .apply(&self.w3)?;

        let h_message = self.dropout1.forward(&h_message, training_bool)?;

        let h_message = mask_attend
            .map(|mask| mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_message))
            .transpose()?
            .unwrap_or(h_message);

        let dh = (h_message.sum(D::Minus2)? / self.scale)?;
        let h_v = self.norm1.forward(&(h_v + dh)?)?;
        let dh = self.dense.forward(&h_v)?;
        let dh_dropout = self.dropout2.forward(&dh, training_bool)?;
        let h_v = self.norm2.forward(&(h_v + dh_dropout)?)?;

        // Apply optional node mask
        let h_v = mask_v
            .map(|mask| mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_v))
            .transpose()?
            .unwrap_or(h_v);

        Ok(h_v)
    }
}

/// How many context-encoder rounds LigandMPNN runs.
///
/// Fixed at 2 in the reference — `range(2)` for both `context_encoder_layers`
/// and `y_context_encoder_layers` — not derived from any config field, and
/// every published checkpoint has exactly two of each.
const LIGAND_CONTEXT_LAYERS: i32 = 2;

/// [`DecLayer`] applied one dimension deeper, over LigandMPNN's ligand graph.
///
/// Port of `DecLayerJ`
/// ([model_utils.py](https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L1582)).
/// Structurally identical to [`DecLayer`] — same weight names, same
/// `W3(act(W2(act(W1))))` message and same two residual norms — and the
/// reference comments the single difference as "the only difference": the node
/// expansion.
///
/// [`DecLayer`] carries `h_V` as `(B, L, C)` and broadcasts it over `K`
/// neighbours. Here `h_V` is `Y_nodes`, already `(B, L, M, C)` — one node per
/// (residue, ligand atom) pair — and it broadcasts over the ligand graph's own
/// `M` edges to `(B, L, M, M, C)`.
#[derive(Clone, Debug)]
pub struct DecLayerJ {
    scale: f64,
    norm1: LayerNorm,
    norm2: LayerNorm,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    dense: PositionWiseFeedForward,
}

impl DecLayerJ {
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig, layer: i32) -> Result<Self> {
        let vb = vb.pp(layer);
        let num_hidden = config.hidden_dim as usize;
        // `DecLayerJ(hidden_dim, hidden_dim)`: the message is [Y_nodes,
        // Y_edges], two hidden-width blocks, so W1 is (128, 256).
        let num_in = num_hidden;
        let eps = 1e-5f64;

        Ok(Self {
            scale: config.scale_factor,
            norm1: layer_norm::layer_norm(num_hidden, eps, vb.pp("norm1"))?,
            norm2: layer_norm::layer_norm(num_hidden, eps, vb.pp("norm2"))?,
            w1: linear::linear(num_hidden + num_in, num_hidden, vb.pp("W1"))?,
            w2: linear::linear(num_hidden, num_hidden, vb.pp("W2"))?,
            w3: linear::linear(num_hidden, num_hidden, vb.pp("W3"))?,
            dense: PositionWiseFeedForward::new(vb.pp("dense"), num_hidden, num_hidden * 4)?,
        })
    }

    /// `h_v` is `(B, L, M, C)`; `h_e` is `(B, L, M, M, C)`.
    pub fn forward(
        &self,
        h_v: &Tensor,
        h_e: &Tensor,
        mask_v: Option<&Tensor>,
        mask_attend: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, m, _, _) = h_e.dims5()?;
        let c = h_v.dim(D::Minus1)?;
        let h_v_expand = h_v.unsqueeze(D::Minus2)?.broadcast_as((b, l, m, m, c))?;
        let h_ev = Tensor::cat(&[&h_v_expand, h_e], D::Minus1)?.contiguous()?;

        // Rank 5 throughout: `Linear::forward` matmuls directly and caps at
        // rank 3, so every projection here goes through `linear_last_dim`.
        let h_message = linear_last_dim(&self.w1, &h_ev)?.gelu_erf()?;
        let h_message = linear_last_dim(&self.w2, &h_message)?.gelu_erf()?;
        let h_message = linear_last_dim(&self.w3, &h_message)?;
        let h_message = mask_attend
            .map(|mask| mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_message))
            .transpose()?
            .unwrap_or(h_message);

        let dh = (h_message.sum(D::Minus2)? / self.scale)?;
        let h_v = self.norm1.forward(&(h_v + dh)?)?;
        let dh = self.dense.forward_any_rank(&h_v)?;
        let h_v = self.norm2.forward(&(h_v + dh)?)?;

        mask_v
            .map(|mask| mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_v))
            .transpose()?
            .map_or(Ok(h_v), Ok)
    }
}

/// ProteinMPNN Model
/// - [link](https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L10C7-L10C18)
pub struct ProteinMPNN {
    pub(crate) config: ProteinMPNNConfig,
    pub(crate) decoder_layers: Vec<DecLayer>,
    pub(crate) device: Device,
    pub(crate) encoder_layers: Vec<EncLayer>,
    /// The protein-only featurizer, for `ModelTypes::ProteinMPNN`.
    pub(crate) features: Option<ProteinFeaturesModel>,
    /// Everything only `ModelTypes::LigandMPNN` has.
    pub(crate) ligand: Option<LigandModules>,
    pub(crate) w_e: Linear,
    pub(crate) w_out: Linear,
    pub(crate) w_s: Embedding,
}

/// The modules LigandMPNN adds on top of the shared ProteinMPNN stack.
///
/// Grouped rather than scattered as eight `Option` fields so that "this is a
/// LigandMPNN" is one check, and so a ProteinMPNN cannot be half-constructed
/// with some of them present.
pub(crate) struct LigandModules {
    features: ProteinFeaturesLigand,
    /// Projects the ligand node features `V` into the context message.
    w_v: Linear,
    /// Projects `h_V` into the context path before the context encoder runs.
    w_c: Linear,
    w_nodes_y: Linear,
    w_edges_y: Linear,
    /// Final projection of the context path, added back onto `h_V`.
    v_c: Linear,
    v_c_norm: LayerNorm,
    context_encoder_layers: Vec<DecLayer>,
    y_context_encoder_layers: Vec<DecLayerJ>,
}

impl ProteinMPNN {
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig) -> Result<Self> {
        let hidden_dim = config.hidden_dim as usize;
        let edge_features = config.edge_features as usize;
        let num_letters = config.num_letters as usize;
        let node_features = config.node_features as usize;
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

        // The two model types use DIFFERENT featurizers under the same
        // `features.` prefix: ProteinFeatures for ProteinMPNN,
        // ProteinFeaturesLigand for LigandMPNN. The latter is a superset —
        // same `embeddings`/`edge_embedding`/`norm_edges`, plus the ligand
        // node and graph projections — so loading the wrong one against a
        // LigandMPNN checkpoint silently ignores nine tensors rather than
        // failing.
        let (features, ligand) = match config.model_type {
            ModelTypes::ProteinMPNN => (
                Some(ProteinFeaturesModel::load(
                    vb.pp("features"),
                    config.clone(),
                )?),
                None,
            ),
            ModelTypes::LigandMPNN => (
                None,
                Some(LigandModules {
                    features: ProteinFeaturesLigand::load(vb.pp("features"), config)?,
                    w_v: linear::linear(node_features, hidden_dim, vb.pp("W_v"))?,
                    w_c: linear::linear(hidden_dim, hidden_dim, vb.pp("W_c"))?,
                    w_nodes_y: linear::linear(hidden_dim, hidden_dim, vb.pp("W_nodes_y"))?,
                    w_edges_y: linear::linear(hidden_dim, hidden_dim, vb.pp("W_edges_y"))?,
                    v_c: linear::linear_no_bias(hidden_dim, hidden_dim, vb.pp("V_C"))?,
                    v_c_norm: layer_norm::layer_norm(hidden_dim, 1e-5f64, vb.pp("V_C_norm"))?,
                    // DecLayer(hidden, hidden * 2), not the decoder's
                    // hidden * 3: the message is [h_V_C, h_E_context, Y_nodes].
                    context_encoder_layers: (0..LIGAND_CONTEXT_LAYERS)
                        .map(|i| {
                            DecLayer::load_with_num_in(
                                vb.pp("context_encoder_layers"),
                                config,
                                i,
                                hidden_dim * 2,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                    y_context_encoder_layers: (0..LIGAND_CONTEXT_LAYERS)
                        .map(|i| DecLayerJ::load(vb.pp("y_context_encoder_layers"), config, i))
                        .collect::<Result<Vec<_>>>()?,
                }),
            ),
        };

        Ok(Self {
            config: config.clone(),
            decoder_layers,
            device: vb.device().clone(),
            encoder_layers,
            features,
            ligand,
            w_e,
            w_out,
            w_s,
        })
    }
    /// Encode the structure into node and edge embeddings.
    ///
    /// Both model types run the same three [`EncLayer`]s over the same protein
    /// neighbour graph. LigandMPNN then folds in a ligand context path before
    /// returning — `Self::encode_ligand_context`, a private helper, so not
    /// linkable from here.
    pub fn encode(&self, features: &ProteinFeatures) -> Result<(Tensor, Tensor, Tensor)> {
        let mask = self.sequence_mask(features)?;

        let (v, e, e_idx, ligand_graph) = match (&self.features, &self.ligand) {
            (Some(protein), None) => {
                let (e, e_idx) = protein.forward(features, &self.device)?;
                (None, e, e_idx, None)
            }
            (None, Some(ligand)) => {
                let f = ligand.features.forward(features)?;
                (Some(f.v), f.e, f.e_idx, Some((f.y_nodes, f.y_edges, f.y_m)))
            }
            // `load` builds exactly one of the two, keyed on model_type.
            _ => candle_core::bail!(
                "{:?}: built with no featurizer, or with both",
                self.config.model_type
            ),
        };

        let h_v = Tensor::zeros(
            (e.dim(0)?, e.dim(1)?, e.dim(D::Minus1)?),
            DType::F32,
            &self.device,
        )?;
        let h_e = self.w_e.forward(&e)?;

        // mask_attend = mask[..., None] * gather_nodes(mask[..., None], E_idx)
        let mask_gathered =
            gather_nodes(&mask.unsqueeze(D::Minus1)?, &e_idx)?.squeeze(D::Minus1)?;
        let mask_attend = mask.unsqueeze(D::Minus1)?.broadcast_mul(&mask_gathered)?;

        let (h_v, h_e) = self
            .encoder_layers
            .iter()
            .try_fold((h_v, h_e), |(h_v, h_e), layer| {
                layer.forward(
                    &h_v,
                    &h_e,
                    &e_idx,
                    Some(&mask),
                    Some(&mask_attend),
                    Some(false),
                )
            })?;

        let h_v = match (&self.ligand, v, ligand_graph) {
            (Some(ligand), Some(v), Some((y_nodes, y_edges, y_m))) => {
                self.encode_ligand_context(ligand, &h_v, &v, &y_nodes, &y_edges, &y_m, &mask)?
            }
            _ => h_v,
        };

        Ok((h_v, h_e, e_idx))
    }

    /// Per-residue validity as F32, defaulting to all-valid.
    fn sequence_mask(&self, features: &ProteinFeatures) -> Result<Tensor> {
        match features.get_sequence_mask() {
            Some(m) => m.to_dtype(DType::F32),
            None => Tensor::ones_like(features.get_sequence())?.to_dtype(DType::F32),
        }
    }

    /// LigandMPNN's ligand context path.
    ///
    /// Two interleaved message-passing rounds, then one residual back onto the
    /// protein node embeddings:
    ///
    /// 1. `Y_nodes` exchange messages over the ligand graph's own edges
    ///    ([`DecLayerJ`], one dimension deeper than the protein layers).
    /// 2. The updated `Y_nodes` are concatenated onto the projected ligand
    ///    node features and passed to a [`DecLayer`], updating a per-residue
    ///    context vector `h_V_C`.
    /// 3. `h_V = h_V + V_C_norm(V_C(h_V_C))`.
    ///
    /// Note both loops read `y_context_encoder_layers[i]` and
    /// `context_encoder_layers[i]` in the same iteration — the ligand graph is
    /// updated first, and the context layer sees that round's output, not the
    /// previous one's.
    #[allow(clippy::too_many_arguments)] // one call site; naming a struct for it adds nothing
    fn encode_ligand_context(
        &self,
        ligand: &LigandModules,
        h_v: &Tensor,
        v: &Tensor,
        y_nodes: &Tensor,
        y_edges: &Tensor,
        y_m: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        // Rank 4 (B, L, M, C).
        let h_e_context = linear_last_dim(&ligand.w_v, v)?;
        let mut h_v_c = ligand.w_c.forward(h_v)?;

        let y_m = y_m.to_dtype(DType::F32)?;
        // Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        let y_m_edges = y_m
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&y_m.unsqueeze(D::Minus2)?)?;

        // Rank 4 (B, L, M, C).
        let mut y_nodes = linear_last_dim(&ligand.w_nodes_y, y_nodes)?;
        // Rank 5 (B, L, M, M, C).
        let y_edges = linear_last_dim(&ligand.w_edges_y, y_edges)?;

        for (y_layer, context_layer) in ligand
            .y_context_encoder_layers
            .iter()
            .zip(&ligand.context_encoder_layers)
        {
            y_nodes = y_layer.forward(&y_nodes, &y_edges, Some(&y_m), Some(&y_m_edges))?;
            let h_e_context_cat = Tensor::cat(&[&h_e_context, &y_nodes], D::Minus1)?;
            h_v_c = context_layer.forward(
                &h_v_c,
                &h_e_context_cat,
                Some(mask),
                Some(&y_m),
                Some(false),
            )?;
        }

        let h_v_c = ligand.v_c.forward(&h_v_c)?;
        h_v + ligand.v_c_norm.forward(&h_v_c)?
    }

    // Removed unused decode methods
    pub fn simple_decode(&self, features: &ProteinFeatures) -> Result<ScoreOutput> {
        // Create a batch size of 1 for simple decoding
        let b_decoder = 1;

        // Extract relevant features
        let ProteinFeatures { s, x_mask, .. } = features;
        let device = s.device();
        let (_, l) = s.dims2()?;

        // Encode the structure once
        let (h_v_enc, h_e_enc, e_idx_enc) = self.encode(features)?;

        // Process all positions at once with a simplified approach
        let s_true = s.clone();
        let mask = x_mask.clone().unwrap();

        // Create tensors for the decoder
        let zeros = Tensor::zeros((b_decoder, l, h_v_enc.dim(D::Minus1)?), DType::F32, device)?;
        let h_v = h_v_enc.clone();
        let h_e = h_e_enc.clone();
        let e_idx = e_idx_enc.clone();

        // Build encoder embeddings for neighbors
        let h_ex_encoder = cat_neighbors_nodes(&zeros, &h_e, &e_idx)?;
        let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;

        // Apply decoder layers using only structure information
        let h_v_final = self.decoder_layers.iter().fold(Ok(h_v), |acc, layer| {
            layer.forward(&acc?, &h_exv_encoder, Some(&mask), None, None)
        })?;

        // Calculate logits and log probabilities
        let logits = self.w_out.forward(&h_v_final)?;
        let log_probs = log_softmax(&logits, D::Minus1)?;

        // For the decoding order, just use a placeholder
        let decoding_order = Tensor::arange(0, l as i64, device)?
            .reshape((1, l))?
            .broadcast_as((b_decoder, l))?
            .to_dtype(DType::F32)?;

        // Return the output directly
        Ok(ScoreOutput {
            s: s_true,
            log_probs,
            logits,
            decoding_order,
        })
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
        // Ascending, matching `torch.argsort`, whose default is ascending.
        // Positions with chain_mask == 0 are fixed, so their product is ~1e-4
        // and they must be decoded FIRST; sorting descending put them last and
        // inverted the intent (ferritin-100.11).
        let decoding_order = (&chain_mask + 0.0001)?
            .mul(&rand_tensor.abs()?)?
            .arg_sort_last_dim(true)?;
        // TodoL add  bias
        // # [B,L,21] - amino acid bias per position
        let bias = Tensor::ones((b, l, 21), sample_dtype, device)?;
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
                Ok(ScoreOutput {
                    s,
                    log_probs: all_probs,
                    logits: all_log_probs,
                    decoding_order,
                })
            }
            Some(_symmetry_residues) => {
                candle_core::bail!("symmetry residues in sample() not yet implemented")
            }
        }
    }

    pub fn score(&self, features: &ProteinFeatures, use_sequence: bool) -> Result<ScoreOutput> {
        let ProteinFeatures { s, x_mask, .. } = &features;
        let sample_dtype = DType::F32;
        let s_true = &s.clone();
        let device = s_true.device();
        let (b, l) = s_true.dims2()?;
        let mask = &x_mask.as_ref().clone();
        let b_decoder: usize = b;

        // Todo: This is a hack. we should be passing in encoded chains.
        // Update chain_mask to include missing regions
        let chain_mask = Tensor::zeros_like(mask.unwrap())?.to_dtype(sample_dtype)?;
        let chain_mask = mask.unwrap().mul(&chain_mask)?; // does the order count here?

        // encode ...
        let (h_v, h_e, e_idx) = self.encode(features)?;
        let rand_tensor = Tensor::randn(0f32, 1f32, (b, l), device)?.to_dtype(sample_dtype)?;
        // Compute decoding order
        // Ascending: see the note in `sample` (ferritin-100.11).
        let decoding_order = (chain_mask + 0.001)?
            .mul(&rand_tensor.abs()?)?
            .arg_sort_last_dim(true)?;

        let symmetry_residues: Option<Vec<i32>> = None;

        let (mask_fw, mask_bw, e_idx, decoding_order) = match symmetry_residues {
            Some(_symmetry_residues) => {
                candle_core::bail!("symmetry residues in score() not yet implemented")
            }
            None => {
                let e_idx = e_idx.repeat(&[b_decoder, 1, 1])?;
                let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1f32, 0f32)?
                    .to_dtype(sample_dtype)?
                    .contiguous()?;

                let tril = Tensor::tril2(l, sample_dtype, device)?.unsqueeze(0)?;
                let temp = tril
                    .matmul(&permutation_matrix_reverse.transpose(1, 2)?)?
                    .contiguous()?; // shape (b, i, q)
                let order_mask_backward = temp
                    .matmul(&permutation_matrix_reverse.transpose(1, 2)?)?
                    .contiguous()?; // shape (b, q, p)
                let mask_attend = order_mask_backward
                    .gather(&e_idx, 2)?
                    .unsqueeze(D::Minus1)?;

                // Broadcast mask_1d to match mask_attend's shape
                let mask_1d = mask
                    .unwrap()
                    .reshape((b, l, 1, 1))?
                    .broadcast_as(mask_attend.shape())?
                    .to_dtype(sample_dtype)?;

                let mask_bw = mask_1d.mul(&mask_attend)?;
                let mask_fw = mask_1d.mul(&(mask_attend - 1.0)?.neg()?)?;
                (mask_fw, mask_bw, e_idx, decoding_order)
            }
        };

        let s_true = s_true.repeat(&[b_decoder, 1])?;
        let h_v = h_v.repeat(&[b_decoder, 1, 1])?;
        let h_e = h_e.repeat(&[b_decoder, 1, 1, 1])?;
        let mask = mask.as_ref().unwrap().repeat(&[b_decoder, 1])?;

        let h_s = self.w_s.forward(&s_true)?; // embedding layer
        let h_es = cat_neighbors_nodes(&h_s, &h_e, &e_idx)?;

        // Build encoder embeddings
        let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
        let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;
        let h_exv_encoder_fw = mask_fw
            .broadcast_as(h_exv_encoder.shape())?
            .to_dtype(h_exv_encoder.dtype())?
            .mul(&h_exv_encoder)?;

        // Apply decoder layers
        let h_v = if !use_sequence {
            // Simple forward pass through decoder layers
            self.decoder_layers.iter().fold(Ok(h_v), |acc, layer| {
                layer.forward(&acc?, &h_exv_encoder_fw, Some(&mask), None, None)
            })?
        } else {
            // Forward pass with sequence-aware processing
            self.decoder_layers.iter().fold(Ok(h_v), |acc, layer| {
                let current_h_v = acc?;
                let h_esv = cat_neighbors_nodes(&current_h_v, &h_es, &e_idx)?
                    .mul(&mask_bw)?
                    .add(&h_exv_encoder_fw)?;
                layer.forward(&current_h_v, &h_esv, Some(&mask), None, None)
            })?
        };

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
