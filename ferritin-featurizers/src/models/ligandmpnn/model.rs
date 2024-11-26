//! A message passing protein design neural network
//! that samples sequences diffusing conditional probabilities.
//!
//!
//! Consider factoring out model creation of the DEC
//! and ENC layers using a function.
//!
//! here is an example of paramatereizable netowrk creation:
//! https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/resnet.rs
//!
use super::configs::{ModelTypes, ProteinMPNNConfig};
use super::featurizer::ProteinFeatures;
use super::proteinfeatures::ProteinFeaturesModel;
use super::utilities::{cat_neighbors_nodes, gather_nodes};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::encoding::one_hot;
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::{embedding, layer_norm, linear, Dropout, Embedding, Linear, VarBuilder};
use candle_transformers::generation::LogitsProcessor;

// Primary Return Object from the ProtMPNN Model
#[derive(Clone, Debug)]
pub struct ScoreOutput {
    // Sequence
    s: Tensor,
    log_probs: Tensor,
    logits: Tensor,
    decoding_order: Tensor,
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
    norm1: layer_norm::LayerNorm,
    norm2: layer_norm::LayerNorm,
    norm3: layer_norm::LayerNorm,
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
        println!("EncLayer: Forward method.");
        let h_ev = cat_neighbors_nodes(h_v, h_e, e_idx)?;
        let h_v_expand = h_v.unsqueeze(D::Minus2)?;
        // Explicitly specify the expansion dimensions
        let expand_shape = [
            h_ev.dims()[0],       // batch size
            h_ev.dims()[1],       // sequence length
            h_ev.dims()[2],       // number of neighbors
            h_v_expand.dims()[3], // hidden dimension
        ];
        let h_v_expand = h_v_expand.expand(&expand_shape)?;
        let h_v_expand = h_v_expand.to_dtype(h_ev.dtype())?;

        // Now concatenate along the last dimension
        let h_ev = Tensor::cat(&[&h_v_expand, &h_ev], D::Minus1)?;
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
        let dh = h_message.sum(D::Minus2)? / self.scale;
        let h_v = {
            let dh_dropout = self
                .dropout1
                .forward(&dh?, training.expect("Training must be specified"))?;
            let dh_dropout = dh_dropout.to_dtype(DType::F32)?;
            let h_v = h_v.to_dtype(DType::F32)?;
            self.norm1.forward(&(h_v + dh_dropout)?)?
        };

        let dh = self.dense.forward(&h_v)?;
        let h_v = {
            let dh_dropout = self
                .dropout2
                .forward(&dh, training.expect("Training Must be specified"))?;
            self.norm2.forward(&(&h_v + &dh_dropout)?)?
        };
        let h_v = if let Some(mask) = mask_v {
            mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_v)?
        } else {
            h_v
        };
        let h_ev = cat_neighbors_nodes(&h_v, h_e, e_idx)?;
        println!("EncLayer: 09");
        // let h_v_expand = h_v.unsqueeze(D::Minus2)?.expand(h_ev.shape().dims())?;
        let h_v_expand = h_v.unsqueeze(D::Minus2)?;
        // Explicitly specify the expansion dimensions
        let expand_shape = [
            h_ev.dims()[0],       // batch size
            h_ev.dims()[1],       // sequence length
            h_ev.dims()[2],       // number of neighbors
            h_v_expand.dims()[3], // hidden dimension
        ];
        let h_v_expand = h_v_expand.expand(&expand_shape)?;
        let h_v_expand = h_v_expand.to_dtype(h_ev.dtype())?;

        println!("EncLayer: 10");
        let h_ev = Tensor::cat(&[&h_v_expand, &h_ev], D::Minus1)?;
        println!("EncLayer: 11");
        let h_message = self
            .w11
            .forward(&h_ev)?
            .gelu()?
            .apply(&self.w12)?
            .gelu()?
            .apply(&self.w13)?;
        let h_e = {
            let h_message_dropout = self
                .dropout3
                .forward(&h_message, training.expect("Training Must be specified"))?;
            self.norm3.forward(&(h_e + h_message_dropout)?)?
        };

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
    norm1: layer_norm::LayerNorm,
    norm2: layer_norm::LayerNorm,
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
        // todo: fix this. hardcoding Training is false
        let training_bool = match training {
            None => false,
            Some(v) => v,
        };

        let expand_shape = [
            h_e.dims()[0], // batch (1)
            h_e.dims()[1], // sequence length (93)
            h_e.dims()[2], // number of neighbors (24)
            h_v.dims()[2], // keep original hidden dim (128)
        ];
        let h_v_expand = h_v.unsqueeze(D::Minus2)?.expand(&expand_shape)?;
        let h_ev = Tensor::cat(&[&h_v_expand, h_e], D::Minus1)?;
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
        let dh = (h_message.sum(D::Minus2)? / self.scale)?;
        let h_v = {
            let dh_dropout = self.dropout1.forward(&dh, training_bool)?;
            self.norm1.forward(&(h_v + dh_dropout)?)?
        };
        let dh = self.dense.forward(&h_v)?;
        let h_v = {
            let dh_dropout = self.dropout2.forward(&dh, training_bool)?;
            self.norm2.forward(&(h_v + dh_dropout)?)?
        };
        let h_v = if let Some(mask) = mask_v {
            mask.unsqueeze(D::Minus1)?.broadcast_mul(&h_v)?
        } else {
            h_v
        };
        Ok(h_v)
    }
}

// https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L10C7-L10C18
pub struct ProteinMPNN {
    config: ProteinMPNNConfig, // device here ??
    decoder_layers: Vec<DecLayer>,
    device: Device,
    encoder_layers: Vec<EncLayer>,
    features: ProteinFeaturesModel, // this needs to be a model with weights etc
    w_e: Linear,
    w_out: Linear,
    // self.W_s = torch.nn.Embedding(vocab, hidden_dim)
    w_s: Embedding, // This should be an embedding layer....
}

impl ProteinMPNN {
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig) -> Result<Self> {
        // Encoder
        let mut encoder_layers = Vec::with_capacity(config.num_encoder_layers as usize);
        for i in 0..config.num_encoder_layers {
            encoder_layers.push(EncLayer::load(vb.pp("encoder_layers"), config, i as i32)?);
        }

        // Decoder
        let mut decoder_layers = Vec::with_capacity(config.num_decoder_layers as usize);
        for i in 0..config.num_decoder_layers {
            decoder_layers.push(DecLayer::load(vb.pp("decoder_layers"), config, i as i32)?);
        }

        // Weights
        let w_e = linear::linear(
            config.edge_features as usize,
            config.hidden_dim as usize,
            vb.pp("W_e"),
        )?;

        let w_out = linear::linear(
            config.hidden_dim as usize,
            config.num_letters as usize,
            vb.pp("W_out"),
        )?;

        let w_s = embedding(
            config.vocab as usize,
            config.hidden_dim as usize,
            vb.pp("W_s"),
        )?;

        // Features
        let features = ProteinFeaturesModel::load(vb.pp("features"), config.clone())?;

        Ok(Self {
            config: config.clone(), // check the clone later...
            decoder_layers,
            device: Device::Cpu,
            encoder_layers,
            features,
            w_e,
            w_out,
            w_s,
        })
    }
    fn predict(&self) {
        // Implement prediction logic
        todo!()
    }
    fn train(&mut self) {
        // Implement training logic
        // .forward()?
        todo!()
    }
    fn encode(&self, features: &ProteinFeatures) -> Result<(Tensor, Tensor, Tensor)> {
        // todo: get device more elegantly
        let device = &candle_core::Device::Cpu;
        let s_true = &features.get_sequence();
        let (b, l) = s_true.dims2()?;

        let mask = match features.get_sequence_mask() {
            Some(m) => m,
            None => &Tensor::zeros_like(&s_true)?,
        };

        match self.config.model_type {
            ModelTypes::ProteinMPNN => {
                let (e, e_idx) = self.features.forward(features, device)?;
                let mut h_v = Tensor::zeros(
                    (e.dim(0)?, e.dim(1)?, e.dim(D::Minus1)?),
                    DType::F64,
                    device,
                )?;
                let mut h_e = self.w_e.forward(&e)?;
                let mask_attend =
                    gather_nodes(&mask.unsqueeze(D::Minus1)?, &e_idx)?.squeeze(D::Minus1)?;

                let mask_expanded = mask.unsqueeze(D::Minus1)?.expand((
                    mask.dim(0)?,
                    mask.dim(1)?,
                    mask_attend.dim(2)?,
                ))?;
                let mask_attend = (&mask_expanded * &mask_attend)?;

                for layer in &self.encoder_layers {
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
    fn sample(&self, features: &ProteinFeatures) -> Result<ScoreOutput> {
        let ProteinFeatures {
            x,
            s,
            x_mask,
            // symmetry_residues,
            // symmetry_weights,
            ..
        } = features;

        // let b_decoder = batch_size.unwrap() as usize;

        let (b, l) = s.shape().dims2()?;
        let device = s.device();
        let s_true = s.clone();

        // Todo: fix chain_mask
        // output_dict.get_chain_mask(vec!['A'.to_string(), 'B'.to_string()], device)?;
        // let chains_to_design = vec!["A".to_string(), "B".to_string()];
        // let chains = self
        //     .chain_letters
        //     .iter()
        //     .map(|item| chains_to_design.contains(item) as u32);
        // let chain_mask = Tensor::from_iter(chains, &device);

        // true and utter hack!
        let chain_mask = Tensor::from_vec(vec![0i64, 0], (2, 1), &device)?;

        // encode...
        let (h_v, h_e, e_idx) = self.encode(features)?;
        let chain_mask = x_mask.as_ref().unwrap().mul(&chain_mask)?; // update chain_M to include missing regions;

        // this might be  a bad rand implementation
        let rand_tensor = Tensor::randn(0., 0.25, (b as usize, l as usize), device)?;
        let decoding_order = (&chain_mask + 0.0001)?;
        let decoding_order = decoding_order.mul(&rand_tensor.abs()?)?;
        let decoding_order = decoding_order.arg_sort_last_dim(false)?;

        // add match statement here
        // I'd like to add the other optional components to the match

        // Todo! Fix this hack.
        let symmetry_residues: Option<Vec<i32>> = None;
        match symmetry_residues {
            None => {
                let e_idx = e_idx.repeat(&[b, 1, 1])?;
                let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1., 0.)?;
                let tril = Tensor::tril2(l, DType::F64, device)?;
                let temp = tril.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; //tensor of shape (b, i, q)
                let order_mask_backward =
                    temp.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // This will give us a tensor of shape (b, q, p)
                let mask_attend = order_mask_backward
                    .gather(&e_idx, 2)?
                    .unsqueeze(D::Minus1)?;
                let mask_1d = x_mask.as_ref().unwrap().reshape((b, l, 1, 1))?;
                let mask_bw = mask_1d.mul(&mask_attend)?;
                let mask_fw = mask_1d.mul(&(Tensor::ones_like(&mask_attend)? - mask_attend)?)?;

                // repeat for decoding
                let s_true = s_true.repeat((b, 1))?;
                let h_v = h_v.repeat((b, 1, 1))?;
                let h_e = h_e.repeat((b, 1, 1, 1))?;
                let chain_mask = &chain_mask.repeat((b, 1))?;
                let mask = x_mask.as_ref().unwrap().repeat((b, 1))?;

                // Todo add  bias
                // let bias = bias.repeat((b_decoder, 1, 1))?;
                let bias = Tensor::zeros((b, l, 20), DType::F32, device)?;
                let all_probs = Tensor::zeros((b, l, 20), DType::F32, device)?;
                let all_log_probs = Tensor::zeros((b, l, 21), DType::F32, device)?; // why is this one 21 and the others are 20?
                let mut h_s = Tensor::zeros_like(&h_v)?;
                let s =
                    Tensor::ones((b, l), DType::I64, device)?.mul(&Tensor::new(20., device)?)?;

                // updated layers are here.
                let mut h_v_stack = vec![h_v.clone()];
                h_v_stack.extend(
                    (0..self.decoder_layers.len()).map(|_| Tensor::zeros_like(&h_v).unwrap()),
                );
                let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
                let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;
                let h_exv_encoder_fw = mask_fw.mul(&h_exv_encoder)?;

                for t_ in 0..l {
                    let t = decoding_order.i((.., t_ as usize))?;
                    let chain_mask_t = chain_mask.gather(&t.unsqueeze(1)?, 1)?.squeeze(1)?;
                    let mask_t = mask.gather(&t.unsqueeze(1)?, 1)?.squeeze(1)?;
                    let bias_t = bias
                        .gather(&t.unsqueeze(1)?.unsqueeze(2)?.repeat((1, 1, 21))?, 1)?
                        .squeeze(1)?;
                    let e_idx_t = e_idx.gather(
                        &t.unsqueeze(1)?
                            .unsqueeze(2)?
                            .repeat((1, 1, e_idx.dim(2)?))?,
                        1,
                    )?;
                    let h_e_t = h_e.gather(
                        &t.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?.repeat((
                            1,
                            1,
                            h_e.dim(2)?,
                            h_e.dim(3)?,
                        ))?,
                        1,
                    )?;
                    let h_es_t = cat_neighbors_nodes(&h_s, &h_e_t, &e_idx_t)?;
                    let h_exv_encoder_t = h_exv_encoder_fw.gather(
                        &t.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?.repeat((
                            1,
                            1,
                            h_exv_encoder_fw.dim(2)?,
                            h_exv_encoder_fw.dim(3)?,
                        ))?,
                        1,
                    )?;
                    let mask_bw_t = mask_bw.gather(
                        &t.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?.repeat((
                            1,
                            1,
                            mask_bw.dim(2)?,
                            mask_bw.dim(3)?,
                        ))?,
                        1,
                    )?;

                    // Todo: Consider factorign this out..
                    for (l, layer) in self.decoder_layers.iter().enumerate() {
                        let h_esv_decoder_t =
                            cat_neighbors_nodes(&h_v_stack[l], &h_es_t, &e_idx_t)?;
                        let h_v_t = h_v_stack[l].gather(
                            &t.unsqueeze(1)?
                                .unsqueeze(2)?
                                .repeat((1, 1, h_v_stack[l].dim(2)?))?,
                            1,
                        )?;
                        let h_esv_t = mask_bw_t.mul(&h_esv_decoder_t)?.add(&h_exv_encoder_t)?;
                        let new_h_v = layer.forward(&h_v_t, &h_esv_t, Some(&mask_t), None, None)?;
                        // note: scatter_add is different between pytorch and candle
                        h_v_stack[l + 1] = h_v_stack[l + 1].scatter_add(
                            &t.unsqueeze(1)?.unsqueeze(2)?.repeat((1, 1, h_v.dim(2)?))?,
                            &new_h_v,
                            1,
                        )?;
                    }
                    let h_v_t = h_v_stack
                        .last()
                        .unwrap()
                        .gather(
                            &t.unsqueeze(1)?.unsqueeze(2)?.repeat((
                                1,
                                1,
                                h_v_stack.last().unwrap().dim(2)?,
                            ))?,
                            1,
                        )?
                        .squeeze(1)?;
                    let logits = self.w_out.forward(&h_v_t)?;
                    let log_probs = log_softmax(&logits, D::Minus1)?;

                    // Todo: Temperature should be added upstream
                    let temperature = 20f64;
                    let probs = softmax(&(logits.add(&bias_t)? / temperature)?, D::Minus1)?;
                    let probs_sample = probs
                        .narrow(1, 0, 20)?
                        .div(&probs.narrow(1, 0, 20)?.sum_keepdim(1)?)?;

                    // pytorch direct translation
                    // let s_t = probs_sample.multinomial(1, true)?.squeeze(1)?;
                    //
                    // note: this sampling is not the same as pytorch's.
                    // https://github.com/huggingface/candle/blob/dcd83336b68049763973709733bf2721a687507d/candle-transformers/src/generation/mod.rs#L47
                    // this sample should probably be brought up higher for reuse
                    // Todo: this may ormay not be the same as pytorch multinomial
                    let seed = 2;
                    let mut logproc = LogitsProcessor::new(seed, Some(temperature), Some(0.25));
                    let logits: Vec<u32> = vec![(); l]
                        .iter()
                        .map(|_| logproc.sample(&probs_sample))
                        .filter_map(Result::ok)
                        .collect();
                    // Todo: this definitely needs to be checked for Dimensions
                    let s_t = Tensor::from_vec(logits, l, device)?;

                    // note: need to double check pytorch vs candle
                    let all_probs = all_probs.scatter_add(
                        &t.unsqueeze(1)?.unsqueeze(2)?.repeat((1, 1, 20))?,
                        &(chain_mask_t
                            .unsqueeze(1)?
                            .unsqueeze(2)?
                            .mul(&probs_sample.unsqueeze(1)?))?,
                        1,
                    )?;

                    // these need to mutate out of scope - e.g. to the top-level cvar
                    let all_log_probs = all_log_probs.scatter_add(
                        &t.unsqueeze(1)?.unsqueeze(2)?.repeat((1, 1, 21))?,
                        &(chain_mask_t
                            .unsqueeze(1)?
                            .unsqueeze(2)?
                            .mul(&log_probs.unsqueeze(1)?))?,
                        1,
                    )?;

                    let s_true_t = s_true.gather(&t.unsqueeze(1)?, 1)?.squeeze(1)?;
                    let s_t = s_t
                        .mul(&chain_mask_t)?
                        .add(&s_true_t.mul(&(Tensor::ones_like(&chain_mask_t)? - chain_mask_t)?)?)?
                        .to_dtype(candle_core::DType::I64)?;

                    h_s = h_s.scatter_add(
                        &t.unsqueeze(1)?.unsqueeze(2)?.repeat((1, 1, h_s.dim(2)?))?,
                        &self.w_s.forward(&s_t)?.unsqueeze(1)?,
                        1,
                    )?;

                    // the below line approach may note be correct.
                    let zeros = Tensor::zeros_like(&s)?;
                    let scattered = zeros.scatter_add(
                        &t.unsqueeze(1)?,
                        &s_t.unsqueeze(1)?,
                        1, // Assuming you're scattering along dimension 1
                    )?;
                    let s = s.add(&scattered)?;
                }

                Ok(ScoreOutput {
                    s,
                    log_probs: all_probs, // needs a fix - currently these don't get updated
                    logits: all_log_probs, // needs a fix - currently these don't get updated
                    decoding_order,
                })
            }
            Some(symmetry_residues) => {
                todo!()
                // // note this is a literal translation of the code... Howver I think this could lead to
                // // possible unintentional overwritign of value - e.g. if there are multiple identical
                // // values in the index. (I guess you might expect that if they are symetrical. Howver the
                // // weigths do no have to be thesame)
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

    // fn single_aa_score(
    //     &self,
    //     feature_dict: LigandMPNNData,
    //     use_sequence: bool,
    // ) -> Result<ScoreOutput> {
    //     let LigandMPNNData {
    //         output_dict,
    //         batch_size,
    //         ..
    //     } = feature_dict;

    //     let LigandMPNNDataDict {
    //         x,
    //         s,
    //         mask,
    //         // chain_mask,
    //         // bias,
    //         // randn,
    //         // temperature,
    //         // symmetry_residues,
    //         // symmetry_weights
    //         ..
    //     } = output_dict;

    //     let b_decoder = batch_size;
    //     let s_true_enc = s;
    //     let mask_enc = mask;
    //     let chain_mask_enc = output_dict.get_chain_mask(vec!['A', 'B'], device);
    //     let (b, l) = s_true_enc.shape().dims2()?;
    //     let device = s_true_enc.device();

    //     let (h_v_enc, h_e_enc, e_idx_enc) = self.encode(feature_dict)?;

    //     let mut log_probs_out =
    //         Tensor::zeros(&[b_decoder, l, 21], candle_core::DType::F32, device)?;
    //     let mut logits_out = Tensor::zeros(&[b_decoder, l, 21], candle_core::DType::F32, device)?;
    //     let mut decoding_order_out =
    //         Tensor::zeros(&[b_decoder, l, l], candle_core::DType::F32, device)?;

    //     for idx in 0..l {
    //         let mut h_v = h_v_enc.clone();
    //         let mut e_idx = e_idx_enc.clone();
    //         let mut mask = mask_enc.clone();
    //         let mut s_true = s_true_enc.clone();

    //         let order_mask = if !use_sequence {
    //             let mut mask =
    //                 Tensor::zeros((chain_mask_enc.dim(1)?,), candle_core::DType::F32, device)?;
    //             mask.i(idx)?.set(&Tensor::from_slice(&[1.0], device)?)?;
    //             mask
    //         } else {
    //             let mut mask =
    //                 Tensor::ones((chain_mask_enc.dim(1)?,), candle_core::DType::F32, device)?;
    //             mask.i(idx)?.set(&Tensor::from_slice(&[0.0], device)?)?;
    //             mask
    //         };

    //         let decoding_order = ((order_mask.add_scalar(0.0001)?)?.mul(&randn.abs())?)?
    //             .argsort(candle_core::D::Minus1, true)?;

    //         e_idx = e_idx.repeat((b_decoder, 1, 1))?;
    //         let permutation_matrix_reverse =
    //             ops::one_hot(&decoding_order, l)?.to_dtype(candle_core::DType::F32)?;
    //         let tril = Tensor::tril2(l, DType::F64, device);
    //         // First, perform the matrix multiplication between the lower triangle and the first permutation matrix
    //         // This will give us a tensor of shape (b, i, q)
    //         let temp = tril.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?;
    //         // Now perform the matrix multiplication between the result and the second permutation matrix
    //         let order_mask_backward = temp.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // This will give us a tensor of shape (b, q, p)
    //         let mask_attend = order_mask_backward
    //             .gather(&e_idx, 2)?
    //             .unsqueeze(D::Minus1)?;

    //         let mask_1d = mask.reshape((b, l, 1, 1))?;
    //         let mask_bw = mask_1d.mul(&mask_attend)?;
    //         let mask_fw = mask_1d.mul(&(Tensor::ones_like(&mask_attend)? - mask_attend)?)?;
    //         s_true = s_true.repeat(&[b_decoder, 1])?;
    //         h_v = h_v.repeat(&[b_decoder, 1, 1])?;
    //         let h_e = h_e_enc.repeat(&[b_decoder, 1, 1, 1])?;
    //         mask = mask.repeat(&[b_decoder, 1])?;
    //         let h_s = self.w_s.forward(&s_true)?;
    //         let h_es = cat_neighbors_nodes(&h_s, &h_e, &e_idx)?;
    //         let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
    //         let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;

    //         let h_exv_encoder_fw = mask_fw.mul(&h_exv_encoder)?;

    //         for layer in &self.decoder_layers {
    //             let h_esv = cat_neighbors_nodes(&h_v, &h_es, &e_idx)?;
    //             let h_esv = mask_bw.mul(&h_esv)?.add(&h_exv_encoder_fw)?;
    //             h_v = layer.forward(&h_v, &h_esv, Some(&mask))?;
    //         }

    //         let logits = self.w_out.forward(&h_v)?;
    //         let log_probs = ops::log_softmax(&logits, D::Minus1)?;

    //         log_probs_out
    //             .narrow_mut(1, idx as i64, 1)?
    //             .copy_(&log_probs.narrow(1, idx as i64, 1)?)?;
    //         logits_out
    //             .narrow_mut(1, idx as i64, 1)?
    //             .copy_(&logits.narrow(1, idx as i64, 1)?)?;
    //         decoding_order_out
    //             .narrow_mut(1, idx as i64, 1)?
    //             .copy_(&decoding_order.unsqueeze(1)?)?;
    //     }

    //     Ok(ScoreOutput {
    //         s: s_true_enc,
    //         log_probs: log_probs_out,
    //         logits: logits_out,
    //         decoding_order: decoding_order_out,
    //     })
    // }

    pub fn score(&self, features: &ProteinFeatures, use_sequence: bool) -> Result<ScoreOutput> {
        let ProteinFeatures { s, x, x_mask, .. } = &features;
        let s_true = &s.clone();
        let mask = &x_mask.as_ref().clone();
        let (b, l) = s_true.dims2()?;
        let b_decoder = b;
        let device = s_true.device();
        let randn = Tensor::randn(0., 1., (b, l), device)?.to_dtype(DType::F32)?;
        let (h_v, h_e, e_idx) = self.encode(features)?;

        // Todo! This is a hack. we shou ldbe passing in encoded chains.
        let chain_mask = Tensor::zeros_like(mask.unwrap())?.to_dtype(DType::F32)?;

        // Update chain_mask to include missing regions
        let chain_mask = mask.unwrap().mul(&chain_mask)?;

        // Compute decoding order
        let decoding_order = (chain_mask + 0.001)?
            .mul(&randn.abs()?)?
            .arg_sort_last_dim(false)?;

        let symmetry_residues: Option<Vec<i32>> = None;

        let (mask_fw, mask_bw, e_idx, decoding_order) = match symmetry_residues {
            // Note: I lifted this code form above. I didn't look to see if they are 100pct identical.
            // If they ARE then I will want to refactor to a score fn that can be used in a few places.
            Some(symmetry_residues) => {
                todo!();
                // let symmetry_weights = symmetry_weights.as_ref().unwrap();
                // let mut symmetry_weights_vec = vec![1.0_f64; l];
                // for (i1, item_list) in symmetry_residues.iter().enumerate() {
                //     for (i2, &item) in item_list.iter().enumerate() {
                //         let value = symmetry_weights[i1][i2];
                //         symmetry_weights_vec[item as usize] = value;
                //     }
                // }
                // let symmetry_weights_tensor = Tensor::from_vec(symmetry_weights_vec, l, device)?;
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
                // let flattened_order: Vec<i64> = new_decoding_order.into_iter().flatten().collect();
                // let decoding_order = Tensor::from_vec(flattened_order, l, device)?
                //     .unsqueeze(0)?
                //     .repeat((b, 1))?;

                // let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1., 0.)?; // need to double check here
                // let tril = Tensor::tril2(l, DType::F64, device)?;

                // // First, perform the matrix multiplication between the lower triangle and the first permutation matrix
                // // This will give us a tensor of shape (b, i, q)
                // let temp = tril.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?;
                // // Now perform the matrix multiplication between the result and the second permutation matrix
                // let order_mask_backward =
                //     temp.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // This will give us a tensor of shape (b, q, p)
                // let e_idx = Tensor::arange(0, l as i64, device)?
                //     .unsqueeze(0)?
                //     .unsqueeze(0)?
                //     .repeat((b, l, 1))?;

                // let mask_attend = order_mask_backward
                //     .gather(&e_idx, 2)?
                //     .unsqueeze(D::Minus1)?;
                // let mask = Tensor::ones((b, l), DType::F64, device)?;
                // let mask_1d = mask.reshape((b, l, 1, 1))?;
                // let mask_bw = mask_1d.broadcast_mul(&mask_attend)?;
                // let mask_fw =
                //     mask_1d.broadcast_mul(&(Tensor::ones_like(&mask_attend)? - mask_attend)?)?;

                // (mask_fw, mask_bw, e_idx, decoding_order)
            }
            None => {
                let b_decoder = b_decoder as usize;
                let e_idx = e_idx.repeat(&[b_decoder, 1, 1])?;
                let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1., 0.)?;
                let tril = Tensor::tril2(l, DType::F64, device)?;
                let tril = tril.unsqueeze(0)?;
                let temp = tril.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // shape (b, i, q)
                let order_mask_backward =
                    temp.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // shape (b, q, p)
                let mask_attend = order_mask_backward
                    .gather(&e_idx, 2)?
                    .unsqueeze(D::Minus1)?;
                let mask_1d = mask.unwrap().reshape((b, l, 1, 1))?;
                // Broadcast mask_1d to match mask_attend's shape
                let mask_1d = mask_1d
                    .broadcast_as(mask_attend.shape())?
                    .to_dtype(DType::F64)?;
                let mask_bw = mask_1d.mul(&mask_attend)?;
                let mask_fw = mask_1d.mul(&(mask_attend - 1.0)?.neg()?)?;
                (mask_fw, mask_bw, e_idx, decoding_order)
            }
        };
        let b_decoder = b_decoder as usize;
        let s_true = s_true.repeat(&[b_decoder, 1])?;
        let h_v = h_v.repeat(&[b_decoder, 1, 1])?;
        let h_e = h_e.repeat(&[b_decoder, 1, 1, 1])?;
        let mask = x_mask.as_ref().unwrap().repeat(&[b_decoder, 1])?;
        let h_s = self.w_s.forward(&s_true)?; // embedding layer
        let h_es = cat_neighbors_nodes(&h_s, &h_e, &e_idx)?;
        // Build encoder embeddings
        let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
        let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;
        let mask_fw = mask_fw
            .broadcast_as(h_exv_encoder.shape())?
            .to_dtype(h_exv_encoder.dtype())?;
        let h_exv_encoder_fw = mask_fw.mul(&h_exv_encoder)?;
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
