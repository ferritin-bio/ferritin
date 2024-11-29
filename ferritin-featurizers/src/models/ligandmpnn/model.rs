//! A message passing protein design neural network
//! that samples sequences diffusing conditional probabilities.
//!
//!
//! Consider factoring out model creation of the DEC
//! and ENC layers using a function.
//!
//! here is an example of paramatereizable network creation:
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
use candle_transformers::generation::{LogitsProcessor, Sampling};

pub fn multinomial_sample(probs: &Tensor, temperature: f64, seed: u64) -> Result<Tensor> {
    // Create the logits processor with its required arguments
    let mut logits_processor = LogitsProcessor::new(
        seed,              // seed for reproducibility
        Some(temperature), // temperature scaling
        None,              // top_p (nucleus sampling), we don't need this
    );

    // Sample from the probabilities
    let idx = logits_processor.sample(&probs)?;

    // Convert to tensor
    Tensor::new(&[idx], probs.device())
}

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
        println!("EncoderLayer: Starting forward pass");

        // Initial values - use get(0) to get first batch, narrow for first few values
        println!("Input h_v dims: {:?}", h_v.dims());
        println!(
            "Input h_v first values: {:?}",
            h_v.to_dtype(DType::F32)?
                .get(0)?
                .narrow(0, 0, 1)? // Get first sequence position
                .narrow(1, 0, 5)? // Get first 5 features
                .to_vec2::<f32>()?
        );

        let h_ev = cat_neighbors_nodes(h_v, h_e, e_idx)?;
        println!("After cat_neighbors_nodes h_ev dims: {:?}", h_ev.dims());
        println!(
            "h_ev first values: {:?}",
            h_ev.to_dtype(DType::F32)?
                .get(0)?
                .narrow(0, 0, 1)? // First sequence position
                .narrow(1, 0, 5)? // First 5 neighbors
                .narrow(2, 0, 5)? // First 5 features
                .to_vec3::<f32>()?
        );

        let h_v_expand = h_v.unsqueeze(D::Minus2)?;
        // Explicitly specify the expansion dimensions
        let expand_shape = [
            h_ev.dims()[0],       // batch size
            h_ev.dims()[1],       // sequence length
            h_ev.dims()[2],       // number of neighbors
            h_v_expand.dims()[3], // hidden dimension
        ];
        let h_v_expand = h_v_expand.expand(&expand_shape)?.to_dtype(h_ev.dtype())?;
        println!("h_v_expand dims: {:?}", h_v_expand.dims());
        println!(
            "h_v_expand first values: {:?}",
            h_v_expand
                .to_dtype(DType::F32)?
                .get(0)?
                .narrow(0, 0, 1)? // First sequence position
                .narrow(1, 0, 5)? // First 5 neighbors
                .narrow(2, 0, 5)? // First 5 features
                .to_vec3::<f32>()?
        );

        // Now concatenate along the last dimension
        let h_ev = Tensor::cat(&[&h_v_expand, &h_ev], D::Minus1)?;

        println!("After concat h_ev dims: {:?}", h_ev.dims());
        println!(
            "After concat first values: {:?}",
            h_ev.to_dtype(DType::F32)?
                .get(0)?
                .narrow(0, 0, 1)? // First sequence position
                .narrow(1, 0, 5)? // First 5 neighbors
                .narrow(2, 0, 5)? // First 5 features
                .to_vec3::<f32>()?
        );

        let h_message = self.w1.forward(&h_ev)?;
        println!(
            "After w1 max: {:?}",
            h_message.max(D::Minus1)?.to_vec3::<f32>()?
        );
        let h_message = h_message.clamp(-20.0, 20.0)?; // Clip after w1

        let h_message = h_message.gelu()?;
        println!(
            "After gelu max: {:?}",
            h_message.max(D::Minus1)?.to_vec3::<f32>()?
        );

        let h_message = h_message.apply(&self.w2)?;
        println!(
            "After w2 max: {:?}",
            h_message.max(D::Minus1)?.to_vec3::<f32>()?
        );
        let h_message = h_message.clamp(-20.0, 20.0)?; // Clip after w2

        let h_message = h_message.gelu()?;
        println!(
            "After second gelu max: {:?}",
            h_message.max(D::Minus1)?.to_vec3::<f32>()?
        );

        let h_message = h_message.apply(&self.w3)?;
        println!(
            "After w3 max: {:?}",
            h_message.max(D::Minus1)?.to_vec3::<f32>()?
        );
        let h_message = h_message.clamp(-20.0, 20.0)?; // Clip after w3

        let h_message = if let Some(mask) = mask_attend {
            let mask = mask.unsqueeze(D::Minus1)?;
            println!("ENCODER_LAYER: Mask dims: {:?}", mask.dims());
            println!("ENCODER_LAYER: h_message dims: {:?}", h_message.dims());
            println!(
                "ENCODER_LAYER: Mask first values: {:?}",
                mask.get(0)?
                    .narrow(0, 0, 1)?
                    .narrow(1, 0, 5)?
                    .to_vec3::<f32>()?
            );

            println!("ENCODER_LAYER: Before broadcast_mul:");
            println!("mask shape: {:?}", mask.dims());
            println!("h_message shape: {:?}", h_message.dims());

            let result = mask.broadcast_mul(&h_message)?;
            println!(
                "ENCODER_LAYER: After broadcast_mul shape: {:?}",
                result.dims()
            );

            result
        } else {
            h_message
        };

        // Safe division with scale
        println!("Scale value: {:?}", self.scale);
        let dh = {
            let sum = h_message.sum(D::Minus2)?;
            let scale = if self.scale == 0.0 { 1.0 } else { self.scale };
            (sum / scale)?
        };
        println!(
            "After division dh max: {:?}",
            dh.max(D::Minus1)?.to_vec2::<f32>()?
        );

        let h_v = {
            let dh_dropout = self
                .dropout1
                .forward(&dh, training.expect("Training must be specified"))?;
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
        let h_ev = Tensor::cat(&[&h_v_expand, &h_ev], D::Minus1)?;
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
        println!("EncoderLayer: Finishing forward pass");
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
    config: ProteinMPNNConfig,
    decoder_layers: Vec<DecLayer>,
    device: Device,
    encoder_layers: Vec<EncLayer>,
    features: ProteinFeaturesModel,
    w_e: Linear,
    w_out: Linear,
    w_s: Embedding,
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
            config: config.clone(), // todo: check the\is clone later...
            decoder_layers,
            device: Device::Cpu,
            encoder_layers,
            features,
            w_e,
            w_out,
            w_s,
        })
    }
    // fn predict(&self) {
    //     // Implement prediction logic
    //     todo!()
    // }
    // fn train(&mut self) {
    //     // Implement training logic
    //     // .forward()?
    //     todo!()
    // }
    fn encode(&self, features: &ProteinFeatures) -> Result<(Tensor, Tensor, Tensor)> {
        let device = &Device::Cpu; // todo: get device more elegantly
        let s_true = &features.get_sequence();

        // needed for the MaskAttend
        let mask = match features.get_sequence_mask() {
            Some(m) => m,
            None => &Tensor::ones_like(&s_true)?,
        };

        println!("Starting encode function");

        match self.config.model_type {
            ModelTypes::ProteinMPNN => {
                let (e, e_idx) = self.features.forward(features, device)?;
                println!("After embedding dims: {:?}", e.dims());

                let mut h_v = Tensor::zeros(
                    (e.dim(0)?, e.dim(1)?, e.dim(D::Minus1)?),
                    DType::F64,
                    device,
                )?;

                let mut h_e = self.w_e.forward(&e)?;

                let mask_attend = if let Some(mask) = features.get_sequence_mask() {
                    println!("Original mask dims: {:?}", mask.dims());
                    println!(
                        "Original mask values: {:?}",
                        mask.get(0)?.narrow(0, 0, 5)?.to_vec1::<f32>()?
                    );

                    // First unsqueeze mask
                    let mask_expanded = mask.unsqueeze(D::Minus1)?; // [B, L, 1]
                    println!(
                        "Expanded mask values: {:?}",
                        mask_expanded.get(0)?.narrow(0, 0, 5)?.to_vec2::<f32>()?
                    );

                    // Gather using E_idx
                    let mask_gathered = gather_nodes(&mask_expanded, &e_idx)?;
                    println!("Gathered mask dims: {:?}", mask_gathered.dims());
                    println!(
                        "Gathered mask values: {:?}",
                        mask_gathered
                            .get(0)?
                            .narrow(0, 0, 5)?
                            .narrow(1, 0, 5)?
                            .to_vec3::<f32>()?
                    );

                    let mask_gathered = mask_gathered.squeeze(D::Minus1)?;

                    // Multiply original mask with gathered mask
                    let mask_attend = {
                        let mask_unsqueezed = mask.unsqueeze(D::Minus1)?; // [B, L, 1]
                        println!("mask_unsqueezed dims: {:?}", mask_unsqueezed.dims());

                        // Explicitly expand mask_unsqueezed to match mask_gathered dimensions
                        let mask_expanded = mask_unsqueezed
                            .expand((
                                mask_gathered.dim(0)?, // batch
                                mask_gathered.dim(1)?, // sequence length
                                mask_gathered.dim(2)?, // number of neighbors
                            ))?
                            .contiguous()?;
                        println!("mask_expanded dims: {:?}", mask_expanded.dims());

                        // Now do the multiplication with explicit shapes
                        mask_expanded.mul(&mask_gathered)?
                    };
                    mask_attend
                } else {
                    let (b, l) = mask.dims2()?;
                    let ones = Tensor::ones((b, l, e_idx.dim(2)?), DType::F32, device)?;
                    println!("Created default ones mask dims: {:?}", ones.dims());
                    println!(
                        "Created default ones mask values: {:?}",
                        ones.get(0)?
                            .narrow(0, 0, 5)?
                            .narrow(1, 0, 5)?
                            .to_vec2::<f32>()?
                    );
                    ones
                };

                for (i, layer) in self.encoder_layers.iter().enumerate() {
                    println!("Starting encoder layer {}", i);

                    // Debug h_v (3D tensor)
                    println!("h_v before layer {} dims: {:?}", i, h_v.dims());
                    let h_v_f32 = h_v.to_dtype(DType::F32)?;
                    println!(
                        "h_v before layer {} values: {:?}",
                        i,
                        h_v_f32.to_vec3::<f32>()?
                    );

                    // Debug h_e (4D tensor) - access first batch and first sequence position
                    println!("h_e before layer {} dims: {:?}", i, h_e.dims());
                    let h_e_f32 = h_e.to_dtype(DType::F32)?;
                    println!(
                        "h_e before layer {} first position values: {:?}",
                        i,
                        h_e_f32.get(0)?.get(0)?.to_vec2::<f32>()?
                    );

                    let (new_h_v, new_h_e) = layer.forward(
                        &h_v,
                        &h_e,
                        &e_idx,
                        Some(&mask),
                        Some(&mask_attend),
                        Some(false),
                    )?;
                    println!("After layer {} forward pass:", i);

                    // Debug new_h_v
                    println!("new_h_v dims: {:?}", new_h_v.dims());
                    let new_h_v_f32 = new_h_v.to_dtype(DType::F32)?;
                    println!("new_h_v values: {:?}", new_h_v_f32.to_vec3::<f32>()?);

                    // Debug new_h_e
                    println!("new_h_e dims: {:?}", new_h_e.dims());
                    let new_h_e_f32 = new_h_e.to_dtype(DType::F32)?;
                    println!(
                        "new_h_e first position values: {:?}",
                        new_h_e_f32.get(0)?.get(0)?.to_vec2::<f32>()?
                    );

                    h_v = new_h_v;
                    h_e = new_h_e;
                }
                println!("Final h_v dims: {:?}", h_v.dims());
                println!("Final h_e dims: {:?}", h_e.dims());
                println!("Final e_idx dims: {:?}", e_idx.dims());
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
    pub fn sample(&self, features: &ProteinFeatures) -> Result<ScoreOutput> {
        let ProteinFeatures {
            x,
            s,
            x_mask,
            // symmetry_residues,
            // symmetry_weights,
            ..
        } = features;

        // let b_decoder = batch_size.unwrap() as usize;
        let s_true = s.clone();
        let device = s.device();
        let (b, l) = s.dims2()?;

        // Todo: This is a hack. we should be passing in encoded chains.
        let chain_mask = Tensor::ones_like(&x_mask.as_ref().unwrap())?.to_dtype(DType::F32)?;
        let chain_mask = x_mask.as_ref().unwrap().mul(&chain_mask)?.contiguous()?; // update chain_M to include missing regions;

        // encode...
        let (h_v, h_e, e_idx) = self.encode(features)?;
        println!("h_v before repeat dims: {:?}", h_v.dims());
        println!("h_e before repeat dims: {:?}", h_e.dims());
        println!("h_v before repeat values: {:?}", h_v.to_vec3::<f32>()?);
        println!("h_e before repeat values: {:?}", h_e.to_vec3::<f32>()?);

        // this might be  a bad rand implementation
        let rand_tensor = Tensor::randn(0., 0.25, (b, l), device)?.to_dtype(DType::F32)?;
        let decoding_order = (&chain_mask + 0.0001)?
            .mul(&rand_tensor.abs()?)?
            .arg_sort_last_dim(false)?;

        println!("Decoding Order: {:?}", decoding_order.dims());

        // Todo add  bias
        // # [B,L,21] - amino acid bias per position
        //  bias = feature_dict["bias"]
        let bias = Tensor::ones((b, l, 21), DType::F32, device)?;
        println!("todo: We need to add the bias!");

        // Todo! Fix this hack.
        println!("todo: move temp and seed upstream");
        let temperature = 1.0f64;
        let seed = 111;
        let symmetry_residues: Option<Vec<i32>> = None;
        match symmetry_residues {
            None => {
                let e_idx = e_idx.repeat(&[b, 1, 1])?.contiguous()?;
                let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1., 0.)?;
                let tril = Tensor::tril2(l, DType::F64, device)?;
                let tril = tril.unsqueeze(0)?;
                let temp = tril.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; //tensor of shape (b, i, q)
                let order_mask_backward =
                    temp.matmul(&permutation_matrix_reverse.transpose(1, 2)?)?; // This will give us a tensor of shape (b, q, p)
                let mask_attend = order_mask_backward
                    .gather(&e_idx, 2)?
                    .unsqueeze(D::Minus1)?;
                let mask_1d = x_mask.as_ref().unwrap().reshape((b, l, 1, 1))?;

                // Broadcast mask_1d to match mask_attend's shape
                let mask_1d = mask_1d
                    .broadcast_as(mask_attend.shape())?
                    .to_dtype(DType::F64)?;

                let mask_bw = mask_1d.mul(&mask_attend)?;
                let mask_fw = mask_1d.mul(&(Tensor::ones_like(&mask_attend)? - mask_attend)?)?;

                // Note: `sample` begins to diverge from the `score` here.
                // repeat for decoding
                let s_true = s_true.repeat((b, 1))?;
                let h_v = h_v.repeat((b, 1, 1))?;
                let h_e = h_e.repeat((b, 1, 1, 1))?;

                let chain_mask = &chain_mask.repeat((b, 1))?.contiguous()?;
                println!("INIT: chain_mask dims: {:?}", chain_mask.dims());
                println!(
                    "INIT: chain_mask first values: {:?}",
                    chain_mask.get(0)?.narrow(0, 0, 5)?.to_vec1::<f32>()?
                );

                let mask = x_mask.as_ref().unwrap().repeat((b, 1))?.contiguous()?;
                let bias = bias.repeat((b, 1, 1))?.contiguous()?;
                println!("INIT: bias dims: {:?}", bias.dims());
                println!(
                    "INIT: bias first values: {:?}",
                    bias.get(0)?.get(0)?.narrow(0, 0, 5)?.to_vec1::<f32>()?
                );

                let mut all_probs = Tensor::zeros((b, l, 20), DType::F32, device)?;
                let mut all_log_probs = Tensor::zeros((b, l, 21), DType::F32, device)?; // why is this one 21 and the others are 20?
                let mut h_s = Tensor::zeros_like(&h_v)?.contiguous()?;
                let s = (Tensor::ones((b, l), DType::I64, device)? * 20.)?;

                // updated layers are here.
                println!("Initial h_v dims: {:?}", h_v.dims());
                println!("Initial h_v values: {:?}", h_v.to_vec3::<f32>()?);

                let mut h_v_stack = vec![h_v.clone()];
                println!("h_v_stack[0] dims: {:?}", h_v_stack[0].dims());
                println!("h_v_stack[0] values: {:?}", h_v_stack[0].to_vec3::<f32>()?);

                for i in 0..self.decoder_layers.len() {
                    let zeros = Tensor::zeros_like(&h_v)?;
                    println!("zeros layer {} dims: {:?}", i, zeros.dims());
                    println!(
                        "zeros layer {} first few values: {:?}",
                        i,
                        zeros.narrow(2, 0, 5)?.to_vec3::<f32>()?
                    );
                    h_v_stack.push(zeros);
                    // h_v_stack.push(Tensor::zeros_like(&h_v)?);
                }
                let h_ex_encoder = cat_neighbors_nodes(&Tensor::zeros_like(&h_s)?, &h_e, &e_idx)?;
                let h_exv_encoder = cat_neighbors_nodes(&h_v, &h_ex_encoder, &e_idx)?;

                let mask_fw = mask_fw
                    .broadcast_as(h_exv_encoder.shape())?
                    .to_dtype(h_exv_encoder.dtype())?;
                let h_exv_encoder_fw = mask_fw.mul(&h_exv_encoder)?;

                println!("Prepare the decoding Tensors....");
                for t_ in 0..l {
                    let t = decoding_order.i((.., t_))?;
                    let t_gather = t.unsqueeze(1)?; // Shape [B, 1]

                    // Gather masks and bias
                    let chain_mask_t = chain_mask.gather(&t_gather, 1)?.squeeze(1)?;
                    let mask_t = mask.gather(&t_gather, 1)?.squeeze(1)?;
                    let bias_t = bias
                        .gather(&t_gather.unsqueeze(2)?.expand((b, 1, 21))?.contiguous()?, 1)?
                        .squeeze(1)?;

                    // Gather edge and node indices/features
                    let e_idx_t = e_idx.gather(
                        &t_gather
                            .unsqueeze(2)?
                            .expand((b, 1, e_idx.dim(2)?))?
                            .contiguous()?,
                        1,
                    )?;

                    println!("Test 05");
                    let h_e_t = h_e.gather(
                        &t_gather
                            .unsqueeze(2)?
                            .unsqueeze(3)?
                            .expand((b, 1, h_e.dim(2)?, h_e.dim(3)?))?
                            .contiguous()?,
                        1,
                    )?;

                    println!("h_s dims: {:?}", h_s.dims());
                    println!("h_e_t dims: {:?}", h_e_t.dims());
                    println!("e_idx_t dims: {:?}", e_idx_t.dims());

                    let b = h_s.dim(0)?; // batch size
                    let l = h_s.dim(1)?; // sequence length
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

                    println!("Test 09");
                    // Decoder layers loop
                    for l in 0..self.decoder_layers.len() {
                        println!("X 01");
                        let h_v_stack_l = &h_v_stack[l];

                        println!("Layer {}", l);
                        println!("h_v_stack_l dims: {:?}", h_v_stack_l.dims());
                        println!("h_v_stack_l values: {:?}", h_v_stack_l.to_vec3::<f32>()?);

                        println!("X 02");
                        let h_esv_decoder_t = cat_neighbors_nodes(h_v_stack_l, &h_es_t, &e_idx_t)?;
                        println!("h_esv_decoder_t dims: {:?}", h_esv_decoder_t.dims());
                        // println!(
                        //     "h_esv_decoder_t first few values: {:?}",
                        //     h_esv_decoder_t.slice_along(2, 0..5)?.to_vec3::<f32>()?
                        // );

                        println!("X 03");
                        let h_v_t = h_v_stack_l.gather(
                            &t_gather
                                .unsqueeze(2)?
                                .expand((b, 1, h_v_stack_l.dim(2)?))?
                                .contiguous()?,
                            1,
                        )?;
                        println!("h_v_t dims: {:?}", h_v_t.dims());
                        println!("h_v_t values: {:?}", h_v_t.to_vec3::<f32>()?);

                        let mask_bw_t = mask_bw_t.expand(h_esv_decoder_t.dims())?.contiguous()?;

                        let h_exv_encoder_t = h_exv_encoder_t
                            .expand(h_esv_decoder_t.dims())?
                            .contiguous()?
                            .to_dtype(DType::F64)?;

                        let h_esv_t = mask_bw_t
                            .mul(&h_esv_decoder_t.to_dtype(DType::F64)?)?
                            .add(&h_exv_encoder_t)?
                            .to_dtype(DType::F32)?;

                        println!("h_esv_t dims: {:?}", h_esv_t.dims());
                        // println!(
                        //     "h_esv_t first few values: {:?}",
                        //     h_esv_t.slice_along(2, 0..5)?.to_vec3::<f32>()?
                        // );

                        let h_v_t = h_v_t
                            .expand((
                                h_esv_t.dim(0)?, // batch size
                                h_esv_t.dim(1)?, // sequence length (93)
                                h_v_t.dim(2)?,   // features (128)
                            ))?
                            .contiguous()?;

                        // Update h_v_stack[l + 1]
                        let new_h_v = self.decoder_layers[l].forward(
                            &h_v_t,
                            &h_esv_t,
                            Some(&mask_t),
                            None,
                            None,
                        )?;

                        println!("new_h_v dims: {:?}", new_h_v.dims());
                        println!("new_h_v values: {:?}", new_h_v.to_vec3::<f32>()?);

                        // Create gather indices matching PyTorch pattern
                        let gather_indices = t_gather
                            .unsqueeze(2)? // Like [:, None, None]
                            .expand((t_gather.dim(0)?, t_gather.dim(1)?, h_v.dim(2)?))? // Like repeat(1, 1, h_V.shape[-1])
                            .contiguous()?;

                        // Gather operation
                        let new_h_v_t = new_h_v.gather(&gather_indices, 1)?;

                        // Use same indices for scatter
                        h_v_stack[l + 1] =
                            h_v_stack[l + 1].scatter_add(&gather_indices, &new_h_v_t, 1)?;
                    }

                    println!("Test 10");
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

                    println!("h_v_t dims: {:?}", h_v_t.dims());
                    println!("h_v_t values: {:?}", h_v_t.to_vec2::<f32>()?);

                    println!("Test 11");
                    // Generate logits and probabilities
                    let logits = self.w_out.forward(&h_v_t)?;
                    println!("Test 12");
                    let log_probs = log_softmax(&logits, D::Minus1)?;
                    println!("Test 13");
                    // Generate probabilities and sample
                    let probs = softmax(&(logits.add(&bias_t)? / temperature)?, D::Minus1)?;
                    println!("Test 14");
                    let probs_sample = probs
                        .narrow(1, 0, 20)?
                        .div(&probs.narrow(1, 0, 20)?.sum_keepdim(1)?.expand((b, 20))?)?;

                    // Let's add debug prints:
                    println!("logits: {:?}", logits.to_vec2::<f32>()?);
                    println!("bias_t: {:?}", bias_t.to_vec2::<f32>()?);
                    println!("probs after softmax: {:?}", probs.to_vec2::<f32>()?);
                    println!(
                        "probs_sample before sampling: {:?}",
                        probs_sample.to_vec2::<f32>()?
                    );

                    // Sample new token
                    println!("Test 15");
                    println!("probs_sample dims: {:?}", probs_sample.dims());
                    println!("probs_sample: {:?}", probs_sample.to_vec2::<f32>()?);

                    let probs_sample_1d = probs_sample
                        .squeeze(0)? // Remove batch dimension -> [20]
                        .clamp(1e-10, 1.0)? // Ensure no zeros
                        .div(&probs_sample.sum_keepdim(1)?.squeeze(0)?)? // Normalize to sum to 1
                        .contiguous()?;

                    println!("probs_sample_1d: {:?}", probs_sample_1d.to_vec1::<f32>()?);
                    println!("probs_sample_1d sum: {:?}", probs_sample_1d.sum(0)?);

                    let s_t = multinomial_sample(&probs_sample_1d, temperature, seed)?;

                    // Gather true sequence values if needed
                    println!("Test 16");
                    let s_true_t = s_true.gather(&t_gather, 1)?.squeeze(1)?;
                    let s_t = s_t
                        .mul(&chain_mask_t)?
                        .add(&s_true_t.mul(&(&chain_mask_t.neg()? + 1.0)?)?)?;

                    // Update h_s
                    let h_s_update = self.w_s.forward(&s_t)?.unsqueeze(1)?;
                    let zero_mask = t_gather
                        .unsqueeze(2)?
                        .expand((b, 1, h_s.dim(2)?))?
                        .contiguous()?
                        .zeros_like()?;
                    let h_s = h_s.scatter_add(&t_gather, &zero_mask, 1)?; // Zero out
                    let h_s = h_s.scatter_add(&t_gather, &h_s_update, 1)?;

                    // Update s
                    let zero_mask = t_gather.zeros_like()?;
                    let s = s.scatter_add(&t_gather, &zero_mask, 1)?; // Zero out
                    let s = s.scatter_add(&t_gather, &s_t.unsqueeze(1)?, 1)?;

                    // Update all_probs
                    let probs_update = chain_mask_t
                        .unsqueeze(1)?
                        .unsqueeze(2)?
                        .expand((b, 1, 20))?
                        .mul(&probs_sample.unsqueeze(1)?)?;
                    let zero_mask = t_gather
                        .unsqueeze(2)?
                        .expand((b, 1, 20))?
                        .contiguous()?
                        .zeros_like()?;
                    all_probs = all_probs.scatter_add(&t_gather, &zero_mask, 1)?; // Zero out
                    all_probs = all_probs.scatter_add(&t_gather, &probs_update, 1)?; // Add new values

                    // Update all_log_probs
                    let log_probs_update = chain_mask_t
                        .unsqueeze(1)?
                        .unsqueeze(2)?
                        .expand((b, 1, 21))?
                        .mul(&log_probs.unsqueeze(1)?)?;
                    let zero_mask = t_gather
                        .unsqueeze(2)?
                        .expand((b, 1, 21))?
                        .contiguous()?
                        .zeros_like()?;
                    all_log_probs = all_log_probs.scatter_add(&t_gather, &zero_mask, 1)?; // Zero out
                    all_log_probs = all_log_probs.scatter_add(&t_gather, &log_probs_update, 1)?;
                    // Add new values

                    // // Get final h_v_t and generate logits
                    // let h_v_t = h_v_stack
                    //     .last()
                    //     .unwrap()
                    //     .gather(
                    //         &t_gather
                    //             .unsqueeze(2)?
                    //             .expand((b, 1, h_v_stack.last().unwrap().dim(2)?))?
                    //             .contiguous()?,
                    //         1,
                    //     )?
                    //     .squeeze(1)?;

                    // let logits = self.w_out.forward(&h_v_t)?;
                    // let log_probs = log_softmax(&logits, D::Minus1)?;

                    // // Generate probabilities and sample
                    // let probs =
                    //     softmax(&(logits.add(&bias_t)?.div_scalar(temperature)?)?, D::Minus1)?;
                    // let probs_sample = probs
                    //     .narrow(1, 0, 20)?
                    //     .div(&probs.narrow(1, 0, 20)?.sum_keepdim(1)?.expand((b, 20))?)?;

                    // let s_t = multinomial_sample(&probs_sample, temperature, seed)?;

                    // // Gather true sequence values if needed
                    // let s_true_t = s_true.gather(&t_gather, 1)?.squeeze(1)?;
                    // let s_t = s_t
                    //     .mul(&chain_mask_t)?
                    //     .add(&s_true_t.mul(&chain_mask_t.neg()?.add_scalar(1.0)?)?)?;

                    // // Update running tensors
                    // h_s = h_s.scatter(
                    //     1,
                    //     &t_gather
                    //         .unsqueeze(2)?
                    //         .expand((b, 1, h_s.dim(2)?))?
                    //         .contiguous()?,
                    //     &self.w_s.forward(&s_t)?.unsqueeze(1)?,
                    // )?;

                    // s = s.scatter(1, &t_gather, &s_t.unsqueeze(1)?)?;

                    // all_probs = all_probs.scatter(
                    //     1,
                    //     &t_gather.unsqueeze(2)?.expand((b, 1, 20))?.contiguous()?,
                    //     &chain_mask_t
                    //         .unsqueeze(1)?
                    //         .unsqueeze(2)?
                    //         .expand((b, 1, 20))?
                    //         .mul(&probs_sample.unsqueeze(1)?)?,
                    // )?;

                    // all_log_probs = all_log_probs.scatter(
                    //     1,
                    //     &t_gather.unsqueeze(2)?.expand((b, 1, 21))?.contiguous()?,
                    //     &chain_mask_t
                    //         .unsqueeze(1)?
                    //         .unsqueeze(2)?
                    //         .expand((b, 1, 21))?
                    //         .mul(&log_probs.unsqueeze(1)?)?,
                    // )?;
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
        let device = s_true.device();
        let (b, l) = s_true.dims2()?;

        let mask = &x_mask.as_ref().clone();
        let b_decoder: usize = b;

        // Todo: This is a hack. we shouldbe passing in encoded chains.
        // Update chain_mask to include missing regions
        let chain_mask = Tensor::zeros_like(mask.unwrap())?.to_dtype(DType::F32)?;
        let chain_mask = mask.unwrap().mul(&chain_mask)?;

        // encode ...
        let (h_v, h_e, e_idx) = self.encode(features)?;

        let rand_tensor = Tensor::randn(0., 1., (b, l), device)?.to_dtype(DType::F32)?;

        // Compute decoding order
        let decoding_order = (chain_mask + 0.001)?
            .mul(&rand_tensor.abs()?)?
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

                // let permutation_matrix_reverse = one_hot(decoding_order.clone(), l, 1., 0.)?; // need to double-check here
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
