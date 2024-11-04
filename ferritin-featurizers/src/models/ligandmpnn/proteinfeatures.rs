use super::featurizer::ProteinFeatures;
use super::utilities::{compute_nearest_neighbors, cross_product};
use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::encoding::one_hot;
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use std::cmp::min;

#[derive(Clone, Debug)]
/// https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L669
pub struct ProteinFeaturesModel {
    edge_features: usize,
    node_features: usize,
    num_positional_embeddings: usize,
    num_rbf: usize,
    top_k: usize,
    augment_eps: f32,
    embeddings: PositionalEncodings,
    edge_in: i64,
    edge_embedding: Linear,
    norm_edges: LayerNorm,
}

impl ProteinFeaturesModel {
    pub fn new(edge_features: usize, node_features: usize, vb: VarBuilder) -> Result<Self> {
        let augment_eps = 0.0; // hardcoding: Todo: refactor
        let top_k = 48; // hardcoding
        let num_rbf = 16; // hardcoding
        let num_positional_embeddings = 16; // hardcoding
        let edge_in = num_positional_embeddings + num_rbf * 25;
        let embeddings = PositionalEncodings::new(
            num_positional_embeddings, // num embeddings.
            32 as usize,               // max_relative_feature
            &Device::Cpu,              // device this should be passed in as para,
            vb.clone(),                // VarBuilder,
        )?;
        let edge_embedding = linear::linear(edge_in, edge_features, vb.pp("w_out")).unwrap();
        let norm_edges = layer_norm(
            edge_features,
            LayerNormConfig::default(),
            vb.pp("norm_edges"),
        )?;

        Ok(Self {
            edge_in: edge_in as i64,
            edge_features,
            node_features,
            num_positional_embeddings,
            num_rbf,
            top_k,
            augment_eps,
            embeddings,
            edge_embedding,
            norm_edges,
        })
    }

    /// This funciton calculates the nearest Ca coordinates.
    fn _dist(&self, x: &Tensor, mask: &Tensor, eps: f64) -> Result<(Tensor, Tensor)> {
        compute_nearest_neighbors(x, mask, self.top_k, self.augment_eps)

        // let mask_2d = mask.unsqueeze(1)?.mul(&mask.unsqueeze(2)?)?;
        // let dx = x.unsqueeze(1)?.sub(&x.unsqueeze(2)?)?;
        // let dx = dx.powf(2.)?.sum_keepdim(3)?;
        // let dx = (dx + eps)?.sqrt()?;
        // let d = mask_2d.mul(&dx)?;
        // let d_max = d.max_keepdim(D::Minus1)?;
        // let mask_tmp = (&mask_2d - 1.0)?.mul(&d_max)?;
        // let d_adjust = d.add(&mask_tmp)?;
        // let top_k = min(self.top_k, x.dim(1)?);

        // let mut tokenizer = Tokenizer::new(WordPiece::default());
        // let mut logits_processor = LogitsProcessor::new(
        //     1,    // seed
        //     None, // temperature
        //     Sampling::TopK{top_k, 1.0},
        //     top_k as usize, // top_n_logprobs
        //     tokenizer,
        //     None, // repeat_penalty
        //     None, // presence_penalty
        //     None, // logits_bias
        // );

        // top k sampling.
        // this section needs some eyeballs.
        // https://github.com/EricLBuehler/candle-sampling/issues/4
        // LogProbs: https://github.com/EricLBuehler/candle-sampling/blob/master/src/logits_processor.rs#L46
        // This will be in Candle after this PR:
        // https://github.com/huggingface/candle/pull/2375
        // Todo: update
        // let logprobs = logits_processor.sample(&d_adjust, None)?;
        // let d_neighbors_vec: Vec<f32> = logprobs.top_logprobs.iter().map(|x| x.logprob).collect();
        // let d_neighbors = Tensor::from_iter(d_neighbors_vec.into_iter(), x.device())?;
        // let e_idx_vec: Vec<u32> = logprobs.top_logprobs.iter().map(|x| x.token).collect();
        // let e_idx = Tensor::from_iter(e_idx_vec.into_iter(), x.device())?;

        // Ok((d_neighbors, e_idx))
    }
    fn _rbf(&self, d: &Tensor) -> Result<Tensor> {
        let device = d.device();
        let d_min = 2.0;
        let d_max = 22.0;
        let d_count = self.num_rbf;
        // Create linspace manually
        let step = (d_max - d_min) / (d_count - 1) as f64;
        let d_mu =
            Tensor::arange(0.0, d_count as f64, device)?.to_dtype(candle_core::DType::F64)?;
        let d_mu = ((&d_mu * step)? + d_min)?;
        let d_mu = d_mu.reshape((1, 1, 1, d_count))?;
        let d_sigma = (d_max - d_min) / d_count as f64;
        let d_expand = d.unsqueeze(D::Minus1)?;
        let diff = ((d_expand - &d_mu)? / d_sigma)?;
        let squared_diff = diff.powf(2.0)?;
        let rbf = squared_diff.neg()?.exp()?;
        Ok(rbf)
    }

    fn _get_rbf(&self, a: &Tensor, b: &Tensor, e_idx: &Tensor) -> Result<Tensor> {
        // Expand dimensions for broadcasting
        let a_expanded = a.unsqueeze(2)?;
        let b_expanded = b.unsqueeze(1)?;

        // Calculate pairwise distances
        let diff = (a_expanded - b_expanded)?;
        let squared_diff = diff.powf(2.0)?;
        let sum_squared_diff = squared_diff.sum(3)?;
        let d_a_b = (sum_squared_diff + 1e-6)?.sqrt()?;

        // Gather edges
        let d_a_b_expanded = d_a_b.unsqueeze(D::Minus1)?;
        let d_a_b_neighbors = d_a_b_expanded.gather(e_idx, 2)?;
        let d_a_b_neighbors = d_a_b_neighbors.squeeze(D::Minus1)?;

        // Apply RBF
        let rbf_a_b = self._rbf(&d_a_b_neighbors)?;

        Ok(rbf_a_b)
    }
    pub fn forward(&self, input_features: &ProteinFeatures) -> Result<(Tensor, Tensor)> {
        let x = input_features.get_coords();
        // let mask = input_features.output_dict.mask.as_ref();
        // let r_idx = input_features.output_dict.r_idx.as_ref();
        // let chain_labels = input_features.output_dict.chain_labels.as_ref();
        // let x = if self.augment_eps > 0.0 {
        //     let noise = x.randn_like(0.0, self.augment_eps as f64)?;
        //     (x + noise)?
        // } else {
        //     x.clone()
        // };
        let b = (&x.narrow(2, 1, 1)? - &x.narrow(2, 0, 1)?)?
            .squeeze(2)?
            .contiguous()?;
        let c = (&x.narrow(2, 2, 1)? - &x.narrow(2, 1, 1)?)?
            .squeeze(2)?
            .contiguous()?;
        let a = cross_product(&b, &c)?;
        let cb = {
            let a_term = &a * -0.58273431;
            let b_term = &b * 0.56802827;
            let c_term = &c * -0.54067466;
            let x_term = x.narrow(2, 1, 1)?.squeeze(2)?;
            (&a_term? + &b_term? - &c_term? + &x_term)?
        }
        .contiguous()?;

        // N/CA/C/O
        let n = x.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
        let ca = x.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
        let c = x.narrow(2, 2, 1)?.squeeze(2)?.contiguous()?;
        let o = x.narrow(2, 3, 1)?.squeeze(2)?.contiguous()?;

        let (d_neighbors, e_idx) = self._dist(&ca, &mask, self.augment_eps as f64)?;

        let mut rbf_all = Vec::new();
        rbf_all.push(self._rbf(&d_neighbors)?);
        rbf_all.push(self._get_rbf(&n, &n, &e_idx)?);
        rbf_all.push(self._get_rbf(&c, &c, &e_idx)?);
        rbf_all.push(self._get_rbf(&o, &o, &e_idx)?);
        rbf_all.push(self._get_rbf(&cb, &cb, &e_idx)?);
        rbf_all.push(self._get_rbf(&ca, &n, &e_idx)?);
        rbf_all.push(self._get_rbf(&ca, &c, &e_idx)?);
        rbf_all.push(self._get_rbf(&ca, &o, &e_idx)?);
        rbf_all.push(self._get_rbf(&ca, &cb, &e_idx)?);
        rbf_all.push(self._get_rbf(&n, &c, &e_idx)?);
        rbf_all.push(self._get_rbf(&n, &o, &e_idx)?);
        rbf_all.push(self._get_rbf(&n, &cb, &e_idx)?);
        rbf_all.push(self._get_rbf(&cb, &c, &e_idx)?);
        rbf_all.push(self._get_rbf(&cb, &o, &e_idx)?);
        rbf_all.push(self._get_rbf(&o, &c, &e_idx)?);
        rbf_all.push(self._get_rbf(&n, &ca, &e_idx)?);
        rbf_all.push(self._get_rbf(&c, &ca, &e_idx)?);
        rbf_all.push(self._get_rbf(&o, &ca, &e_idx)?);
        rbf_all.push(self._get_rbf(&cb, &ca, &e_idx)?);
        rbf_all.push(self._get_rbf(&c, &n, &e_idx)?);
        rbf_all.push(self._get_rbf(&o, &n, &e_idx)?);
        rbf_all.push(self._get_rbf(&cb, &n, &e_idx)?);
        rbf_all.push(self._get_rbf(&c, &cb, &e_idx)?);
        rbf_all.push(self._get_rbf(&o, &cb, &e_idx)?);
        rbf_all.push(self._get_rbf(&c, &o, &e_idx)?);

        let rbf_all = Tensor::cat(&rbf_all, D::Minus1)?;
        let offset = (&r_idx.unsqueeze(2)? - &r_idx.unsqueeze(1)?)?;
        let offset = offset
            .unsqueeze(D::Minus1)?
            .gather(&e_idx, 2)?
            .squeeze(D::Minus1)?;

        let d_chains = (&chain_labels.unsqueeze(2)? - &chain_labels.unsqueeze(1)?)?
            .eq(0.0)?
            .to_dtype(candle_core::DType::I64)?;
        let e_chains = d_chains
            .unsqueeze(D::Minus1)?
            .gather(&e_idx, 2)?
            .squeeze(D::Minus1)?;

        let e_positional = self
            .embeddings
            .forward(&offset.to_dtype(candle_core::DType::I64)?, &e_chains)?;

        let e = Tensor::cat(&[e_positional, rbf_all], D::Minus1)?;
        let e = self.edge_embedding.forward(&e)?;
        let e = self.norm_edges.forward(&e)?;

        Ok((e, e_idx))
    }
}

#[derive(Clone, Debug)]
pub struct PositionalEncodings {
    num_embeddings: usize,
    max_relative_feature: usize,
    linear: Linear,
}

impl PositionalEncodings {
    pub fn new(
        num_embeddings: usize,
        max_relative_feature: usize,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear = linear(
            2 * max_relative_feature + 2,
            num_embeddings,
            vb.pp("positional"),
        )?;

        Ok(Self {
            num_embeddings,
            max_relative_feature,
            linear,
        })
    }
    fn forward(&self, offset: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let ones = Tensor::ones_like(&mask)?;
        let mask_minus_one = mask.sub(&ones)?;
        let max_rel = self.max_relative_feature as i64;
        let d = (offset + max_rel as f64)?;
        let d = Tensor::clamp(&d, 0.0, 2 * max_rel)?;
        let d = d.mul(mask)?;
        let d = (d + mask_minus_one)?;
        let d = (d * (2 * max_rel + 1) as f64)?;
        let d_onehot = one_hot(d.clone(), 2 * self.max_relative_feature + 2, 1f32, 0f32)?;
        let d_onehot_float = d_onehot.to_dtype(candle_core::DType::F32)?;
        self.linear.forward(&d_onehot_float)
    }
}
