use super::featurizer::ProteinFeatures;
use super::utilities::{compute_nearest_neighbors, cross_product, linspace};
use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::encoding::one_hot;
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder};

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
    pub fn new(
        edge_features: usize,
        node_features: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let augment_eps = 0.0; // hardcoding: Todo: refactor
        let top_k = 48; // hardcoding
        let num_rbf = 16; // hardcoding
        let num_positional_embeddings = 16; // hardcoding
        let edge_in = num_positional_embeddings + num_rbf * 25;
        let embeddings = PositionalEncodings::new(
            num_positional_embeddings, // num embeddings.
            32 as usize,               // max_relative_feature
            device,                    // device this should be passed in as param,
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

    /// This function calculates the nearest Ca coordinates and retunrs the ditances and indices.
    fn _dist(&self, x: &Tensor, mask: &Tensor, eps: f64) -> Result<(Tensor, Tensor)> {
        compute_nearest_neighbors(x, mask, self.top_k, self.augment_eps)
    }
    fn _rbf(&self, d: &Tensor, device: &Device) -> Result<Tensor> {
        // 1. It takes a tensor `d` as input and creates a set of RBF features
        // 2. Sets up parameters:
        //    - `d_min` = 2.0 (minimum distance)
        //    - `d_max` = 22.0 (maximum distance)
        //    - `d_count` = number of RBF centers
        // 3. Creates evenly spaced centers (μ) between d_min and d_max
        // 4. Calculates the width (σ) of the Gaussian functions
        // 5. Applies the RBF formula: exp(-(x-μ)²/σ²)
        const D_MIN: f64 = 2.0;
        const D_MAX: f64 = 22.0;

        // Create centers (μ)
        let d_mu =
            linspace(D_MIN, D_MAX, self.num_rbf, device)?.reshape((1, 1, 1, self.num_rbf))?;

        // Calculate width (σ)
        let d_sigma = (D_MAX - D_MIN) / self.num_rbf as f64;

        // Expand input tensor
        let d_expanded = d.unsqueeze(D::Minus1)?;

        // Calculate RBF values
        let diff = ((d_expanded - &d_mu)? / d_sigma)?;
        let rbf = diff.powf(2.0)?.neg()?.exp()?;

        Ok(rbf)
    }

    /// Computes RBF features for pairs of points specified by edge indices
    ///
    /// # Arguments
    /// * `a` - First set of points (N × D tensor)
    /// * `b` - Second set of points (M × D tensor)
    /// * `edge_indices` - Indices specifying which pairs to consider
    ///
    /// # Returns
    /// * RBF features for the specified pairs
    fn _get_rbf(&self, a: &Tensor, b: &Tensor, e_idx: &Tensor, device: &Device) -> Result<Tensor> {
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
        let rbf_a_b = self._rbf(&d_a_b_neighbors, device)?;

        Ok(rbf_a_b)
    }

    fn compute_pairwise_distances(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        const EPSILON: f64 = 1e-6; // Numerical stability constant

        let a_expanded = a.unsqueeze(2)?;
        let b_expanded = b.unsqueeze(1)?;

        // Euclidean distance calculation
        let diff = (a_expanded - b_expanded)?;
        let squared_distances = diff.powf(2.0)?.sum(3)?;
        let distances = (squared_distances + EPSILON)?.sqrt()?;

        Ok(distances)
    }

    fn gather_edge_distances(&self, distances: &Tensor, edge_indices: &Tensor) -> Result<Tensor> {
        distances
            .unsqueeze(D::Minus1)?
            .gather(edge_indices, 2)?
            .squeeze(D::Minus1)
    }

    pub fn forward(
        &self,
        input_features: &ProteinFeatures,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let x = input_features.get_coords();

        // let mask = input_features.output_dict.mask.as_ref();
        let mask = Tensor::zeros_like(x)?; // todo: fix mask

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
        rbf_all.push(self._rbf(&d_neighbors, device)?);
        rbf_all.push(self._get_rbf(&n, &n, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&c, &c, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&o, &o, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&cb, &cb, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&ca, &n, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&ca, &c, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&ca, &o, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&ca, &cb, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&n, &c, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&n, &o, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&n, &cb, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&cb, &c, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&cb, &o, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&o, &c, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&n, &ca, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&c, &ca, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&o, &ca, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&cb, &ca, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&c, &n, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&o, &n, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&cb, &n, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&c, &cb, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&o, &cb, &e_idx, device)?);
        rbf_all.push(self._get_rbf(&c, &o, &e_idx, device)?);

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
