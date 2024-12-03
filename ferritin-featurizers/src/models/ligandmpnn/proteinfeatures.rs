use super::configs::ProteinMPNNConfig;
use super::featurizer::ProteinFeatures;
use super::utilities::{compute_nearest_neighbors, cross_product, gather_edges, linspace};
use candle_core::{DType, Device, Module, Result, Tensor, D};
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
    pub fn load(vb: VarBuilder, config: ProteinMPNNConfig) -> Result<Self> {
        let augment_eps = config.augment_eps;
        let top_k = config.k_neighbors as usize; // Todo: check that this is 48
        let num_rbf = config.num_rbf as usize; // Todo: check that this is 16
        let num_positional_embeddings = 16usize; // todo: hardcoding : check where this number
        let edge_in = num_positional_embeddings + num_rbf * 25;
        let edge_features = config.edge_features as usize;
        let node_features = config.node_features as usize;
        let embeddings = PositionalEncodings::new(
            num_positional_embeddings, // num embeddings.
            32usize,                   // max_relative_feature
            vb.pp("embeddings"),       // VarBuilder,
        )?;
        let edge_embedding =
            linear::linear_no_bias(edge_in, edge_features, vb.pp("edge_embedding"))?;
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

    /// This function calculates the nearest Ca coordinates and returns the distances and indices.
    // Todo: potential refactor
    fn _dist(&self, x: &Tensor, mask: &Tensor, eps: f64) -> Result<(Tensor, Tensor)> {
        compute_nearest_neighbors(x, mask, self.top_k, eps as f32)
    }
    /// 1. It takes a tensor `d` as input and creates a set of RBF features
    /// 2. Sets up parameters:
    ///    - `d_min` = 2.0 (minimum distance)
    ///    - `d_max` = 22.0 (maximum distance)
    ///    - `d_count` = number of RBF centers
    /// 3. Creates evenly spaced centers (μ) between d_min and d_max
    /// 4. Calculates the width (σ) of the Gaussian functions
    /// 5. Applies the RBF formula: exp(-(x-μ)²/σ²)
    fn _rbf(&self, d: &Tensor, device: &Device) -> Result<Tensor> {
        const D_MIN: f64 = 2.0;
        const D_MAX: f64 = 22.0;
        // Create centers (μ)
        let d_mu = linspace(D_MIN, D_MAX, self.num_rbf, &Device::Cpu)? // Use CPU device
            .to_dtype(DType::F32)? // Convert to F32 on CPU
            .reshape((1, 1, 1, self.num_rbf))?
            .to_device(device)?; // Move to Metal device after conversion
                                 // Calculate width (σ)
        let d_sigma = (D_MAX - D_MIN) / self.num_rbf as f64;
        let dims = d.dims();
        let d_expanded = d.unsqueeze(D::Minus1)?; // [N, N, C, 1]
        let d_mu_broadcast = d_mu.broadcast_as((dims[0], dims[1], dims[2], self.num_rbf))?;
        let d_expanded_broadcast =
            d_expanded.broadcast_as((dims[0], dims[1], dims[2], self.num_rbf))?;
        let d_sigma_tensor =
            Tensor::new(&[d_sigma as f32], &device)?.broadcast_as(d_expanded_broadcast.shape())?;
        let diff = ((d_expanded_broadcast - d_mu_broadcast)? / d_sigma_tensor)?;
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
    //     def _get_rbf(self, A, B, E_idx):
    // D_A_B = torch.sqrt(
    //     torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
    // )  # [B, L, L]
    // D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
    //     :, :, :, 0
    // ]  # [B,L,K]
    // RBF_A_B = self._rbf(D_A_B_neighbors)
    // return RBF_A_B
    fn _get_rbf(&self, a: &Tensor, b: &Tensor, e_idx: &Tensor, device: &Device) -> Result<Tensor> {
        let (batch, seq_len, pos) = a.dims3()?;
        let target_shape = (batch, seq_len, seq_len, pos); // [1, 93, 93, 3]
        let a_expanded = a.unsqueeze(2)?.broadcast_as(target_shape)?;
        let b_expanded = b.unsqueeze(1)?.broadcast_as(target_shape)?;
        let diff = (a_expanded - b_expanded)?;
        let squared_diff = diff.powf(2.0)?;
        let sum_squared_diff = squared_diff.sum(3)?;
        let d_a_b = (sum_squared_diff + 1e-6)?.sqrt()?;
        let d_a_b_neighbors = gather_edges(&d_a_b.unsqueeze(D::Minus1)?, &e_idx)?;
        let d_a_b_neighbors = d_a_b_neighbors.squeeze(D::Minus1)?;
        let rbf_a_b = self._rbf(&d_a_b_neighbors, device)?;
        Ok(rbf_a_b)
    }
    /// Euclidean distance calculation
    fn compute_pairwise_distances(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        const EPSILON: f64 = 1e-6; // Numerical stability constant
        let a_expanded = a.unsqueeze(2)?;
        let b_expanded = b.unsqueeze(1)?;
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
        let mask = input_features.x_mask.as_ref().unwrap();
        let r_idx = input_features.get_residue_index();
        // let chain_labels = input_features.chain_labels.as_ref();
        //
        // todo: fix
        // let chain_labels = input_features.get_chain_labels();
        let chain_labels = Tensor::zeros_like(r_idx)?;
        let x = if self.augment_eps > 0.0 {
            let noise = x.randn_like(0.0, self.augment_eps as f64)?;
            (x + noise)?
        } else {
            x.clone()
        };
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
        let dims = r_idx.dims();
        let target_shape = (dims[0], dims[1], dims[1]);
        let r_idx_expanded1 = r_idx
            .unsqueeze(2)?
            .broadcast_as(target_shape)?
            .to_dtype(DType::F32)?; // [1, 93, 93]
        let r_idx_expanded2 = r_idx
            .unsqueeze(1)?
            .broadcast_as(target_shape)?
            .to_dtype(DType::F32)?; // [1, 93, 93]

        let offset = (r_idx_expanded1 - r_idx_expanded2)?;
        let offset = gather_edges(&offset.unsqueeze(D::Minus1)?, &e_idx)?;
        let offset = offset.squeeze(D::Minus1)?;
        let dims = chain_labels.dims();
        let target_shape = (dims[0], dims[1], dims[1]);
        let d_chains = (&chain_labels.unsqueeze(2)?.broadcast_as(target_shape)?
            - &chain_labels.unsqueeze(1)?.broadcast_as(target_shape)?)?
            .eq(0.0)?
            .to_dtype(DType::U32)?;
        // E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        let e_chains = gather_edges(&d_chains.unsqueeze(D::Minus1)?, &e_idx)?.squeeze(D::Minus1)?;
        let e_positional = self
            .embeddings
            .forward(&offset.to_dtype(DType::U32)?, &e_chains)?;
        let e = Tensor::cat(&[e_positional, rbf_all], D::Minus1)?;
        let e = self.edge_embedding.forward(&e)?;
        let e = self.norm_edges.forward(&e)?;
        Ok((e, e_idx))
    }
}

#[derive(Clone, Debug)]
pub struct PositionalEncodings {
    max_relative_feature: usize,
    linear: Linear,
}
impl PositionalEncodings {
    pub fn new(num_embeddings: usize, max_relative_feature: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(
            2 * max_relative_feature + 2,
            num_embeddings,
            vb.pp("linear"),
        )?;
        Ok(Self {
            max_relative_feature,
            linear,
        })
    }
    /// - [pytorch](https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L1645)
    fn forward(&self, offset: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // def forward(self, offset, mask):
        //     d = torch.clip(
        //         offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        //     ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        //     d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        //     E = self.linear(d_onehot.float())
        //     return E
        //
        // println!("In positional Embedding: forward");
        // Offset: Tensor[dims 1, 93, 24; u32, metal:4294969325]
        let max_rel = self.max_relative_feature as f64;
        // First part: clip(offset + max_rel, 0, 2*max_rel)
        let d = (offset + max_rel)?;
        let d = d.clamp(0f64, 2.0 * max_rel)?;
        // Second part: d * mask + (1-mask)*(2*max_rel + 1)
        let masked_d = d.mul(mask)?;
        let inverse_mask = (mask * -1.0)? + 1.0;
        let extra_term = inverse_mask? * ((2.0 * max_rel) + 1.0);
        let d = (masked_d + extra_term?)?;
        // Todo: confirms this is correct: we are converting the mask
        // Normalize the values by subtracting 97 (ASCII 'a') to make them 0-based
        // let d_normalized = (d - 97u32)?; // This will make 'a'=0, 'b'=1, etc.
        let offset_val = Tensor::full(97u32, d.dims(), d.device());
        let d_normalized = (d - offset_val)?;
        let depth = (2 * self.max_relative_feature + 2) as i64;
        let d_onehot = one_hot(d_normalized, depth as usize, 1f32, 0f32)?;
        let d_onehot_float = d_onehot.to_dtype(DType::F32)?;
        self.linear.forward(&d_onehot_float)
    }
}
