//! LigandMPNN's featurizer: protein edges plus a per-residue ligand context.
//!
//! Port of `ProteinFeaturesLigand`
//! ([model_utils.py](https://github.com/dauparas/LigandMPNN/blob/main/model_utils.py#L669)).
//!
//! Where [`ProteinFeaturesModel`][super::proteinfeaturesmodel::ProteinFeaturesModel]
//! produces only `(E, E_idx)` — the protein's own k-nearest-neighbour edge
//! graph — this one additionally produces, for each residue, a window onto the
//! `atom_context_num` ligand atoms nearest its virtual Cβ:
//!
//! * `V` `(B, L, M, 128)` — one node embedding per (residue, ligand atom)
//!   pair, built from five backbone-to-ligand distances, the atom's element
//!   identity, and four local-frame angle features.
//! * `Y_nodes` `(B, L, M, 128)` — the same atoms as nodes of a ligand-only
//!   graph.
//! * `Y_edges` `(B, L, M, M, 128)` — the ligand graph's own edges.
//!
//! The protein edge path (`E`, `E_idx`) is identical to ProteinMPNN's, down to
//! the same 25 RBF blocks and the same positional encoding, so the two share
//! [`PositionalEncodings`][super::proteinfeaturesmodel::PositionalEncodings]
//! and the `edge_embedding`/`norm_edges` weight names.

use super::configs::ProteinMPNNConfig;
use super::proteinfeatures::ProteinFeatures;
use super::proteinfeaturesmodel::{
    ATOM37_C, ATOM37_CA, ATOM37_N, ATOM37_O, DIST_EPSILON, PositionalEncodings, RBF_MAX_DISTANCE,
    RBF_MIN_DISTANCE, virtual_cb,
};
use crate::featurize::utilities::{
    compute_nearest_neighbors, cross_product, gather_edges, get_nearest_neighbours,
    linear_last_dim, linspace_f32,
};
use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::encoding::one_hot;
use candle_nn::{LayerNorm, LayerNormConfig, Linear, VarBuilder, layer_norm, linear};

/// Widest element the one-hot covers; `periodic_table_features` has 119
/// columns (Z = 0..=118) and the reference one-hots `Y_t` to 120.
const ELEMENT_ONE_HOT: usize = 120;
/// Periodic-table group, 0..=18 — 19 categories counting the 0 used for
/// padding slots and for hydrogen's absent group.
const GROUP_ONE_HOT: usize = 19;
/// Periodic-table period, 0..=7 — 8 categories counting 0.
const PERIOD_ONE_HOT: usize = 8;
/// `ELEMENT_ONE_HOT + GROUP_ONE_HOT + PERIOD_ONE_HOT`, the width both
/// `type_linear` and `y_nodes` consume.
const ELEMENT_FEATURES: usize = ELEMENT_ONE_HOT + GROUP_ONE_HOT + PERIOD_ONE_HOT;
/// Width of `type_linear`'s output, concatenated into the node features.
const ELEMENT_EMBED: usize = 64;
/// The four local-frame angle features per (residue, ligand atom) pair.
const ANGLE_FEATURES: usize = 4;
/// Backbone atoms measured against each ligand atom: N, CA, C, O, Cβ.
const BACKBONE_ATOMS_FOR_LIGAND: usize = 5;

/// Periodic-table group for each atomic number 0..=118.
///
/// `periodic_table_features[1]` in the reference. It is a plain tensor
/// attribute there rather than a registered buffer, so it is **not** in the
/// checkpoint and has to be transcribed. Index 0 is the padding slot.
#[rustfmt::skip]
const PERIODIC_GROUP: [u32; 119] = [
    0,
    1, 18,
    1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
];

/// Periodic-table period for each atomic number 0..=118.
///
/// `periodic_table_features[2]` in the reference; see [`PERIODIC_GROUP`].
#[rustfmt::skip]
const PERIODIC_PERIOD: [u32; 119] = [
    0,
    1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7,
];

/// LigandMPNN's featurizer.
#[derive(Clone, Debug)]
pub struct ProteinFeaturesLigand {
    num_rbf: usize,
    top_k: usize,
    atom_context_num: usize,
    embeddings: PositionalEncodings,
    edge_embedding: Linear,
    norm_edges: LayerNorm,
    node_project_down: Linear,
    norm_nodes: LayerNorm,
    type_linear: Linear,
    y_nodes: Linear,
    y_edges: Linear,
    norm_y_nodes: LayerNorm,
    norm_y_edges: LayerNorm,
}

/// What the featurizer hands the encoder.
pub struct LigandFeatures {
    /// `(B, L, M, 128)` per-(residue, ligand atom) node embeddings.
    pub v: Tensor,
    /// `(B, L, K, 128)` protein edge embeddings.
    pub e: Tensor,
    /// `(B, L, K)` protein neighbour indices.
    pub e_idx: Tensor,
    /// `(B, L, M, 128)` ligand graph nodes.
    pub y_nodes: Tensor,
    /// `(B, L, M, M, 128)` ligand graph edges.
    pub y_edges: Tensor,
    /// `(B, L, M)` ligand atom validity, after per-residue selection.
    pub y_m: Tensor,
}

impl ProteinFeaturesLigand {
    pub fn load(vb: VarBuilder, config: &ProteinMPNNConfig) -> Result<Self> {
        let num_rbf = config.num_rbf as usize;
        let node_features = config.node_features as usize;
        let edge_features = config.edge_features as usize;
        let num_positional_embeddings = 16usize;
        let edge_in = num_positional_embeddings + num_rbf * 25;
        // 5 backbone-to-ligand RBF blocks + the element embedding + 4 angles.
        let node_in = BACKBONE_ATOMS_FOR_LIGAND * num_rbf + ELEMENT_EMBED + ANGLE_FEATURES;

        Ok(Self {
            num_rbf,
            top_k: config.k_neighbors as usize,
            atom_context_num: config.atom_context_num,
            embeddings: PositionalEncodings::new(
                num_positional_embeddings,
                32,
                vb.pp("embeddings"),
            )?,
            edge_embedding: linear::linear_no_bias(
                edge_in,
                edge_features,
                vb.pp("edge_embedding"),
            )?,
            norm_edges: layer_norm(
                edge_features,
                LayerNormConfig::default(),
                vb.pp("norm_edges"),
            )?,
            node_project_down: linear(node_in, node_features, vb.pp("node_project_down"))?,
            norm_nodes: layer_norm(
                node_features,
                LayerNormConfig::default(),
                vb.pp("norm_nodes"),
            )?,
            type_linear: linear(ELEMENT_FEATURES, ELEMENT_EMBED, vb.pp("type_linear"))?,
            y_nodes: linear::linear_no_bias(ELEMENT_FEATURES, node_features, vb.pp("y_nodes"))?,
            y_edges: linear::linear_no_bias(num_rbf, node_features, vb.pp("y_edges"))?,
            norm_y_nodes: layer_norm(
                node_features,
                LayerNormConfig::default(),
                vb.pp("norm_y_nodes"),
            )?,
            norm_y_edges: layer_norm(
                node_features,
                LayerNormConfig::default(),
                vb.pp("norm_y_edges"),
            )?,
        })
    }

    /// Gaussian RBF expansion over `[2, 22]` Å, for a tensor of any rank.
    ///
    /// The ProteinMPNN copy of this assumes a rank-3 input; here distances
    /// arrive as `(B, L, K)`, `(B, L, M)` and `(B, L, M, M)`, so it appends the
    /// bin axis to whatever shape it is given.
    fn rbf(&self, d: &Tensor) -> Result<Tensor> {
        let device = d.device();
        let mu = linspace_f32(RBF_MIN_DISTANCE, RBF_MAX_DISTANCE, self.num_rbf, device)?;
        let sigma = (RBF_MAX_DISTANCE - RBF_MIN_DISTANCE) / self.num_rbf as f32;

        let mut shape = d.dims().to_vec();
        shape.push(self.num_rbf);
        let mut mu_shape = vec![1usize; d.dims().len()];
        mu_shape.push(self.num_rbf);

        let expanded = d.unsqueeze(D::Minus1)?.broadcast_as(shape.as_slice())?;
        let mu = mu
            .reshape(mu_shape.as_slice())?
            .broadcast_as(shape.as_slice())?;
        ((expanded - mu)? / sigma as f64)?.powf(2.0)?.neg()?.exp()
    }

    /// `sqrt(sum((a - b)^2) + 1e-6)` between every residue's `a` and its
    /// `E_idx` neighbours' `b`.
    fn edge_rbf(&self, a: &Tensor, b: &Tensor, e_idx: &Tensor) -> Result<Tensor> {
        let (batch, len, dims) = a.dims3()?;
        let shape = (batch, len, len, dims);
        let d = (a.unsqueeze(2)?.broadcast_as(shape)? - b.unsqueeze(1)?.broadcast_as(shape)?)?
            .powf(2.0)?
            .sum(3)?;
        let d = (d + DIST_EPSILON as f64)?.sqrt()?;
        let gathered = gather_edges(&d.unsqueeze(D::Minus1)?, e_idx)?.squeeze(D::Minus1)?;
        self.rbf(&gathered)
    }

    /// `sqrt(sum((a[:, :, None, :] - y)^2) + 1e-6)`, one distance per
    /// (residue, ligand atom) pair, expanded into RBF bins.
    fn ligand_rbf(&self, a: &Tensor, y: &Tensor) -> Result<Tensor> {
        let d = a
            .unsqueeze(2)?
            .broadcast_sub(y)?
            .powf(2.0)?
            .sum(D::Minus1)?;
        self.rbf(&(d + DIST_EPSILON as f64)?.sqrt()?)
    }

    /// Each ligand atom's position in its residue's local backbone frame,
    /// encoded as four trigonometric features.
    ///
    /// Port of `_make_angle_features`: build an orthonormal frame `(e1, e2,
    /// e3)` from the N→CA and C→CA vectors by Gram-Schmidt, project
    /// `Y - CA` into it, then take `(cos φ, sin φ, cos θ, sin θ)` in
    /// cylindrical-ish coordinates.
    fn angle_features(&self, n: &Tensor, ca: &Tensor, c: &Tensor, y: &Tensor) -> Result<Tensor> {
        let v1 = (n - ca)?;
        let v2 = (c - ca)?;
        let e1 = normalize_last_dim(&v1)?;
        // einsum("bli, bli -> bl", e1, v2), kept as a keepdim so it broadcasts.
        let e1_dot_v2 = (&e1 * &v2)?.sum_keepdim(D::Minus1)?;
        let u2 = (&v2 - e1.broadcast_mul(&e1_dot_v2)?)?;
        let e2 = normalize_last_dim(&u2)?;
        let e3 = cross_product(&e1, &e2)?;

        // R_residue is (B, L, 3, 3) with the basis vectors as COLUMNS, and
        // einsum("blqp, blyq -> blyp") contracts over q — the row index. So
        // `local = (Y - CA) @ R`, i.e. one dot product per basis vector.
        let rel = y.broadcast_sub(&ca.unsqueeze(2)?)?;
        let project =
            |e: &Tensor| -> Result<Tensor> { rel.broadcast_mul(&e.unsqueeze(2)?)?.sum(D::Minus1) };
        let lx = project(&e1)?;
        let ly = project(&e2)?;
        let lz = project(&e3)?;

        let rxy = ((lx.powf(2.0)? + ly.powf(2.0)?)? + 1e-8)?.sqrt()?;
        let rxyz = (((lx.powf(2.0)? + ly.powf(2.0)?)? + lz.powf(2.0)?)?.sqrt()? + 1e-8)?;
        let f1 = (&lx / &rxy)?;
        let f2 = (&ly / &rxy)?;
        let f3 = (&rxy / &rxyz)?;
        let f4 = (&lz / &rxyz)?;
        Tensor::cat(
            &[
                f1.unsqueeze(D::Minus1)?,
                f2.unsqueeze(D::Minus1)?,
                f3.unsqueeze(D::Minus1)?,
                f4.unsqueeze(D::Minus1)?,
            ],
            D::Minus1,
        )
    }

    /// One-hot an atomic number as (element, group, period), width 147.
    fn element_one_hot(&self, y_t: &Tensor) -> Result<Tensor> {
        let device = y_t.device();
        let dims = y_t.dims().to_vec();
        let flat = y_t.flatten_all()?.to_dtype(DType::U32)?;

        let table = |values: &[u32], width: usize| -> Result<Tensor> {
            let lookup = Tensor::from_slice(values, (values.len(),), device)?;
            let mapped = lookup.gather(&flat, 0)?;
            one_hot(mapped.reshape(dims.as_slice())?, width, 1f32, 0f32)
        };

        let element = one_hot(flat.reshape(dims.as_slice())?, ELEMENT_ONE_HOT, 1f32, 0f32)?;
        let group = table(&PERIODIC_GROUP, GROUP_ONE_HOT)?;
        let period = table(&PERIODIC_PERIOD, PERIOD_ONE_HOT)?;
        Tensor::cat(&[element, group, period], D::Minus1)
    }

    pub fn forward(&self, features: &ProteinFeatures) -> Result<LigandFeatures> {
        let x = features.get_coords();
        let mask = features.x_mask.as_ref().unwrap();
        let r_idx = features.get_residue_index();

        let n = x.narrow(2, ATOM37_N, 1)?.squeeze(2)?.contiguous()?;
        let ca = x.narrow(2, ATOM37_CA, 1)?.squeeze(2)?.contiguous()?;
        let c = x.narrow(2, ATOM37_C, 1)?.squeeze(2)?.contiguous()?;
        let o = x.narrow(2, ATOM37_O, 1)?.squeeze(2)?.contiguous()?;
        let cb = virtual_cb(&n, &ca, &c)?;

        // ── Per-residue ligand context ──────────────────────────────────────
        //
        // The reference does this selection in `data_utils.featurize`, before
        // the model. Here it lives with the model, because how many atoms a
        // residue sees is `atom_context_num` — a property of the checkpoint,
        // not of the structure.
        let (y, y_t, y_m) = {
            let y = features.y.squeeze(0)?;
            let y_t = features.y_t.squeeze(0)?;
            let y_m = features
                .y_m
                .as_ref()
                .expect("LigandMPNN requires a ligand mask")
                .squeeze(0)?;
            let (y, y_t, y_m, _d_closest) =
                get_nearest_neighbours(&cb, mask, &y, &y_t, &y_m, self.atom_context_num as i64)?;
            (y.unsqueeze(0)?, y_t.unsqueeze(0)?, y_m.unsqueeze(0)?)
        };

        // ── Protein edges: identical to ProteinMPNN's ───────────────────────
        let (d_neighbors, e_idx) = compute_nearest_neighbors(&ca, mask, self.top_k, DIST_EPSILON)?;
        let mut rbf_all = vec![self.rbf(&d_neighbors)?];
        for (a, b) in [
            (&n, &n),
            (&c, &c),
            (&o, &o),
            (&cb, &cb),
            (&ca, &n),
            (&ca, &c),
            (&ca, &o),
            (&ca, &cb),
            (&n, &c),
            (&n, &o),
            (&n, &cb),
            (&cb, &c),
            (&cb, &o),
            (&o, &c),
            (&n, &ca),
            (&c, &ca),
            (&o, &ca),
            (&cb, &ca),
            (&c, &n),
            (&o, &n),
            (&cb, &n),
            (&c, &cb),
            (&o, &cb),
            (&c, &o),
        ] {
            rbf_all.push(self.edge_rbf(a, b, &e_idx)?);
        }
        let rbf_all = Tensor::cat(&rbf_all, D::Minus1)?;

        let dims = r_idx.dims();
        let pair = (dims[0], dims[1], dims[1]);
        let offset = (r_idx
            .unsqueeze(2)?
            .broadcast_as(pair)?
            .to_dtype(DType::F32)?
            - r_idx
                .unsqueeze(1)?
                .broadcast_as(pair)?
                .to_dtype(DType::F32)?)?;
        let offset = gather_edges(&offset.unsqueeze(D::Minus1)?, &e_idx)?.squeeze(D::Minus1)?;

        // Single chain is the only case the featurizer builds today; chain
        // labels are not yet plumbed through `ProteinFeatures`, so every pair
        // reads as same-chain. Matches the ProteinMPNN featurizer.
        let e_chains = Tensor::ones(offset.dims(), DType::F32, offset.device())?;
        let e_positional = self.embeddings.forward(&offset, &e_chains)?;

        let e = Tensor::cat(&[e_positional, rbf_all], D::Minus1)?;
        let e = self.norm_edges.forward(&self.edge_embedding.forward(&e)?)?;

        // ── Ligand nodes ────────────────────────────────────────────────────
        let y_t_one_hot = self.element_one_hot(&y_t)?;
        let y_t_embed = self.type_linear.forward(&y_t_one_hot)?;

        let d_all = Tensor::cat(
            &[
                self.ligand_rbf(&n, &y)?,
                self.ligand_rbf(&ca, &y)?,
                self.ligand_rbf(&c, &y)?,
                self.ligand_rbf(&o, &y)?,
                self.ligand_rbf(&cb, &y)?,
                y_t_embed,
                self.angle_features(&n, &ca, &c, &y)?,
            ],
            D::Minus1,
        )?;
        let v = self
            .norm_nodes
            .forward(&self.node_project_down.forward(&d_all)?)?;

        // ── Ligand graph ────────────────────────────────────────────────────
        // Pairwise distances among the M context atoms of each residue.
        let y_pair = y
            .unsqueeze(3)?
            .broadcast_sub(&y.unsqueeze(2)?)?
            .powf(2.0)?
            .sum(D::Minus1)?;
        let y_edges = self.rbf(&(y_pair + DIST_EPSILON as f64)?.sqrt()?)?;
        // Rank 5 — one M x M edge matrix per residue — so it needs
        // `linear_last_dim` rather than `Linear::forward`.
        let y_edges = linear_last_dim(&self.y_edges, &y_edges)?;
        let y_edges = self.norm_y_edges.forward(&y_edges)?;
        let y_nodes = self
            .norm_y_nodes
            .forward(&self.y_nodes.forward(&y_t_one_hot)?)?;

        Ok(LigandFeatures {
            v,
            e,
            e_idx,
            y_nodes,
            y_edges,
            y_m,
        })
    }
}

/// `torch.nn.functional.normalize(v, dim=-1)`: unit length, with torch's
/// `1e-12` floor on the denominator.
fn normalize_last_dim(v: &Tensor) -> Result<Tensor> {
    let norm = v
        .powf(2.0)?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f64::INFINITY)?;
    v.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two periodic-table rows must line up with the reference table,
    /// which is a plain attribute rather than a buffer and so is absent from
    /// the checkpoint — nothing at load time would catch a transcription slip.
    #[test]
    fn test_periodic_table_rows_have_expected_shape_and_spot_values() {
        assert_eq!(PERIODIC_GROUP.len(), 119);
        assert_eq!(PERIODIC_PERIOD.len(), 119);
        assert_eq!(*PERIODIC_GROUP.iter().max().unwrap(), 18);
        assert_eq!(*PERIODIC_PERIOD.iter().max().unwrap(), 7);

        // The elements 1BC8's ligands actually contain: C, N, O, P, Zn.
        for (z, group, period) in [
            (6usize, 14u32, 2u32), // carbon
            (7, 15, 2),            // nitrogen
            (8, 16, 2),            // oxygen
            (15, 15, 3),           // phosphorus
            (30, 12, 4),           // zinc
            (0, 0, 0),             // the padding slot
        ] {
            assert_eq!(PERIODIC_GROUP[z], group, "group of Z={z}");
            assert_eq!(PERIODIC_PERIOD[z], period, "period of Z={z}");
        }
    }

    #[test]
    fn test_element_feature_width_matches_the_checkpoint() {
        // `type_linear` and `y_nodes` are both (_, 147) in every LigandMPNN
        // checkpoint; a wrong width here is a load-time shape error, but this
        // states the number so the constants can be read at a glance.
        assert_eq!(ELEMENT_FEATURES, 147);
    }
}
