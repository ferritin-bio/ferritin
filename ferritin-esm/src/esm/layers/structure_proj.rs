use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, GELU};

use crate::esm::utils::constants::physics::BB_COORDINATES;
use crate::esm::utils::structure::affine3d::{Affine3D, RotationMatrix};

pub struct Dim6RotStructureHead {
    ffn1: linear::Linear,
    activation_fn: GELU,
    norm: layer_norm::LayerNorm,
    proj: linear::Linear,
    trans_scale_factor: f64,
    predict_torsion_angles: bool,
    bb_local_coords: Tensor,
}

impl Dim6RotStructureHead {
    pub fn new(
        input_dim: usize,
        trans_scale_factor: f64,
        norm_type: &str,
        activation_fn: &str,
        predict_torsion_angles: bool,
    ) -> Result<Self> {
        let ffn1 = linear::Config::new(input_dim, input_dim).build()?;
        let activation_fn = GELU::new();
        let norm = layer_norm::Config::new(input_dim).build()?;
        let proj = linear::Config::new(input_dim, 9 + 7 * 2).build()?;
        let bb_local_coords = Tensor::new(BB_COORDINATES)?;

        Ok(Self {
            ffn1,
            activation_fn,
            norm,
            proj,
            trans_scale_factor,
            predict_torsion_angles,
            bb_local_coords,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        affine: Option<Affine3D>,
        affine_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let rigids = match affine {
            None => Affine3D::identity(
                x.shape()?.split_last()?.1,
                x.dtype(),
                x.device(),
                self.training,
                RotationMatrix,
            )?,
            Some(a) => a,
        };

        // [*, N]
        let x = self.ffn1.forward(x)?;
        let x = self.activation_fn.forward(&x)?;
        let x = self.norm.forward(&x)?;

        let proj_out = self.proj.forward(&x)?;
        let (trans, x, y, angles) = proj_out.split(&[3, 3, 3, 7 * 2], -1)?;

        let trans = trans.mul_scalar(self.trans_scale_factor)?;
        let x = x.div(&(x.norm_scaler(2)? + 1e-5)?)?;
        let y = y.div(&(y.norm_scaler(2)? + 1e-5)?)?;

        let update = Affine3D::from_graham_schmidt(&(x.add(&trans)?), &trans, &(y.add(&trans)?))?;
        let rigids = rigids.compose(&update.mask(affine_mask)?)?;
        let affine = rigids.tensor()?;

        // Approximate backbone atom positions
        let mut shape = x.shape()?.to_vec();
        shape.extend_from_slice(&[3, 3]);
        let all_bb_coords_local = self.bb_local_coords.broadcast_as(shape.as_slice())?;
        let pred_xyz = rigids.apply_expanded(&all_bb_coords_local)?;

        Ok((affine, pred_xyz))
    }
}
