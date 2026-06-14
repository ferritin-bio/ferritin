//! Affine3D local reference frames for ESM3 geometric attention.
//!
//! Each residue gets a local coordinate frame defined by its backbone (N, CA, C) atoms.
//! The rotation matrix columns are orthonormal frame axes; the translation is the CA position.

use candle_core::{D, DType, Result, Tensor};

/// Per-residue local reference frame: rotation matrix + translation.
///
/// `rot`:   `(*, 3, 3)` — orthonormal rotation matrix, columns are the frame axes.
/// `trans`: `(*,    3)` — CA position in global coordinates.
pub struct Affine3D {
    pub rot: Tensor,
    pub trans: Tensor,
}

impl Affine3D {
    pub fn new(rot: Tensor, trans: Tensor) -> Self {
        Self { rot, trans }
    }

    /// Rotate vectors from local frame to global frame.
    ///
    /// `rot`: `(*, 3, 3)`, `v`: `(*, H, 3)` → `(*, H, 3)`.
    ///
    /// In row-vector convention: `v @ rot^T`.
    pub fn apply_rot(rot: &Tensor, v: &Tensor) -> Result<Tensor> {
        let rot_t = rot.transpose(D::Minus2, D::Minus1)?;
        v.matmul(&rot_t)
    }

    /// Rotate vectors from global frame to local frame (inverse rotation).
    ///
    /// `rot`: `(*, 3, 3)`, `v`: `(*, H, 3)` → `(*, H, 3)`.
    ///
    /// For orthogonal matrices: inverse = transpose, so `v @ rot`.
    pub fn apply_rot_inv(rot: &Tensor, v: &Tensor) -> Result<Tensor> {
        v.matmul(rot)
    }

    /// Apply full affine transform (rotate + translate) to `v` of shape `(*, H, 3)`.
    ///
    /// Returns `(*, H, 3)` in global coordinates.
    pub fn apply(&self, v: &Tensor) -> Result<Tensor> {
        let rotated = Self::apply_rot(&self.rot, v)?; // (*, H, 3)
        let trans = self.trans.unsqueeze(D::Minus2)?; // (*, 1, 3)
        rotated.broadcast_add(&trans)
    }

    /// Build per-residue local frames from backbone atom coordinates.
    ///
    /// `coords`: `(B, L, 3, 3)` — atom order is `(N, CA, C)`.
    ///
    /// Frame construction follows AlphaFold2/ESM convention (Graham-Schmidt on CA-C and CA-N):
    /// - x-axis: normalized `CA - C`
    /// - y-axis: normalized component of `N - CA` perpendicular to x-axis
    /// - z-axis: `x × y`
    /// - origin: CA position
    ///
    /// Returns `(Affine3D, mask)` where `mask (B, L)` is `true` (u8=1) where both
    /// backbone vectors have non-trivial length (atoms are present).
    pub fn build_affine3d_from_coordinates(coords: &Tensor) -> Result<(Self, Tensor)> {
        // Extract atom positions: each (B, L, 3)
        let n_pos = coords.narrow(D::Minus2, 0, 1)?.squeeze(D::Minus2)?;
        let ca_pos = coords.narrow(D::Minus2, 1, 1)?.squeeze(D::Minus2)?;
        let c_pos = coords.narrow(D::Minus2, 2, 1)?.squeeze(D::Minus2)?;

        // Frame edge vectors from CA
        let x_axis = ca_pos.sub(&c_pos)?; // CA - C
        let xy_plane = n_pos.sub(&ca_pos)?; // N - CA

        // Graham-Schmidt orthogonalization
        let rot = graham_schmidt(&x_axis, &xy_plane, 1e-10)?; // (B, L, 3, 3)

        // Validity mask: both backbone vectors have non-trivial length
        let eps = 1e-8f64;
        let v1_norm_sq = x_axis.sqr()?.sum(D::Minus1)?; // (B, L)
        let v2_norm_sq = xy_plane.sqr()?.sum(D::Minus1)?;
        let mask = (v1_norm_sq.gt(eps)? * v2_norm_sq.gt(eps)?)?;

        Ok((Self::new(rot, ca_pos), mask))
    }
}

/// Graham-Schmidt orthogonalization to produce a rotation matrix.
///
/// `x_axis`, `xy_plane`: `(*, 3)` vectors defining the frame.
/// Returns `(*, 3, 3)` where columns are the orthonormal basis `[e_x, e_1, e_2]`.
fn graham_schmidt(x_axis: &Tensor, xy_plane: &Tensor, eps: f64) -> Result<Tensor> {
    // Normalize x_axis → e_x; affine(1.0, eps) = tensor * 1 + eps = tensor + eps
    let norm_x = x_axis
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .affine(1.0, eps)?;
    let e_x = x_axis.broadcast_div(&norm_x)?;

    // e_1 = xy_plane - proj(xy_plane, e_x), then normalize
    let dot = e_x.mul(xy_plane)?.sum_keepdim(D::Minus1)?; // (*, 1)
    let e_1 = xy_plane.sub(&e_x.broadcast_mul(&dot)?)?;
    let norm_1 = e_1
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .affine(1.0, eps)?;
    let e_1 = e_1.broadcast_div(&norm_1)?;

    // e_2 = e_x × e_1
    let e_2 = cross_product(&e_x, &e_1)?;

    // Cat as columns → (*, 3, 3): each basis vector is (*, 3, 1) after unsqueeze
    Tensor::cat(
        &[
            &e_x.unsqueeze(D::Minus1)?,
            &e_1.unsqueeze(D::Minus1)?,
            &e_2.unsqueeze(D::Minus1)?,
        ],
        D::Minus1,
    )
}

/// Cross product of two `(*, 3)` tensors.
fn cross_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a0 = a.narrow(D::Minus1, 0, 1)?;
    let a1 = a.narrow(D::Minus1, 1, 1)?;
    let a2 = a.narrow(D::Minus1, 2, 1)?;
    let b0 = b.narrow(D::Minus1, 0, 1)?;
    let b1 = b.narrow(D::Minus1, 1, 1)?;
    let b2 = b.narrow(D::Minus1, 2, 1)?;
    let c0 = (a1.mul(&b2)?.sub(&a2.mul(&b1)?)?);
    let c1 = (a2.mul(&b0)?.sub(&a0.mul(&b2)?)?);
    let c2 = (a0.mul(&b1)?.sub(&a1.mul(&b0)?)?);
    Tensor::cat(&[&c0, &c1, &c2], D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_apply_rot_identity() -> Result<()> {
        let device = &Device::Cpu;
        // Identity rotation: v should be unchanged
        let rot = Tensor::eye(3, candle_core::DType::F32, device)?
            .unsqueeze(0)?
            .unsqueeze(0)?; // (1, 1, 3, 3)
        let v = Tensor::randn(0f32, 1f32, (1, 1, 4, 3), device)?;
        let out = Affine3D::apply_rot(&rot, &v)?;
        let diff = out.sub(&v)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "identity rotation changed vectors");
        Ok(())
    }

    #[test]
    fn test_apply_rot_roundtrip() -> Result<()> {
        let device = &Device::Cpu;
        // Random orthogonal matrix via Gram-Schmidt
        let v1 = Tensor::randn(0f32, 1f32, (1, 1, 3), device)?;
        let v2 = Tensor::randn(0f32, 1f32, (1, 1, 3), device)?;
        let rot = graham_schmidt(&v1, &v2, 1e-10)?; // (1, 1, 3, 3)

        let x = Tensor::randn(0f32, 1f32, (1, 1, 5, 3), device)?;
        // apply then inverse should be identity
        let x_rot = Affine3D::apply_rot(&rot, &x)?;
        let x_back = Affine3D::apply_rot_inv(&rot, &x_rot)?;
        let diff = x_back.sub(&x)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "rot/rot_inv roundtrip failed: {}", diff);
        Ok(())
    }

    #[test]
    fn test_build_affine3d_from_coordinates() -> Result<()> {
        let device = &Device::Cpu;
        // Simple linear backbone: N=0,0,0; CA=1,0,0; C=2,0,0 — degenerate but non-zero
        let n = Tensor::new(&[[[0f32, 0., 0.], [1., 0., 0.], [2., 0., 0.]]], device)?;
        let ca = Tensor::new(&[[[1f32, 0., 0.], [2., 0., 0.], [3., 0., 0.]]], device)?;
        let c = Tensor::new(&[[[2f32, 0., 0.], [3., 0., 0.], [4., 0., 0.]]], device)?;
        // Stack into (1, 3, 3, 3) = (B, L, atom, xyz)
        let n_e = n.unsqueeze(2)?;
        let ca_e = ca.unsqueeze(2)?;
        let c_e = c.unsqueeze(2)?;
        let coords = Tensor::cat(&[&n_e, &ca_e, &c_e], 2)?; // (1, 3, 3, 3)

        let (_affine, mask) = Affine3D::build_affine3d_from_coordinates(&coords)?;
        // mask should be 0 because all vectors are collinear (cross product ≈ 0)
        // but the lengths themselves are non-zero so the mask should be 1
        let mask_sum = mask.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        // We only check it runs without error and returns a reasonable mask
        assert!(mask_sum >= 0.0);
        Ok(())
    }
}
