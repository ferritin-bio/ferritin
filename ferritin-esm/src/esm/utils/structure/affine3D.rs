use candle_core::{DType, Device, Result, Shape, Tensor};
use std::fmt;
use thiserror::Error;

// Enhanced Rotation trait with additional functionality
pub trait Rotation: Sized + Clone {
    fn identity(shape: &[usize], device: &Device, dtype: DType) -> Result<Self>;
    fn random(shape: &[usize], device: &Device, dtype: DType) -> Result<Self>;
    fn get_slice(&self, idx: usize) -> Result<Self>;
    fn tensor(&self) -> &Tensor;
    fn shape(&self) -> &[usize];
    fn as_matrix(&self) -> Result<RotationMatrix>;
    fn compose(&self, other: &Self) -> Result<Self>;
    fn convert_compose(&self, other: &Self) -> Result<Self>;
    fn apply(&self, p: &Tensor) -> Result<Tensor>;
    fn invert(&self) -> Result<Self>;

    // New methods matching Python functionality
    fn dtype(&self) -> DType {
        self.tensor().dtype()
    }

    fn device(&self) -> &Device {
        self.tensor().device()
    }

    fn requires_grad(&self) -> bool {
        self.tensor().requires_grad()
    }

    fn to_dtype(&self, dtype: DType) -> Result<Self>;
    fn detach(&self) -> Result<Self>;
}

// Enhanced RotationMatrix implementation
#[derive(Clone)]
pub struct RotationMatrix {
    rots: Tensor,
}

impl RotationMatrix {
    pub fn new(rots: Tensor) -> Result<Self> {
        let shape = rots.dims();
        let rots = if shape[shape.len() - 1] == 9 {
            rots.reshape((shape[..shape.len() - 1].to_vec(), [3, 3]))?
        } else {
            rots
        };

        // Verify shape
        let shape = rots.dims();
        if shape[shape.len() - 1] != 3 || shape[shape.len() - 2] != 3 {
            return Err(RotationError::InvalidShape.into());
        }
        Ok(Self {
            rots: rots.to_dtype(DType::F32)?,
        })
    }

    pub fn to_3x3(&self) -> &Tensor {
        &self.rots
    }

    pub fn from_graham_schmidt(x_axis: &Tensor, xy_plane: &Tensor, eps: f64) -> Result<Self> {
        graham_schmidt(x_axis, xy_plane, Some(eps)).map(Self::new)?
    }
}

// Enhanced Affine3D implementation
#[derive(Clone)]
pub struct Affine3D {
    trans: Tensor,
    rot: Box<dyn Rotation>,
}

impl Affine3D {
    pub fn new(trans: Tensor, rot: Box<dyn Rotation>) -> Result<Self> {
        // Verify shapes match
        if trans.dims()[..trans.dims().len() - 1] != *rot.shape() {
            return Err(RotationError::ShapeMismatch.into());
        }
        Ok(Self { trans, rot })
    }

    pub fn identity(shape: &[usize], device: &Device, dtype: DType) -> Result<Self> {
        let mut trans_shape = shape.to_vec();
        trans_shape.push(3);
        let trans = Tensor::zeros(trans_shape.as_slice(), device)?.to_dtype(dtype)?;
        let rot = Box::new(RotationMatrix::identity(shape, device, dtype)?);
        Ok(Self { trans, rot })
    }

    pub fn random(shape: &[usize], std: f64, device: &Device, dtype: DType) -> Result<Self> {
        let mut trans_shape = shape.to_vec();
        trans_shape.push(3);
        let trans = Tensor::randn(0.0, std, trans_shape.as_slice(), device)?.to_dtype(dtype)?;
        let rot = Box::new(RotationMatrix::random(shape, device, dtype)?);
        Ok(Self { trans, rot })
    }

    pub fn compose(&self, other: &Self) -> Result<Self> {
        let new_rot = self.rot.compose(&*other.rot)?;
        let new_trans = self.rot.apply(&other.trans)?.add(&self.trans)?;
        Ok(Self {
            trans: new_trans,
            rot: Box::new(new_rot),
        })
    }

    pub fn invert(&self) -> Result<Self> {
        let inv_rot = self.rot.invert()?;
        let inv_trans = inv_rot.apply(&self.trans)?.neg()?;
        Ok(Self {
            trans: inv_trans,
            rot: Box::new(inv_rot),
        })
    }

    pub fn apply(&self, p: &Tensor) -> Result<Tensor> {
        self.rot.apply(p)?.add(&self.trans)
    }
}

#[derive(Error, Debug)]
pub enum RotationError {
    #[error("Invalid rotation matrix shape")]
    InvalidShape,
    #[error("Shape mismatch between translation and rotation")]
    ShapeMismatch,
}

// Helper function for Graham-Schmidt orthogonalization
fn graham_schmidt(x_axis: &Tensor, xy_plane: &Tensor, eps: Option<f64>) -> Result<Tensor> {
    let eps = eps.unwrap_or(1e-12);

    // Normalize x_axis
    let denom = x_axis
        .pow_scalar(2.0)?
        .sum_keepdim(-1)?
        .sqrt()?
        .add_scalar(eps)?;
    let x_axis = x_axis.div(&denom)?;

    // Calculate e1
    let dot = x_axis.mul(xy_plane)?.sum_keepdim(-1)?;
    let e1 = xy_plane.sub(&x_axis.mul(&dot)?)?;
    let denom = e1
        .pow_scalar(2.0)?
        .sum_keepdim(-1)?
        .sqrt()?
        .add_scalar(eps)?;
    let e1 = e1.div(&denom)?;

    // Calculate e2 via cross product
    let e2 = x_axis.cross(&e1)?;

    // Stack the basis vectors
    Tensor::stack(&[x_axis, e1, e2], -1)
}
