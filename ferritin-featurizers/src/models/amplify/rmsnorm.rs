use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(dim: usize, vb: VarBuilder, eps: f64) -> Result<Self> {
        // Initialize the weight parameter with ones
        let weight = vb.get_with_hints(dim, "weight", candle_nn::init::Const::new(1.0))?;
        Ok(Self { weight, eps })
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // // Calculate the mean of squared values along the last dimension
        // let variance = x.pow_scalar(2.0)?.mean_keepdim(x.dim() - 1)?;

        // // Add epsilon and calculate reciprocal square root
        // let denom = (variance + self.eps)?.sqrt()?;
        // let normalized = x.div(&denom)?;

        // // Apply the learned weight parameter
        // normalized.mul(&self.weight)
        unimplemented!()
    }
}
