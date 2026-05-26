use candle_core::{Module, Result, Tensor};
use candle_nn as nn;

// NOT CURRENTLY USED

pub struct SwiGLU {}

impl SwiGLU {
    pub fn new() -> Self {
        Self {}
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x1, x2) = x.chunk(2, -1)?;
        let hidden = x1.silu()?.mul(&x2)?;
        Ok(hidden)
    }
}

pub struct FFN {
    in_proj: Box<dyn Module>,
    activation: Box<dyn Module>,
    out_proj: Box<dyn Module>,
}

impl FFN {
    pub fn new(
        in_proj: Box<dyn Module>,
        activation: Box<dyn Module>,
        out_proj: Box<dyn Module>,
    ) -> Self {
        Self {
            in_proj,
            activation,
            out_proj,
        }
    }
}

impl Module for FFN {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.in_proj.forward(x)?;
        let x = self.activation.forward(&x)?;
        let x = self.out_proj.forward(&x)?;
        Ok(x)
    }
}
