use candle_nn::{Linear, Module, Sequential};
use candle_core::Tensor;

pub struct RegressionHead {
    model: Sequential,
}

impl RegressionHead {
    pub fn new(d_model: usize, output_dim: usize, hidden_dim: Option<usize>) -> candle_core::Result<Self> {
        let hidden_dim = hidden_dim.unwrap_or(d_model);

        let model = Sequential::new(vec![
            Linear::new(d_model as usize, hidden_dim as usize)?.into(),
            candle_nn::Activation::Gelu.into(),
            candle_nn::LayerNorm::new(vec![hidden_dim])?.into(),
            Linear::new(hidden_dim as usize, output_dim as usize)?.into(),
        ]);

        Ok(Self { model })
    }
}

impl Module for RegressionHead {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.model.forward(x)
    }
}
