use crate::esm::models::esmc::ESMCConfig;
use candle_core::Tensor;
use candle_nn::{self as nn, LayerNormConfig, Module, Sequential, VarBuilder};

pub struct RegressionHead {
    model: Sequential,
}

impl RegressionHead {
    // pub fn new(d_model: usize, output_dim: usize, hidden_dim: Option<usize>) -> candle_core::Result<Self> {
    //     let hidden_dim = hidden_dim.unwrap_or(d_model);

    //     let model = Sequential::new(vec![
    //         Linear::new(d_model as usize, hidden_dim as usize)?.into(),
    //         candle_nn::Activation::Gelu.into(),
    //         candle_nn::LayerNorm::new(vec![hidden_dim])?.into(),
    //         Linear::new(hidden_dim as usize, output_dim as usize)?.into(),
    //     ]);

    //     Ok(Self { model })
    // }
    pub fn load(vb: VarBuilder, config: &ESMCConfig) -> candle_core::Result<Self> {
        let ESMCConfig {
            d_model,
            regression_head_output_dim,
            regression_head_hidden_dim,
            ..
        } = config;

        let linear1 = nn::linear(
            *d_model,
            *regression_head_hidden_dim,
            vb.pp("regression_linear"),
        )?;
        let gelu = candle_nn::Activation::Gelu;
        let ln_conf = LayerNormConfig::from(1e-5);
        let norm = nn::layer_norm(*regression_head_hidden_dim, ln_conf, vb.pp("layer_norm"))?;
        let linear2 = nn::linear(
            *regression_head_hidden_dim,
            *regression_head_output_dim,
            vb.pp("linear2"),
        )?;

        let model = nn::seq().add(linear1).add(gelu).add(norm).add(linear2);

        Ok(Self { model })
    }
}

impl Module for RegressionHead {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.model.forward(x)
    }
}
