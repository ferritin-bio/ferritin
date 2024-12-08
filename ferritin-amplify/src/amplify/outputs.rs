use super::config::AMPLIFYConfig;
use super::encoder::EncoderBlock;
use super::rotary::precompute_freqs_cis;
use super::
use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use tokenizers::Tokenizer;


// Helper structs and enums
#[derive(Debug)]
/// Amplify Model Output
///
/// logits, hidden states, and attentions.
///
///  logits -> distribution of the sequences.
///  attentions -> contact map
pub struct ModelOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

impl ModelOutput {
    /// "Perform average product correct, used for contact prediction."
    /// https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/examples/utils.py#L83
    /// "Perform average product correct, used for contact prediction."
    fn apc(&self, x: &Tensor) -> Result<Tensor> {
        let a1 = x.sum_keepdim(D::Minus1)?;
        let a2 = x.sum_keepdim(D::Minus2)?;
        let a12 = x.sum_keepdim((D::Minus1, D::Minus2))?;
        let avg = a1.matmul(&a2)?;
        // Divide by a12 (equivalent to pytorch's div_)
        // println!("IN the APC: avg, a12 {:?}, {:?}", avg, a12);
        // let avg = avg.div(&a12)?;
        let a12_broadcast = a12.broadcast_as(avg.shape())?;
        let avg = avg.div(&a12_broadcast)?;
        x.sub(&avg)
    }
    // From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
    // https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/examples/utils.py#L77
    // "Make layer symmetric in final two dimensions, used for contact prediction."
    fn symmetrize(&self, x: &Tensor) -> Result<Tensor> {
        let x_transpose = x.transpose(D::Minus1, D::Minus2)?;
        x.add(&x_transpose)
    }
    /// Contact maps can be obtained from the self-attentions
    pub fn get_contact_map(&self) -> Result<Option<Tensor>> {
        let Some(attentions) = &self.attentions else {
            return Ok(None);
        };
        // we need the dimensions to reshape below.
        // the attention blocks have the following shape
        let (_1, _n_head, _seq_length, seq_length) = attentions.first().unwrap().dims4()?;
        let last_dim = seq_length;
        let attn_stacked = Tensor::stack(attentions, 0)?;
        let total_elements = attn_stacked.dims().iter().product::<usize>();
        let first_dim = total_elements / (last_dim * last_dim);
        let attn_map_combined2 = attn_stacked.reshape(&[first_dim, last_dim, last_dim])?;

        // In PyTorch: attn_map = attn_map[:, 1:-1, 1:-1]
        let attn_map_combined2 = attn_map_combined2
            .narrow(1, 1, attn_map_combined2.dim(1)? - 2)? // second dim
            .narrow(2, 1, attn_map_combined2.dim(2)? - 2)?; // third dim
        let symmetric = self.symmetrize(&attn_map_combined2)?;
        let normalized = self.apc(&symmetric)?;
        let proximity_map = normalized.permute((1, 2, 0))?; //  # (residues, residues, map)

        Ok(Some(proximity_map))
    }
}
