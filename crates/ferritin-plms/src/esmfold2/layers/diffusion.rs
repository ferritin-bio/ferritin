//! AF3-style EDM diffusion module for all-atom coordinate generation.
//!
//! Architecture:
//! 1. Single repr projected to token dim: `[B, N, d_single] → [B, N, c_token=768]`
//! 2. Fourier noise embedding encodes σ_t → `[fourier_dim=256]`
//! 3. 12 token-level transformer blocks (c_token=768, 16 heads, with pair bias)
//! 4. 3 atom-level transformer blocks (c_atom=128, 4 heads) — TODO
//! 5. Output projection: `[B, N, c_token] → [B, N*n_atoms_per_token, 3]`
//!
//! EDM noise schedule (inference):
//! ```text
//! sigma_t = s_max * (s_min / s_max)^((t / T)^p)
//!   s_max = 160.0,  s_min = 0.0004,  T = num_steps,  p = 7.0
//!
//! Stochastic correction: gamma_t = min(gamma_0, sqrt(sigma_next/sigma_t) - 1)
//! sigma_hat = sigma_t * (1 + gamma_t)
//! x_hat = x + sqrt(sigma_hat^2 - sigma_t^2) * noise * noise_scale
//! D = score_network(x_hat, sigma_hat)
//! x = x_hat + step_scale * (D - x_hat) * (sigma_next / sigma_hat - 1)
//! ```
//!
//! Weight layout (rooted at `structure_head`):
//! ```text
//! token_proj.*                            — d_single → c_token (no bias)
//! noise_embedding.*                       — Fourier frequencies (learnable)
//! noise_proj.*                            — fourier_dim → c_token (no bias)
//! token_transformer.blocks.{0..11}.*     — 12 token transformer blocks
//! token_transformer.blocks.{i}.norm.*
//! token_transformer.blocks.{i}.attn.*    — MHA with pair bias
//! token_transformer.blocks.{i}.ffn.*     — SwiGLU FFN
//! out_proj.*                             — c_token → 3 (Cα coords, no bias)
//! ```

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{self as nn, LayerNorm, LayerNormConfig, Module, VarBuilder};

// ── Fourier Noise Embedding ───────────────────────────────────────────────────

/// Learnable Fourier embedding for the noise level σ.
///
/// Maps a scalar σ to `[fourier_dim]` features:
/// `[sin(2π σ w_0), cos(2π σ w_0), ..., sin(2π σ w_{d/2-1}), cos(2π σ w_{d/2-1})]`
/// where `w_i` are learnable scalar weights.
struct FourierEmbedding {
    weights: Tensor, // [fourier_dim / 2]
    #[allow(dead_code)]
    dim: usize,
}

impl FourierEmbedding {
    fn load(vb: VarBuilder, fourier_dim: usize) -> Result<Self> {
        let weights = vb.get((fourier_dim / 2,), "weights")?;
        Ok(Self {
            weights,
            dim: fourier_dim,
        })
    }

    /// `sigma` — scalar (f32); returns `[fourier_dim]`
    fn embed(&self, sigma: f32) -> Result<Tensor> {
        let device = self.weights.device();
        let sigma_t = Tensor::from_vec(vec![sigma], 1, device)?.to_dtype(DType::F32)?;
        // [1] * [d/2] → [d/2]  (broadcast multiply)
        let x = sigma_t.broadcast_mul(&self.weights)?;
        let x = (x * (2.0 * std::f64::consts::PI))?;
        let sin_x = x.sin()?; // [d/2]
        let cos_x = x.cos()?; // [d/2]
        Tensor::cat(&[&sin_x, &cos_x], 0) // [fourier_dim]
    }
}

// ── Token Transformer Block ───────────────────────────────────────────────────

/// Pre-norm MHA block at token level, conditioned by the pair representation.
///
/// Follows the same pattern as LMEncoder blocks but with pair bias conditioning.
struct TokenAttn {
    norm: LayerNorm,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    pair_bias: nn::Linear, // d_pair → n_heads
    gate: nn::Linear,
    out_proj: nn::Linear,
    n_heads: usize,
    d_head: usize,
}

impl TokenAttn {
    fn load(vb: VarBuilder, c_token: usize, n_heads: usize, d_pair: usize) -> Result<Self> {
        let d_head = c_token / n_heads;
        Ok(Self {
            norm: nn::layer_norm(c_token, LayerNormConfig::from(1e-5), vb.pp("norm"))?,
            q_proj: nn::linear_no_bias(c_token, n_heads * d_head, vb.pp("q_proj"))?,
            k_proj: nn::linear_no_bias(c_token, n_heads * d_head, vb.pp("k_proj"))?,
            v_proj: nn::linear_no_bias(c_token, n_heads * d_head, vb.pp("v_proj"))?,
            pair_bias: nn::linear_no_bias(d_pair, n_heads, vb.pp("pair_bias"))?,
            gate: nn::linear_no_bias(c_token, n_heads * d_head, vb.pp("gate"))?,
            out_proj: nn::linear_no_bias(n_heads * d_head, c_token, vb.pp("out_proj"))?,
            n_heads,
            d_head,
        })
    }

    /// `x`: `[B, N, c_token]`, `pair`: `[B, N, N, d_pair]`
    fn forward(&self, x: &Tensor, pair: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let (h, dh) = (self.n_heads, self.d_head);

        let x_n = self.norm.forward(x)?;

        let q = self.q_proj.forward(&x_n)?; // [B, N, H*dh]
        let k = self.k_proj.forward(&x_n)?;
        let v = self.v_proj.forward(&x_n)?;
        let gate = nn::ops::sigmoid(&self.gate.forward(&x_n)?)?; // [B, N, H*dh]

        // pair bias: [B, N, N, d_pair] → [B, N, N, H] → [B, H, N, N]
        let pair_b = self
            .pair_bias
            .forward(pair)? // [B, N, N, H]
            .permute((0, 3, 1, 2))?
            .contiguous()?; // [B, H, N, N]

        // → [B, H, N, dh]
        let to_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, n, h, dh))?
                .permute((0, 2, 1, 3))?
                .contiguous()
        };
        let q = to_heads(q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;

        let scale = (dh as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / scale)?;
        // scores: [B, H, N, N]; pair_b: [B, H, N, N]
        let scores = (scores + pair_b)?;
        let attn = nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // [B, H, N, dh]

        // [B, H, N, dh] → [B, N, H*dh]
        let out = out
            .permute((0, 2, 1, 3))?
            .contiguous()? // [B, N, H, dh]
            .reshape((b, n, h * dh))?;

        let out = (gate * out)?;
        let out = self.out_proj.forward(&out)?;
        x + &out
    }
}

struct TokenFfn {
    norm: LayerNorm,
    gate_up: nn::Linear, // c_token → 2 * hidden
    down: nn::Linear,    // hidden → c_token
}

impl TokenFfn {
    fn load(vb: VarBuilder, c_token: usize) -> Result<Self> {
        // SwiGLU hidden: nearest-256 multiple of (c_token * 8/3)
        let hidden = ((8.0 / 3.0 * c_token as f64 + 255.0) / 256.0).floor() as usize * 256;
        Ok(Self {
            norm: nn::layer_norm(c_token, LayerNormConfig::from(1e-5), vb.pp("norm"))?,
            gate_up: nn::linear_no_bias(c_token, hidden * 2, vb.pp("gate_up"))?,
            down: nn::linear_no_bias(hidden, c_token, vb.pp("down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.gate_up.forward(&self.norm.forward(x)?)?;
        let chunks = h.chunk(2, D::Minus1)?;
        let act = (chunks[0].silu()? * &chunks[1])?;
        let out = self.down.forward(&act)?;
        x + &out
    }
}

struct TokenBlock {
    attn: TokenAttn,
    ffn: TokenFfn,
}

impl TokenBlock {
    fn load(vb: VarBuilder, c_token: usize, n_heads: usize, d_pair: usize) -> Result<Self> {
        Ok(Self {
            attn: TokenAttn::load(vb.pp("attn"), c_token, n_heads, d_pair)?,
            ffn: TokenFfn::load(vb.pp("ffn"), c_token)?,
        })
    }

    fn forward(&self, x: &Tensor, pair: &Tensor) -> Result<Tensor> {
        let x = self.attn.forward(x, pair)?;
        self.ffn.forward(&x)
    }
}

// ── DiffusionModule ───────────────────────────────────────────────────────────

/// AF3-style EDM diffusion module.
///
/// Denoises atom coordinates conditioned on single and pair representations
/// produced by the folding trunk. Uses 12 token-level transformer blocks with
/// pair bias conditioning, followed by coordinate output projection.
pub struct DiffusionModule {
    token_proj: nn::Linear, // d_single → c_token
    noise_emb: FourierEmbedding,
    noise_proj: nn::Linear,  // fourier_dim → c_token
    blocks: Vec<TokenBlock>, // 12 blocks
    out_proj: nn::Linear,    // c_token → 3 (Cα or token centroid)
    // TODO: atom_transformer (3 blocks, c_atom=128) for all-atom output
    #[allow(dead_code)]
    c_token: usize,
    #[allow(dead_code)]
    d_single: usize,
    // Noise schedule parameters
    s_max: f64,
    s_min: f64,
    p: f64,
    gamma_0: f64,
    noise_scale: f64,
    step_scale: f64,
    device: candle_core::Device,
}

impl DiffusionModule {
    /// Load the diffusion module from a `VarBuilder` rooted at `structure_head.*`.
    ///
    /// # Arguments
    /// * `vb`       — builder rooted at `structure_head`
    /// * `c_token`  — token channel dimension (768)
    /// * `c_atom`   — atom channel dimension (128, reserved for atom transformer TODO)
    /// * `d_single` — single repr dimension from folding trunk (384)
    /// * `d_pair`   — pair repr dimension from folding trunk (256)
    /// * `n_token_blocks` — number of token transformer blocks (12)
    /// * `n_token_heads`  — number of heads per token block (16)
    /// * `fourier_dim`    — Fourier noise embedding dimension (256)
    #[allow(clippy::too_many_arguments)] // one arg per diffusion-module hyperparameter
    pub fn load(
        vb: VarBuilder,
        c_token: usize,
        c_atom: usize,
        d_single: usize,
        d_pair: usize,
        n_token_blocks: usize,
        n_token_heads: usize,
        fourier_dim: usize,
    ) -> Result<Self> {
        let _ = c_atom; // reserved for atom transformer

        let token_proj = nn::linear_no_bias(d_single, c_token, vb.pp("token_proj"))?;
        let noise_emb = FourierEmbedding::load(vb.pp("noise_embedding"), fourier_dim)?;
        let noise_proj = nn::linear_no_bias(fourier_dim, c_token, vb.pp("noise_proj"))?;

        let blocks = (0..n_token_blocks)
            .map(|i| {
                TokenBlock::load(
                    vb.pp(format!("token_transformer.blocks.{i}")),
                    c_token,
                    n_token_heads,
                    d_pair,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let out_proj = nn::linear_no_bias(c_token, 3, vb.pp("out_proj"))?;
        let device = vb.device().clone();

        Ok(Self {
            token_proj,
            noise_emb,
            noise_proj,
            blocks,
            out_proj,
            c_token,
            d_single,
            s_max: 160.0,
            s_min: 0.0004,
            p: 7.0,
            gamma_0: 0.8,
            noise_scale: 1.003,
            step_scale: 1.5,
            device,
        })
    }

    /// Compute sigma schedule: `s_max * (s_min/s_max)^((t/T)^p)` for t in 0..T.
    fn sigma_schedule(&self, num_steps: usize) -> Vec<f32> {
        (0..=num_steps)
            .map(|t| {
                let frac = t as f64 / num_steps as f64;
                (self.s_max * (self.s_min / self.s_max).powf(frac.powf(self.p))) as f32
            })
            .collect()
    }

    /// Run the score network: projects single, embeds σ, runs transformer blocks.
    ///
    /// Input:  `noisy_coords [B, N, 3]`,  `single [B, N, d_single]`,  `pair [B, N, N, d_pair]`
    /// Output: `denoised_coords [B, N, 3]`
    fn score_network(
        &self,
        noisy: &Tensor,
        sigma: f32,
        single: &Tensor,
        pair: &Tensor,
    ) -> Result<Tensor> {
        let (b, n, _) = single.dims3()?;

        // Project single to token dim and inject noise level
        let mut tok = self.token_proj.forward(single)?; // [B, N, c_token]
        let noise_feat = self
            .noise_proj
            .forward(&self.noise_emb.embed(sigma)?.unsqueeze(0)?.unsqueeze(0)?)?; // [1, 1, c_token]
        tok = tok.broadcast_add(&noise_feat)?;

        // Run 12 transformer blocks
        for block in &self.blocks {
            tok = block.forward(&tok, pair)?;
        }

        // Project to 3D coordinates
        let coords = self.out_proj.forward(&tok)?; // [B, N, 3]

        // Rescale output: coords are expressed in the frame where sigma=1;
        // multiply by sigma for correct scale  (simplified AF3 preconditioner)
        let scale = Tensor::from_vec(vec![sigma], 1, &self.device)?.to_dtype(DType::F32)?;
        let _ = (b, n, noisy); // noisy coords available if needed for skip connection
        coords.broadcast_mul(&scale.reshape((1, 1, 1))?.broadcast_as(coords.shape())?)
    }

    /// Run the AF3-style EDM denoising loop.
    ///
    /// # Arguments
    /// * `single`    — single representation `[B, N_tok, d_single]`
    /// * `pair`      — pair representation `[B, N_tok, N_tok, d_pair]`
    /// * `n_atoms`   — number of atoms (= N_tok for Cα-only, otherwise N_tok * atoms_per_tok)
    /// * `num_steps` — number of denoising steps (14 fast / 50 quality)
    ///
    /// # Returns
    /// Token-level coordinates `[B, N_tok, 3]` in Ångströms (Cα output).
    pub fn forward(
        &self,
        single: &Tensor,
        pair: &Tensor,
        n_atoms: usize,
        num_steps: usize,
    ) -> Result<Tensor> {
        let (b, _n_tok, _) = single.dims3()?;
        let sigmas = self.sigma_schedule(num_steps);

        // Initialise x ~ N(0, sigma_0^2 * I)
        let sigma_0 = sigmas[0];
        let mut x = Tensor::randn(0f32, sigma_0, (b, n_atoms, 3_usize), &self.device)?;

        for step in 0..num_steps {
            let sigma_t = sigmas[step];
            let sigma_next = sigmas[step + 1];

            // Stochastic correction factor (gamma from DDPM-flavored EDM)
            let gamma = (self.gamma_0 as f32)
                .min((sigma_next / sigma_t).sqrt() - 1.0)
                .max(0.0);
            let sigma_hat = sigma_t * (1.0 + gamma);

            // Add stochastic noise when gamma > 0
            let x_hat = if gamma > 0.0 {
                let extra_noise_std =
                    (sigma_hat * sigma_hat - sigma_t * sigma_t).sqrt() * self.noise_scale as f32;
                let noise = Tensor::randn(0f32, extra_noise_std, x.shape(), &self.device)?;
                (&x + &noise)?
            } else {
                x.clone()
            };

            // Score network forward
            let d = self.score_network(&x_hat, sigma_hat, single, pair)?;

            // DDPM update step
            let ratio = sigma_next / sigma_hat;
            let step_factor = self.step_scale * (ratio as f64 - 1.0); // f64 for Tensor::Mul<f64>
            // x = x_hat + step_factor * (d - x_hat)
            let diff = (d - &x_hat)?;
            x = (x_hat + (diff * step_factor)?)?;
        }

        Ok(x)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    const B: usize = 1;
    const N: usize = 6;
    const D_SINGLE: usize = 64; // small for fast tests
    const D_PAIR: usize = 32;
    const C_TOKEN: usize = 64;
    const C_ATOM: usize = 16;
    const N_HEADS: usize = 4;
    const FOURIER_DIM: usize = 16;
    const N_BLOCKS: usize = 2; // use 2 blocks in tests (not 12)

    fn make_module(device: &Device) -> DiffusionModule {
        let vb = VarBuilder::zeros(DType::F32, device);
        DiffusionModule::load(
            vb,
            C_TOKEN,
            C_ATOM,
            D_SINGLE,
            D_PAIR,
            N_BLOCKS,
            N_HEADS,
            FOURIER_DIM,
        )
        .unwrap()
    }

    #[test]
    fn test_fourier_embedding_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = FourierEmbedding::load(vb, FOURIER_DIM).unwrap();
        let out = emb.embed(1.0).unwrap();
        assert_eq!(out.dims(), &[FOURIER_DIM]);
    }

    #[test]
    fn test_sigma_schedule_length() {
        let device = Device::Cpu;
        let m = make_module(&device);
        let sigmas = m.sigma_schedule(14);
        assert_eq!(sigmas.len(), 15); // num_steps + 1 (includes sigma_T = s_min)
        assert!(sigmas[0] > sigmas[14], "sigma should decrease");
    }

    #[test]
    fn test_sigma_schedule_bounds() {
        let device = Device::Cpu;
        let m = make_module(&device);
        let sigmas = m.sigma_schedule(14);
        // sigma_0 ≈ s_max, sigma_T ≈ s_min
        assert!(sigmas[0] > 100.0, "first sigma near s_max=160");
        assert!(sigmas[14] < 1.0, "last sigma near s_min=0.0004");
    }

    #[test]
    fn test_token_block_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = TokenBlock::load(vb, C_TOKEN, N_HEADS, D_PAIR).unwrap();
        let x = Tensor::zeros(&[B, N, C_TOKEN], DType::F32, &device).unwrap();
        let pair = Tensor::zeros(&[B, N, N, D_PAIR], DType::F32, &device).unwrap();
        let out = block.forward(&x, &pair).unwrap();
        assert_eq!(out.dims(), &[B, N, C_TOKEN]);
    }

    #[test]
    fn test_diffusion_forward_shape() {
        let device = Device::Cpu;
        let m = make_module(&device);
        let single = Tensor::zeros(&[B, N, D_SINGLE], DType::F32, &device).unwrap();
        let pair = Tensor::zeros(&[B, N, N, D_PAIR], DType::F32, &device).unwrap();
        let out = m.forward(&single, &pair, N, 2).unwrap();
        assert_eq!(out.dims(), &[B, N, 3]);
    }

    #[test]
    fn test_diffusion_batch_shape() {
        let device = Device::Cpu;
        let m = make_module(&device);
        let single = Tensor::zeros(&[2, N, D_SINGLE], DType::F32, &device).unwrap();
        let pair = Tensor::zeros(&[2, N, N, D_PAIR], DType::F32, &device).unwrap();
        let out = m.forward(&single, &pair, N, 2).unwrap();
        assert_eq!(out.dims(), &[2, N, 3]);
    }
}
