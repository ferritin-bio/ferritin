//! ESMFold2 configuration types.
//!
//! Defines [`ESMFold2Config`], which captures every hyper-parameter of the
//! ESMFold2-Fast architecture. All defaults come from the official
//! `biohub/ESMFold2-Fast` `config.json` and the architecture audit (2026-05-27).

/// Full configuration for ESMFold2-Fast (biohub/ESMFold2-Fast).
///
/// All values default to the ESMFold2-Fast configuration unless noted.
/// Source: biohub/ESMFold2-Fast config.json + architecture audit (2026-05-27).
#[derive(Debug, Clone)]
pub struct ESMFold2Config {
    // ---- ESMC-6B backbone ----
    /// HuggingFace repo for the frozen ESMC-6B encoder backbone.
    pub esmc_model_id: &'static str, // "biohub/ESMC-6B"
    pub lm_d_model: usize,    // 2560
    pub lm_num_layers: usize, // 80

    // ---- LM adapter (per recycling loop) ----
    pub lm_encoder_n_layers: usize, // 4  (projects 2560 → d_single=384)
    pub lm_dropout: f64,            // 0.25

    // ---- Token representation ----
    pub d_inputs: usize, // 451  (raw input feature dim)
    pub d_single: usize, // 384  (single/sequence repr dim after LM adapter)
    pub d_pair: usize,   // 256  (pair repr dim)

    // ---- Relative position encoding ----
    pub n_relative_residx_bins: usize, // 32
    pub n_relative_chain_bins: usize,  // 2

    // ---- Atom encoder ----
    pub d_atom: usize,                  // 128
    pub d_token_atom: usize,            // 768  (atom encoder output, projected to d_token)
    pub atom_encoder_n_blocks: usize,   // 3
    pub atom_encoder_n_heads: usize,    // 4
    pub atom_encoder_swa_window: usize, // 128

    // ---- Folding trunk ----
    pub trunk_n_layers: usize, // 24
    pub trunk_n_heads: usize,  // 8
    pub trunk_dropout: f64,    // 0.25

    // ---- MSA encoder (disabled in Fast) ----
    pub msa_enabled: bool, // false for ESMFold2-Fast

    // ---- Diffusion module ----
    pub c_token: usize,          // 768   (token dim inside diffusion)
    pub token_num_blocks: usize, // 12
    pub token_num_heads: usize,  // 16
    pub c_atom: usize,           // 128
    pub atom_num_blocks: usize,  // 3
    pub atom_num_heads: usize,   // 4
    pub fourier_dim: usize,      // 256   (noise level encoding)
    pub sigma_data: f64,         // 16.0  Å
    // Inference noise schedule
    pub inference_s_max: f64,       // 160.0
    pub inference_s_min: f64,       // 0.0004
    pub inference_num_steps: usize, // 14   (default; use 50 for quality)
    pub inference_p: f64,           // 7.0
    pub noise_scale: f64,           // 1.003
    pub gamma_0: f64,               // 0.8
    pub gamma_min: f64,             // 1.0
    pub step_scale: f64,            // 1.5

    // ---- Confidence head ----
    pub confidence_n_layers: usize, // 4
    pub num_plddt_bins: usize,      // 50
    pub num_pae_bins: usize,        // 64
    pub num_pde_bins: usize,        // 64
    pub distogram_bins: usize,      // 39
}

impl ESMFold2Config {
    /// Returns the ESMFold2-Fast configuration.
    ///
    /// Values are taken from `biohub/ESMFold2-Fast` `config.json` and the
    /// architecture audit dated 2026-05-27. Use `inference_num_steps = 50`
    /// (instead of the default 14) when higher structure quality is desired.
    pub fn fast() -> Self {
        Self {
            // ESMC-6B backbone
            esmc_model_id: "biohub/ESMC-6B",
            lm_d_model: 2560,
            lm_num_layers: 80,

            // LM adapter
            lm_encoder_n_layers: 4,
            lm_dropout: 0.25,

            // Token representation
            d_inputs: 451,
            d_single: 384,
            d_pair: 256,

            // Relative position encoding
            n_relative_residx_bins: 32,
            n_relative_chain_bins: 2,

            // Atom encoder
            d_atom: 128,
            d_token_atom: 768,
            atom_encoder_n_blocks: 3,
            atom_encoder_n_heads: 4,
            atom_encoder_swa_window: 128,

            // Folding trunk
            trunk_n_layers: 24,
            trunk_n_heads: 8,
            trunk_dropout: 0.25,

            // MSA encoder disabled in Fast variant
            msa_enabled: false,

            // Diffusion module
            c_token: 768,
            token_num_blocks: 12,
            token_num_heads: 16,
            c_atom: 128,
            atom_num_blocks: 3,
            atom_num_heads: 4,
            fourier_dim: 256,
            sigma_data: 16.0,

            // Inference noise schedule
            inference_s_max: 160.0,
            inference_s_min: 0.0004,
            inference_num_steps: 14,
            inference_p: 7.0,
            noise_scale: 1.003,
            gamma_0: 0.8,
            gamma_min: 1.0,
            step_scale: 1.5,

            // Confidence head
            confidence_n_layers: 4,
            num_plddt_bins: 50,
            num_pae_bins: 64,
            num_pde_bins: 64,
            distogram_bins: 39,
        }
    }
}
