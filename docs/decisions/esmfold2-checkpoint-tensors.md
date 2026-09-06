# ESMFold2-Fast checkpoint tensor inventory

Ground truth read from `biohub/ESMFold2-Fast` `model.safetensors` on 2026-09-05.
Numeric path segments are collapsed to `{i}`. See [esmfold2-port-mismatch.md](./esmfold2-port-mismatch.md).

Total tensors: 1032

### `confidence_head` — 101 tensors

| path pattern | shape | count |
|---|---|---|
| `confidence_head.boundaries` | `[38]` | 1 |
| `confidence_head.dist_bin_pairwise_embed.weight` | `[39, 256]` | 1 |
| `confidence_head.folding_trunk.blocks.{i}.pair_transition.ffn.w12.weight` | `[2048, 256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.pair_transition.ffn.w3.weight` | `[256, 1024]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.pair_transition.norm.bias` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.pair_transition.norm.weight` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_in._engine.norm_mix.bias` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_in._engine.norm_mix.weight` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_in._engine.norm_start.bias` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_in._engine.norm_start.weight` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_in._engine.proj_bundle.weight` | `[1024, 256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_in._engine.proj_emit.weight` | `[256, 256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_in._engine.proj_gate.weight` | `[256, 256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_out._engine.norm_mix.bias` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_out._engine.norm_mix.weight` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_out._engine.norm_start.bias` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_out._engine.norm_start.weight` | `[256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_out._engine.proj_bundle.weight` | `[1024, 256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_out._engine.proj_emit.weight` | `[256, 256]` | 4 |
| `confidence_head.folding_trunk.blocks.{i}.tri_mul_out._engine.proj_gate.weight` | `[256, 256]` | 4 |
| `confidence_head.pae_head.weight` | `[64, 256]` | 1 |
| `confidence_head.pae_ln.bias` | `[256]` | 1 |
| `confidence_head.pae_ln.weight` | `[256]` | 1 |
| `confidence_head.pde_head.weight` | `[64, 256]` | 1 |
| `confidence_head.pde_ln.bias` | `[256]` | 1 |
| `confidence_head.pde_ln.weight` | `[256]` | 1 |
| `confidence_head.plddt_ln.bias` | `[384]` | 1 |
| `confidence_head.plddt_ln.weight` | `[384]` | 1 |
| `confidence_head.plddt_weight` | `[23, 384, 50]` | 1 |
| `confidence_head.resolved_ln.bias` | `[384]` | 1 |
| `confidence_head.resolved_ln.weight` | `[384]` | 1 |
| `confidence_head.resolved_weight` | `[23, 384, 2]` | 1 |
| `confidence_head.row_attention_pooling.attn_proj.weight` | `[1, 256]` | 1 |
| `confidence_head.row_attention_pooling.out_proj.weight` | `[384, 256]` | 1 |
| `confidence_head.s_input_to_s.weight` | `[384, 451]` | 1 |
| `confidence_head.s_inputs_norm.bias` | `[451]` | 1 |
| `confidence_head.s_inputs_norm.weight` | `[451]` | 1 |
| `confidence_head.s_inputs_to_single.weight` | `[384, 451]` | 1 |
| `confidence_head.s_norm.bias` | `[384]` | 1 |
| `confidence_head.s_norm.weight` | `[384]` | 1 |
| `confidence_head.s_to_z.weight` | `[256, 451]` | 1 |
| `confidence_head.s_to_z_prod_in1.weight` | `[256, 451]` | 1 |
| `confidence_head.s_to_z_prod_in2.weight` | `[256, 451]` | 1 |
| `confidence_head.s_to_z_prod_out.weight` | `[256, 256]` | 1 |
| `confidence_head.s_to_z_transpose.weight` | `[256, 451]` | 1 |
| `confidence_head.z_norm.bias` | `[256]` | 1 |
| `confidence_head.z_norm.weight` | `[256]` | 1 |

### `distogram_head` — 2 tensors

| path pattern | shape | count |
|---|---|---|
| `distogram_head.bias` | `[64]` | 1 |
| `distogram_head.weight` | `[64, 256]` | 1 |

### `folding_trunk` — 432 tensors

| path pattern | shape | count |
|---|---|---|
| `folding_trunk.blocks.{i}.pair_transition.ffn.w12.weight` | `[2048, 256]` | 24 |
| `folding_trunk.blocks.{i}.pair_transition.ffn.w3.weight` | `[256, 1024]` | 24 |
| `folding_trunk.blocks.{i}.pair_transition.norm.bias` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.pair_transition.norm.weight` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_in._engine.norm_mix.bias` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_in._engine.norm_mix.weight` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_in._engine.norm_start.bias` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_in._engine.norm_start.weight` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_in._engine.proj_bundle.weight` | `[1024, 256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_in._engine.proj_emit.weight` | `[256, 256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_in._engine.proj_gate.weight` | `[256, 256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_out._engine.norm_mix.bias` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_out._engine.norm_mix.weight` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_out._engine.norm_start.bias` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_out._engine.norm_start.weight` | `[256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_out._engine.proj_bundle.weight` | `[1024, 256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_out._engine.proj_emit.weight` | `[256, 256]` | 24 |
| `folding_trunk.blocks.{i}.tri_mul_out._engine.proj_gate.weight` | `[256, 256]` | 24 |

### `inputs_embedder` — 22 tensors

| path pattern | shape | count |
|---|---|---|
| `inputs_embedder.atom_attention_encoder.atom_linear.weight` | `[128, 389]` | 1 |
| `inputs_embedder.atom_attention_encoder.atom_norm.bias` | `[128]` | 1 |
| `inputs_embedder.atom_attention_encoder.atom_norm.weight` | `[128]` | 1 |
| `inputs_embedder.atom_attention_encoder.atom_to_token_linear.weight` | `[384, 128]` | 1 |
| `inputs_embedder.atom_attention_encoder.atom_transformer.blocks.{i}.adaln_modulation.{i}.weight` | `[768, 128]` | 3 |
| `inputs_embedder.atom_attention_encoder.atom_transformer.blocks.{i}.attn.Wqkv.weight` | `[384, 128]` | 3 |
| `inputs_embedder.atom_attention_encoder.atom_transformer.blocks.{i}.attn.gate_proj.weight` | `[128, 128]` | 3 |
| `inputs_embedder.atom_attention_encoder.atom_transformer.blocks.{i}.attn.out_proj.weight` | `[128, 128]` | 3 |
| `inputs_embedder.atom_attention_encoder.atom_transformer.blocks.{i}.ffn.w_down.weight` | `[128, 256]` | 3 |
| `inputs_embedder.atom_attention_encoder.atom_transformer.blocks.{i}.ffn.w_up.weight` | `[512, 128]` | 3 |

### `language_model` — 12 tensors

| path pattern | shape | count |
|---|---|---|
| `language_model.base_z_combine` | `[81]` | 1 |
| `language_model.base_z_linear.{i}.bias` | `[2560]` | 1 |
| `language_model.base_z_linear.{i}.weight` | `[2560]` | 2 |
| `language_model.base_z_mlp.{i}.downproject.bias` | `[256]` | 1 |
| `language_model.base_z_mlp.{i}.downproject.weight` | `[256, 256]` | 1 |
| `language_model.base_z_mlp.{i}.output_mlp.{i}.bias` | `[256]` | 2 |
| `language_model.base_z_mlp.{i}.output_mlp.{i}.weight` | `[256, 512]` | 2 |
| `language_model.base_z_mlp.{i}.bias` | `[256]` | 1 |
| `language_model.base_z_mlp.{i}.weight` | `[256]` | 1 |

### `lm_encoder` — 72 tensors

| path pattern | shape | count |
|---|---|---|
| `lm_encoder.blocks.{i}.pair_transition.ffn.w12.weight` | `[2048, 256]` | 4 |
| `lm_encoder.blocks.{i}.pair_transition.ffn.w3.weight` | `[256, 1024]` | 4 |
| `lm_encoder.blocks.{i}.pair_transition.norm.bias` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.pair_transition.norm.weight` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_in._engine.norm_mix.bias` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_in._engine.norm_mix.weight` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_in._engine.norm_start.bias` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_in._engine.norm_start.weight` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_in._engine.proj_bundle.weight` | `[1024, 256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_in._engine.proj_emit.weight` | `[256, 256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_in._engine.proj_gate.weight` | `[256, 256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_out._engine.norm_mix.bias` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_out._engine.norm_mix.weight` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_out._engine.norm_start.bias` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_out._engine.norm_start.weight` | `[256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_out._engine.proj_bundle.weight` | `[1024, 256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_out._engine.proj_emit.weight` | `[256, 256]` | 4 |
| `lm_encoder.blocks.{i}.tri_mul_out._engine.proj_gate.weight` | `[256, 256]` | 4 |

### `parcae_b_cont` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `parcae_b_cont` | `[256, 256]` | 1 |

### `parcae_coda` — 36 tensors

| path pattern | shape | count |
|---|---|---|
| `parcae_coda.blocks.{i}.pair_transition.ffn.w12.weight` | `[2048, 256]` | 2 |
| `parcae_coda.blocks.{i}.pair_transition.ffn.w3.weight` | `[256, 1024]` | 2 |
| `parcae_coda.blocks.{i}.pair_transition.norm.bias` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.pair_transition.norm.weight` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_in._engine.norm_mix.bias` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_in._engine.norm_mix.weight` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_in._engine.norm_start.bias` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_in._engine.norm_start.weight` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_in._engine.proj_bundle.weight` | `[1024, 256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_in._engine.proj_emit.weight` | `[256, 256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_in._engine.proj_gate.weight` | `[256, 256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_out._engine.norm_mix.bias` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_out._engine.norm_mix.weight` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_out._engine.norm_start.bias` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_out._engine.norm_start.weight` | `[256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_out._engine.proj_bundle.weight` | `[1024, 256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_out._engine.proj_emit.weight` | `[256, 256]` | 2 |
| `parcae_coda.blocks.{i}.tri_mul_out._engine.proj_gate.weight` | `[256, 256]` | 2 |

### `parcae_input_norm` — 2 tensors

| path pattern | shape | count |
|---|---|---|
| `parcae_input_norm.bias` | `[256]` | 1 |
| `parcae_input_norm.weight` | `[256]` | 1 |

### `parcae_log_a` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `parcae_log_a` | `[256]` | 1 |

### `parcae_log_delta` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `parcae_log_delta` | `[256]` | 1 |

### `parcae_readout` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `parcae_readout.weight` | `[256, 256]` | 1 |

### `rel_pos` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `rel_pos.embed.weight` | `[256, 139]` | 1 |

### `structure_head` — 345 tensors

| path pattern | shape | count |
|---|---|---|
| `structure_head.diffusion_module.atom_decoder.atom_transformer.blocks.{i}.adaln_modulation.{i}.weight` | `[768, 128]` | 3 |
| `structure_head.diffusion_module.atom_decoder.atom_transformer.blocks.{i}.attn.Wqkv.weight` | `[384, 128]` | 3 |
| `structure_head.diffusion_module.atom_decoder.atom_transformer.blocks.{i}.attn.gate_proj.weight` | `[128, 128]` | 3 |
| `structure_head.diffusion_module.atom_decoder.atom_transformer.blocks.{i}.attn.out_proj.weight` | `[128, 128]` | 3 |
| `structure_head.diffusion_module.atom_decoder.atom_transformer.blocks.{i}.ffn.w_down.weight` | `[128, 256]` | 3 |
| `structure_head.diffusion_module.atom_decoder.atom_transformer.blocks.{i}.ffn.w_up.weight` | `[512, 128]` | 3 |
| `structure_head.diffusion_module.atom_decoder.norm.bias` | `[128]` | 1 |
| `structure_head.diffusion_module.atom_decoder.norm.weight` | `[128]` | 1 |
| `structure_head.diffusion_module.atom_decoder.output_linear.weight` | `[3, 128]` | 1 |
| `structure_head.diffusion_module.atom_decoder.token_to_atom_linear.weight` | `[128, 768]` | 1 |
| `structure_head.diffusion_module.atom_encoder.atom_linear.weight` | `[128, 389]` | 1 |
| `structure_head.diffusion_module.atom_encoder.atom_norm.bias` | `[128]` | 1 |
| `structure_head.diffusion_module.atom_encoder.atom_norm.weight` | `[128]` | 1 |
| `structure_head.diffusion_module.atom_encoder.atom_to_token_linear.weight` | `[768, 128]` | 1 |
| `structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.{i}.adaln_modulation.{i}.weight` | `[768, 128]` | 3 |
| `structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.{i}.attn.Wqkv.weight` | `[384, 128]` | 3 |
| `structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.{i}.attn.gate_proj.weight` | `[128, 128]` | 3 |
| `structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.{i}.attn.out_proj.weight` | `[128, 128]` | 3 |
| `structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.{i}.ffn.w_down.weight` | `[128, 256]` | 3 |
| `structure_head.diffusion_module.atom_encoder.atom_transformer.blocks.{i}.ffn.w_up.weight` | `[512, 128]` | 3 |
| `structure_head.diffusion_module.atom_encoder.coords_linear.weight` | `[128, 6]` | 1 |
| `structure_head.diffusion_module.conditioning.fourier.b` | `[256]` | 1 |
| `structure_head.diffusion_module.conditioning.fourier.w` | `[256]` | 1 |
| `structure_head.diffusion_module.conditioning.noise_norm.bias` | `[256]` | 1 |
| `structure_head.diffusion_module.conditioning.noise_norm.weight` | `[256]` | 1 |
| `structure_head.diffusion_module.conditioning.noise_proj.weight` | `[768, 256]` | 1 |
| `structure_head.diffusion_module.conditioning.s_input_norm.bias` | `[451]` | 1 |
| `structure_head.diffusion_module.conditioning.s_input_norm.weight` | `[451]` | 1 |
| `structure_head.diffusion_module.conditioning.s_proj.weight` | `[768, 451]` | 1 |
| `structure_head.diffusion_module.conditioning.s_transitions.{i}.a_proj.weight` | `[1536, 768]` | 2 |
| `structure_head.diffusion_module.conditioning.s_transitions.{i}.b_proj.weight` | `[1536, 768]` | 2 |
| `structure_head.diffusion_module.conditioning.s_transitions.{i}.norm.bias` | `[768]` | 2 |
| `structure_head.diffusion_module.conditioning.s_transitions.{i}.norm.weight` | `[768]` | 2 |
| `structure_head.diffusion_module.conditioning.s_transitions.{i}.out_proj.weight` | `[768, 1536]` | 2 |
| `structure_head.diffusion_module.conditioning.z_input_norm.bias` | `[512]` | 1 |
| `structure_head.diffusion_module.conditioning.z_input_norm.weight` | `[512]` | 1 |
| `structure_head.diffusion_module.conditioning.z_proj.weight` | `[256, 512]` | 1 |
| `structure_head.diffusion_module.conditioning.z_transitions.{i}.a_proj.weight` | `[512, 256]` | 2 |
| `structure_head.diffusion_module.conditioning.z_transitions.{i}.b_proj.weight` | `[512, 256]` | 2 |
| `structure_head.diffusion_module.conditioning.z_transitions.{i}.norm.bias` | `[256]` | 2 |
| `structure_head.diffusion_module.conditioning.z_transitions.{i}.norm.weight` | `[256]` | 2 |
| `structure_head.diffusion_module.conditioning.z_transitions.{i}.out_proj.weight` | `[256, 512]` | 2 |
| `structure_head.diffusion_module.s_step_norm.bias` | `[768]` | 1 |
| `structure_head.diffusion_module.s_step_norm.weight` | `[768]` | 1 |
| `structure_head.diffusion_module.s_to_token.weight` | `[768, 768]` | 1 |
| `structure_head.diffusion_module.token_norm.bias` | `[768]` | 1 |
| `structure_head.diffusion_module.token_norm.weight` | `[768]` | 1 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.adaln.s_gate.bias` | `[768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.adaln.s_gate.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.adaln.s_scale` | `[768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.adaln.s_shift.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.g_proj.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.kv_proj.weight` | `[1536, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.out_gate.bias` | `[768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.out_gate.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.out_proj.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.pair_bias_proj.weight` | `[16, 256]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.pair_norm.bias` | `[256]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.pair_norm.weight` | `[256]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.q_proj.bias` | `[768]` | 12 |
| `structure_head.diffusion_module.token_transformer.attn_blocks.{i}.q_proj.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.adaln.s_gate.bias` | `[768]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.adaln.s_gate.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.adaln.s_scale` | `[768]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.adaln.s_shift.weight` | `[768, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.lin_out.weight` | `[768, 1536]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.lin_swish.weight` | `[3072, 768]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.output_gate.bias` | `[768]` | 12 |
| `structure_head.diffusion_module.token_transformer.transition_blocks.{i}.output_gate.weight` | `[768, 768]` | 12 |

### `token_bonds` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `token_bonds.weight` | `[256, 1]` | 1 |

### `z_init_1` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `z_init_1.weight` | `[256, 451]` | 1 |

### `z_init_2` — 1 tensors

| path pattern | shape | count |
|---|---|---|
| `z_init_2.weight` | `[256, 451]` | 1 |

