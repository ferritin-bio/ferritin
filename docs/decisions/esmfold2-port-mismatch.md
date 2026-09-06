# ESMFold2: the ported architecture does not match the released checkpoint

**Status:** accepted — `ESMFold2Runner::from_pretrained*` refuses to load (ferritin-100.16)
**Date:** 2026-09-05

## Summary

The `ferritin-plms` ESMFold2 port cannot load `biohub/ESMFold2-Fast`
`model.safetensors`. This was originally filed as a weight-key-path mismatch —
a wrong prefix or a rename. It is not. **Not one of the checkpoint's 1032
tensors resolves to a parameter of the ported model**, because the ported
module graph is a different network from the one that was released.

The port's *scalar* hyper-parameters are mostly right (`d_single=384`,
`d_pair=256`, `d_inputs=451`, `c_token=768`, `c_atom=128`, 24 trunk blocks,
4 LM-encoder blocks, 12 token blocks, 3 atom blocks, `fourier_dim=256`).
The *module layouts* were evidently written from an architecture sketch rather
than read off the checkpoint.

## Evidence

| Component | What the port loads | What the checkpoint contains |
|---|---|---|
| `lm_encoder` | 4 pre-norm **transformer** blocks over `d=2560` (`attn.layernorm_qkv.{0,1}`, `attn.out_proj`, `attn.q_ln`/`k_ln`, `ffn.{0,1,3}`), then `norm` + `proj` to 384 | 4 **pair-track** blocks at `d=256`: `tri_mul_in`/`tri_mul_out` (`_engine.{norm_start,norm_mix,proj_bundle,proj_emit,proj_gate}`) + `pair_transition.{norm,ffn.w12,ffn.w3}`. No attention, no `norm`, no `proj` |
| `folding_trunk` | `pair_init.*`, then blocks of `tri_attn_row`, `tri_attn_col`, `tri_mult_out`, `tri_mult_in`, `pair_trans`, then `norm` | 24 blocks of `tri_mul_in`/`tri_mul_out`/`pair_transition` **only**. No triangle attention, no `pair_init`, no trailing `norm` |
| `structure_head` | flat `token_proj`, `noise_embedding`, `noise_proj`, `token_transformer.blocks.{i}.{attn,ffn}`, `out_proj` | one more nesting level (`structure_head.diffusion_module.*`), `conditioning.*`, `atom_encoder`/`atom_decoder` with `atom_transformer`, and a token transformer split into separate `attn_blocks` and `transition_blocks` with AdaLN modulation |
| `confidence_head` | `trunk.blocks.{i}`, plus `plddt_head`/`pae_head`/`pde_head`/`distogram_head` linears | its own 4-block `folding_trunk`, `plddt_weight` as a rank-3 `[23, 384, 50]` tensor (not a `Linear`), `s_to_z*`, `row_attention_pooling`, `boundaries`, `dist_bin_pairwise_embed` |
| — | not modelled at all | `parcae_coda.*`, `parcae_log_a`, `parcae_log_delta`, `parcae_b_cont`, `parcae_readout`, `parcae_input_norm` (a selective-SSM-style block), `language_model.base_z_*`, `inputs_embedder.atom_attention_encoder.*`, and top-level `rel_pos`, `token_bonds`, `z_init_1`, `z_init_2`, `distogram_head` |

Two smaller discrepancies worth noting: the checkpoint's top-level
`distogram_head.weight` is `[64, 256]` (64 bins), while `ESMFold2Config`
declares `distogram_bins: 39` — 39 is the size of
`confidence_head.dist_bin_pairwise_embed`, so the two were transposed.

The complete tensor inventory is in
[`esmfold2-checkpoint-tensors.md`](./esmfold2-checkpoint-tensors.md).

## Decision

`from_pretrained` and `from_pretrained_with_backbone` **refuse to load**, with
an error that names the real cause, rather than surfacing a
`cannot find tensor lm_encoder.blocks.0.attn.layernorm_qkv.0.weight` that
invites another round of prefix-patching.

This follows the precedent set for `ESMFold2Models::Full` (ferritin-100.4):
when a checkpoint cannot be loaded faithfully, refusing is the honest failure.

### Why not "just fix the paths"

Renaming load paths would let tensors resolve but would not make the forward
pass correct — the port has no triangle-multiplication engine of the released
shape, no AdaLN diffusion transformer, no atom decoder, and no `parcae` block
at all. Patching until `load` returns `Ok` would convert a loud, obvious
failure into a silent one that emits plausible-looking but meaningless
coordinates. That is strictly worse, and it is close to how the current state
arose.

### What stays

The layer implementations and their shape tests are untouched and still pass;
they exercise the modules under `VarBuilder::zeros`. Only the pretrained
loading path refuses.

## Follow-up

A faithful re-port is tracked separately. It needs the reference
implementation (or reference activations) to validate against — the checkpoint
gives us exact shapes but not the semantics of `proj_bundle`/`proj_emit`, the
`parcae_*` block, or the construction of the 451-dim input features, the
139-bin `rel_pos` embedding, and the 389/6-dim atom features. Guessing those a
second time would reproduce this bug.
