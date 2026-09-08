# ESMFold2: deleting the port, and what a real one would take

**Status:** accepted — port deleted 2026-09-07 (ferritin-100.17)
**Supersedes:** [esmfold2-port-mismatch.md](./esmfold2-port-mismatch.md)
**Ground truth:** [esmfold2-checkpoint-tensors.md](./esmfold2-checkpoint-tensors.md)
  (1032 tensors) and `biohub/ESMFold2-Fast/config.json`

## What was deleted, and why

`crates/ferritin-plms/src/esmfold2/` — 3305 lines across 13 files, plus its
runner, example, tests and registry row.

It implemented a different network from the one that was released. Not one of
the checkpoint's 1032 tensors resolved to a parameter of the ported model. The
earlier decision (ferritin-100.16) was to make it refuse to load rather than
half-load, which was right at the time: half-loading would have produced
plausible, wrong structures. But refusing left 3305 lines in the tree that no
test could reach and no caller could use, describing an architecture that does
not exist. A registry row promising a model that can never load is a promise
the crate cannot keep.

Deleting loses nothing recoverable: a correct implementation shares no code
with it. The tensor inventory is kept, because *that* is the reusable artifact.

## What ESMFold2-Fast actually is

An AlphaFold3-class system, not an ESMFold-class one. ESMFold (v1) was a
language model plus a folding trunk plus a structure module. ESMFold2 is a
**diffusion** model over atom coordinates with a pairformer trunk, an
iterative pair-refinement stage, and a multi-head confidence stack.

From `config.json`:

| component | config | tensors |
|---|---|---|
| `language_model` | `esmc_id: biohub/ESMC-6B`, `lm_d_model: 2560`, `lm_num_layers: 80` | 12 |
| `lm_encoder` | `n_layers: 4` | 72 |
| `inputs_embedder` | atom encoder: `d_atom 128`, `d_token 768`, `n_blocks 3`, `n_heads 4`, sliding-window `swa_window_size 128`, spatial + uid RoPE | 22 |
| `folding_trunk` | `n_layers: 24`, `n_heads: 8`, `d_pair: 256`, `d_single: 384` | 432 |
| `parcae` | `coda_n_layers: 2`, `min_steps: 1`, `max_steps: 6`, `poisson_mean: 3.0` | 42 |
| `structure_head` | diffusion: `token_num_blocks 12`, `token_num_heads 16`, `atom_num_blocks 3`, `c_token 768`, `c_z 256`, `sigma_data 16.0`, `inference_num_steps 14` | 345 |
| `confidence_head` | `n_layers 4`, `n_heads 8`, pLDDT/PAE/PDE heads (50/64/64 bins) | 101 |
| distogram, rel_pos, token_bonds, z_init | `n_relative_residx_bins 32` | 6 |

`num_loops: 3` is recycling. `msa_encoder.enabled: false`, so the MSA stack is
dead weight in this checkpoint and can be skipped entirely.

### The three components with no precedent in this crate

1. **Triangle multiplication.** `tri_mul_in` / `tri_mul_out` with
   `_engine.{norm_start, norm_mix, proj_bundle, proj_emit, proj_gate}` — a
   fused-projection variant of AlphaFold's triangular update. 432 of the
   folding trunk's tensors are this plus pair transitions. Nothing in
   `ferritin-plms` operates on a pair representation at all.

2. **`parcae`.** A diagonal linear recurrence over the 256 pair channels:
   `a = exp(log_a)`, `delta = exp(log_delta)`, continuous-time input matrix
   `b_cont [256,256]`, discretised in the S4/ZOH style — but with **no
   selectivity** (Δ is not input-dependent, unlike Mamba/S6), and iterated a
   *random* number of steps at training time (Poisson mean 3, clamped to
   1..6), followed by a 2-block pairformer coda. Inference step count must be
   pinned and documented; it is a free parameter that changes the output.

3. **A diffusion sampler.** `structure_head` is 345 tensors — a third of the
   model — and its output is produced by 14 denoising steps with
   `step_scale 1.5`, `noise_scale 1.003`, `gamma_0 0.8`, sigma schedule from
   `s_min 4e-4` to `s_max 160`. `num_diffusion_samples: 32` means a real
   prediction is 32 sampled structures. This is stochastic: reproducibility
   requires seeding, and "parity" against a reference means matching a
   trajectory, not a single tensor.

## The honest cost

Roughly AlphaFold3-scale. Beyond the three components above:

- **It cannot run standalone.** `esmc_id: biohub/ESMC-6B` — the trunk consumes
  2560-dim embeddings from an 80-layer, ~12 GB backbone. Any end-to-end test
  needs that loaded first. This alone puts it outside CI.
- **Atom-level, not residue-level.** The inputs embedder and diffusion module
  both operate on atoms with sliding-window attention and two RoPE schemes.
  `ferritin-core` has the structures for this; `ferritin-plms` has no
  atom-level attention anywhere.
- **No reference to check against.** Every port in this crate so far was
  validated by loading real weights and, where possible, a fixture. A
  diffusion sampler's correctness is not visible in tensor shapes — a wrong
  sigma schedule or a transposed triangular update yields plausible
  coordinates. Without a Python reference producing seeded trajectories,
  a port cannot be shown correct at all.

## Recommended sequencing, if it is attempted

Do **not** start at `ESMFold2Model::forward`. Build bottom-up, each phase
loading its own slice of the real checkpoint and testable in isolation:

1. **Pair primitives.** Triangle multiplication in/out and pair transition,
   loading `folding_trunk.blocks.0.*`. Verifiable immediately: shapes, and
   symmetry properties of the triangular update.
2. **Folding trunk.** 24 blocks + `rel_pos`, `z_init_1/2`, `token_bonds`.
   Needs single and pair representations wired but not the LM — feed random
   embeddings of the right width.
3. **`parcae`.** Small (42 tensors) and self-contained once the pair
   representation exists. Pin the inference step count explicitly.
4. **Inputs embedder.** Atom-level encoder, 22 tensors, but requires the atom
   attention machinery that does not exist yet.
5. **Confidence head.** 101 tensors, independent of the diffusion sampler,
   and produces something checkable — pLDDT on a known structure should
   correlate with its resolution.
6. **Structure head.** Largest and last, and only worth starting once a
   Python reference harness exists to compare seeded trajectories.
7. **LM wiring.** ESMC-6B integration; leave until everything else loads.

Phases 1–3 and 5 are checkable without ESMC-6B and without a sampler, which is
what makes this tractable at all.

## Recommendation

**Do not start this without a reason beyond completeness.** ESMFold2 is
popular (`biohub/ESMFold2-Experimental-Fast` has ~626k downloads), so the pull
is real, but the crate's comparative advantage is fast, verifiable *embedding*
models — and this is a stochastic structure predictor whose smallest
end-to-end test needs 12 GB of separate weights. If structure prediction is
wanted, the honest question is whether to reach for it through this port at
all, rather than which phase to begin with.
