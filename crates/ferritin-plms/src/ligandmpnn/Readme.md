# MPNN ports

Rust ports of ProteinMPNN and LigandMPNN, transcribed from
[dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN) — whose
`model_utils.py` carries all three architectures (ProteinMPNN, LigandMPNN,
SolubleMPNN), not from the older `dauparas/ProteinMPNN` repo.

## What is here

| File | Reference |
|---|---|
| `model.rs` | `ProteinMPNN`, `EncLayer`, `DecLayer`, `DecLayerJ` |
| `proteinfeaturesmodel.rs` | `ProteinFeatures` — protein edges only |
| `ligandfeaturesmodel.rs` | `ProteinFeaturesLigand` — edges plus ligand context |
| `pmpnn_runner.rs` | loading, both checkpoints |
| `configs.rs` | architecture configs and the CLI surface |

Both checkpoints run through the same `ProteinMPNN` struct. LigandMPNN swaps
the featurizer and adds a ligand context path — `W_v`, `W_c`, `W_nodes_y`,
`W_edges_y`, `V_C`, and two rounds each of `context_encoder_layers` and
`y_context_encoder_layers` — folded into `h_V` before the decoder runs.

## Parity

Both are checked against the Python reference on 1BC8, a zinc-finger/DNA
complex whose 406 ligand atoms make it a real test of the ligand path rather
than an all-padding one:

```sh
cargo test -p ferritin-plms --test test_plm_ligandmpnn
```

The fixtures come from `scripts/generate_mpnn_fixtures.py`; see its docstring
for the Python setup. They pin the **structure-only** forward pass, not
`score()`, whose decoding order is drawn from `randn` and is not reproducible.

Measured agreement is KL < 1e-6 against the reference distribution — f32
round-off. Before ferritin-100.11 the ProteinMPNN port agreed with the
reference on 2 of 93 positions; see that issue for the six defects, and
ferritin-100.33 for the release consequences.

## Checkpoint facts that are not in the tensors

`num_edges` (k-nearest neighbours) and `atom_context_num` live in the `.pt` as
plain Python ints, which candle's pickle reader does not surface, and they do
not appear in any weight shape. Loading a checkpoint against the wrong one
therefore succeeds and computes the wrong thing, so they are declared per
variant in `ProteinMPNNConfig::proteinmpnn()` / `::ligandmpnn()` and covered by
`test_model_types_carry_distinct_configs`.

## Status of the CLI

The CLI equivalent was abandoned in December 2024. The library API
(`ProteinMPNNRunner`) is the supported surface.

Its test suite — `tests/test_cli_ligandmpnn.rs`, 33 tests transcribed from
upstream's [`run_examples.sh`](https://github.com/dauparas/LigandMPNN/blob/main/run_examples.sh)
— was deleted in ferritin-100.32. It had been 806 lines with every one of them
line-commented since 2024-12-16, so it compiled to zero tests while reading
from the outside like real coverage; ferritin-100.11 cited it as "33 tests, all
ignored". It could not have run in any case: `ferritin-plms` declares no
`[[bin]]` for `Command::cargo_bin` to find, and neither `assert_cmd` nor
`tempfile` is a dev-dependency, so the file would not even compile if
uncommented.

To resurrect it, take the upstream `run_examples.sh` as the source of truth
rather than the deleted file, and add the CLI and dev-dependencies it assumed.

## Resources

- [Candle](https://github.com/huggingface/candle)
- [Candle Tutorial](https://github.com/ToluClassics/candle-tutorial)
