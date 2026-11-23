# Crates

## [ferritin](./ferritin)

Facade crate exposing `ferritin-core` as `ferritin::core`. Its default
features, `mesh` (`ferritin-bevy`) and `plms` (`ferritin-plms`), pull in Bevy
and Candle (about 800 crates in `cargo tree`); build with
`--no-default-features` for the core data model alone (about 20).

## [ferritin-core](./ferritin-core)

Core functionality and shared components for ferritin: structure IO, the
layered model, and selections.

## [ferritin-molviewspec](./ferritin-molviewspec)

Serde types for building and parsing [MolViewSpec](https://molstar.org/mol-view-spec/)
(`.mvsj`) trees.

## [ferritin-bevy](./ferritin-bevy)

Bevy game engine integration for ferritin, including an MVS executor
(`mvs_executor.rs`) that renders MolViewSpec trees. It handles local
PDB/mmCIF downloads, model structures, transforms, all built-in component
selectors and label/auth selector expressions, ball-and-stick, cartoon, line,
putty and spacefill representations, colors and color themes, labels, focus,
camera, canvas, and sphere/line primitives. It reports the following as
unsupported and continues: HTTP downloads, BCIF, assembly and symmetry
expansion (the asymmetric unit is drawn), `*_from_uri` / `*_from_source`
nodes, and `atom_id` selectors. Surfaces render as spacefill, and tooltip and
volume nodes are not rendered.

## [ferritin-plms](./ferritin-plms)

PLMs (Protein Language Models) integration for ferritin. The support matrix in
`src/lib.rs` records which models have Python parity fixtures.

## [ferritin-cellscape](./ferritin-cellscape)

Cell visualization and analysis tools for ferritin.

## [ferritin-wasm](./ferritin-wasm)

Owned JavaScript typed-array exports of ferritin model columns.

## [ferritin-test-data](./ferritin-test-data)

Test data and fixtures for ferritin.

## [ferritin-examples](./ferritin-examples)

Example usage and demonstrations of ferritin functionality.
