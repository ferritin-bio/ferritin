# ferritin-wasm

A minimal wasm-bindgen consumer of `ferritin_core::columns`. No WASM
dependencies are added to core. Build JavaScript glue (Node example):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126 --locked
cargo build -p ferritin-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/ferritin_wasm.wasm \
  --target nodejs --out-dir /tmp/ferritin-wasm
```

```js
const { loadColumns } = require('/tmp/ferritin-wasm/ferritin_wasm.js');
const { topology, frame } = loadColumns(require('node:fs').readFileSync('model.cif', 'utf8'));
console.log(frame.columns.x); // Float32Array, coordinates in Å
const field = 'atom.name';
const { value_bytes, value_offsets } = topology.dictionaries[field];
const code = topology.columns[field][0];
console.log(new TextDecoder().decode(
  value_bytes.subarray(value_offsets[code], value_offsets[code + 1])
));
```

Use `--target web` instead for browsers and await the generated module's default
initializer before calling `loadColumns`.

## Contract

`loadColumns(cif: string)` parses a representative mmCIF model directly into
`Model` (lowest model number, as in core), preserving label/auth identifiers.
PDB and trajectories are not supported. It returns plain objects:

- `frame`: `n_atoms`, `columns`. Keys are `x`, `y`, `z` and present optional
  `occupancy`, `b_iso`, `confidence`; absent optional columns are omitted.
- `topology`: `n_atoms`, `n_residues`, `n_chains`, `n_bonds`, `columns`,
  `dictionaries`. All 20 stable core topology column names are retained,
  including segment/bond CSR offsets (with their terminal sentinel).
- `dictionaries`: keyed by the seven string column names. Each contains
  `value_bytes: Uint8Array` (UTF-8) and `value_offsets: Uint32Array`.
  Per-row codes are in `topology.columns[name]: Uint32Array`.

Every array has a **new JS-owned ArrayBuffer**: no borrowed wasm-memory views.
Arrays remain valid after model destruction, later calls and wasm memory growth;
JS writes cannot mutate core storage or a separately exported snapshot.
Scalar types are preserved (`u8/i8/u32/i32/f32` -> the corresponding typed array).
Core encodings are unchanged: element = atomic number, alt_loc/ins_code = Unicode
scalar (0 absent), charge = signed i8 (-128 absent), missing sequence id =
-2147483648, residue group = 0 polymer / 1 non-polymer. Offsets and handles are
zero-based. Counts and dictionary buffers are included rather than discarded.

Failures throw a JS `Error` with numeric `code`: 1 invalid model (also parse
failures), 2 out of range, 3 stale topology, 4 capacity exceeded. The Rust helper
`boundary_error_to_js` retains the core code/message. Rust consumers may also use
`copy_frame(FrameColumns)` and `copy_topology(&TopologyColumns)`.

## Runtime tests (Node)

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p ferritin-wasm --target wasm32-unknown-unknown
```

These are real wasm-bindgen runtime tests, not native stubs: they inspect JS
typed-array classes, decode all dictionaries, verify codes on JS Errors, mutate
copies on both sides, drop source data, and grow wasm memory explicitly.
